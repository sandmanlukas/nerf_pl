import os
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import *
from models.rendering import *

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning import loggers as pl_loggers

# transforms
from torchvision import transforms as T


import glob


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(vars(hparams))

        self.validation_step_loss = []
        self.validation_step_psnr = []

        self.loss = loss_dict["nerfw"](coef=1)

        self.models_to_train = []

        if hparams.test and not hparams.ckpt_path:
            raise ValueError("Must provide ckpt_path when training on test set")

        self.embedding_xyz = PosEmbedding(hparams.N_emb_xyz - 1, hparams.N_emb_xyz)
        self.embedding_dir = PosEmbedding(hparams.N_emb_dir - 1, hparams.N_emb_dir)
        self.embeddings = {"xyz": self.embedding_xyz, "dir": self.embedding_dir}

        if hparams.encode_a:
            self.embedding_a = torch.nn.Embedding(hparams.N_vocab, hparams.N_a)
            self.embeddings["a"] = self.embedding_a
            self.models_to_train += [self.embedding_a]

        if hparams.encode_t:
            self.embedding_t = torch.nn.Embedding(hparams.N_vocab, hparams.N_tau)
            self.embeddings["t"] = self.embedding_t
            self.models_to_train += [self.embedding_t]

            if hparams.test:
                load_ckpt(self.embeddings["t"], hparams.ckpt_path, model_name="embedding_t")

        self.nerf_coarse = NeRF(
            "coarse",
            in_channels_xyz=6 * hparams.N_emb_xyz + 3,
            in_channels_dir=6 * hparams.N_emb_dir + 3,
            encode_appearance=hparams.encode_a,
        )

        if hparams.test:
            load_ckpt(self.nerf_coarse, hparams.ckpt_path, model_name="nerf_coarse")

        self.models = {"coarse": self.nerf_coarse}
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF(
                "fine",
                in_channels_xyz=6 * hparams.N_emb_xyz + 3,
                in_channels_dir=6 * hparams.N_emb_dir + 3,
                encode_appearance=hparams.encode_a,
                in_channels_a=hparams.N_a,
                encode_transient=hparams.encode_t,
                in_channels_t=hparams.N_tau,
                beta_min=hparams.beta_min,
            )

            if hparams.test:
                load_ckpt(self.nerf_fine, hparams.ckpt_path, model_name="nerf_fine")

            self.models["fine"] = self.nerf_fine
        self.models_to_train += [self.models]

        # Freeze all params except appearance embeddings
        # appearance embeddings are at index 0 in self.models_to_train
        if hparams.test and hparams.encode_a:
            params = get_parameters(self.models_to_train[1:])
            for param in params:
                param.requires_grad = False

    def define_transforms(self):
        self.transform = T.ToTensor()

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, rays, ts):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)

        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = render_rays(
                self.models,
                self.embeddings,
                rays[i : i + self.hparams.chunk],
                ts[i : i + self.hparams.chunk],
                self.hparams.N_samples,
                self.hparams.use_disp,
                self.hparams.perturb,
                self.hparams.noise_std,
                self.hparams.N_importance,
                self.hparams.chunk,  # chunk size is effective in val mode
                self.train_dataset.white_back,
            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {"root_dir": self.hparams.root_dir}

        if self.hparams.dataset_name == "phototourism":
            kwargs["img_downscale"] = self.hparams.img_downscale
            kwargs["val_num"] = self.hparams.num_gpus
            kwargs["use_cache"] = self.hparams.use_cache
            kwargs["exp_name"] = self.hparams.exp_name
            kwargs["mask_path"] = self.hparams.mask_path
            kwargs["tsv_file"] = self.hparams.tsv_file
        elif self.hparams.dataset_name == "blender":
            kwargs["img_wh"] = tuple(self.hparams.img_wh)
            kwargs["perturbation"] = self.hparams.data_perturb

        if hparams.test:
            self.train_dataset = dataset(split="test", **kwargs)
            self.val_dataset = dataset(split="val_test", **kwargs)
        else:
            self.train_dataset = dataset(split="train", **kwargs)
            self.val_dataset = dataset(split="val", **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models_to_train)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=8,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=8,
            batch_size=1,  # Validate one image (H*W rays) at a time
            pin_memory=True,
        )

    def training_step(self, batch, batch_nb):
        if hparams.test:
            rays, rgbs, ts = batch["rays_bottom"], batch["rgbs_bottom"], batch["ts_bottom"]
            mask = batch["mask_bottom"] if "mask_bottom" in batch else None
        else:
            rays, rgbs, ts = batch["rays"], batch["rgbs"], batch["ts"]
            mask = batch["mask"] if "mask" in batch else None

        results = self.forward(rays, ts)

        if mask is not None:
            mask = mask.cuda()
            for k, v in results.items():
                results[k] = v * (
                    mask.view(v.shape)
                    if len(v.shape) == 1
                    else mask[:, None]
                    if len(v.shape) > 1 and len(mask.shape) == 1
                    else mask
                )

                if k == "beta":
                    # Add beta_min to beta to prevent NaN in loss calcs
                    results[k] = torch.where(
                        results[k] == 0, self.hparams.beta_min, results[k]
                    )

        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())

        with torch.no_grad():
            typ = "fine" if "rgb_fine" in results else "coarse"
            psnr_ = psnr(results[f"rgb_{typ}"], rgbs)
            # valid_mask = mask != 0 if mask is not None else None
            # psnr_unmasked = psnr(results[f"rgb_{typ}"], rgbs, valid_mask=valid_mask)

        self.log("lr", get_learning_rate(self.optimizer))
        self.log("train/loss", loss)
        for k, v in loss_d.items():
            self.log(f"train/{k}", v, prog_bar=True)
        self.log("train/psnr", psnr_, prog_bar=True)
        # self.log("train/psnr_unmasked", psnr_unmasked, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        rays, rgbs, ts = batch["rays"], batch["rgbs"], batch["ts"]
        mask = batch["mask"] if "mask" in batch else None

        if hparams.test:
            # get width/height
            img_w = batch["img_wh"][0, 0].item()
            img_h = batch["img_wh"][0, 1].item()

            # reshape to (1, H, W, channels)
            rays = rays.view(1, img_h, img_w, -1)
            rgbs = rgbs.view(1, img_h, img_w, -1)
            ts = ts.view(1, img_h, img_w, -1)

            # split into two, train/val on bottom half
            _, rays = torch.split(rays, [img_h // 2, img_h // 2], dim=1)
            _, rgbs = torch.split(rgbs, [img_h // 2, img_h // 2], dim=1)
            _, ts = torch.split(ts, [img_h // 2, img_h // 2], dim=1)

            # reshape back to original shape
            rays = rays.reshape(rays.shape[0], -1, rays.shape[-1])
            rgbs = rgbs.reshape(rgbs.shape[0], -1, rgbs.shape[-1])
            ts = ts.reshape(ts.shape[0], -1, ts.shape[-1])

            if mask is not None:
                mask = mask.view(1, img_h, img_w, -1)
                _, mask = torch.split(mask, [img_h // 2, img_h // 2], dim=1)
                mask = mask.reshape(mask.shape[0], -1, mask.shape[-1])

        rays = rays.squeeze()  # (H*W, 8)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        ts = ts.squeeze()  # (H*W)
        results = self(rays, ts)

        if mask is not None:
            mask = mask.squeeze()  # (H*W)
            mask = mask.flatten()[:, None].cuda()

            # Add mask to results
            for k, v in results.items():
                results[k] = v * (
                    mask.view(v.shape)
                    if len(v.shape) == 1
                    else mask[:, None]
                    if len(v.shape) > 1 and len(mask.shape) == 1
                    else mask
                )

                # Add beta_min to beta to prevent NaN in loss calcs
                if k == "beta":
                    results[k] = torch.where(
                        results[k] == 0, self.hparams.beta_min, results[k]
                    )

        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())
        self.validation_step_loss.append(loss)
        log = {"val_loss": loss}
        typ = "fine" if "rgb_fine" in results else "coarse"

        if batch_nb == 0:
            if self.hparams.dataset_name == "phototourism":
                WH = batch["img_wh"]
                W, H = WH[0, 0].item(), WH[0, 1].item()
                
                if hparams.test:
                    H = H // 2
            else:
                W, H = self.hparams.img_wh
            img = (
                results[f"rgb_{typ}"].view(H, W, 3).permute(2, 0, 1).cpu()
            )  # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
            depth = visualize_depth(
                results[f"depth_{typ}"].view(H, W).cpu()
            )  # (3, H, W)
            stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)
            self.logger.experiment.add_images(
                "val/GT_pred_depth", stack, self.global_step
            )

        psnr_ = psnr(results[f"rgb_{typ}"], rgbs)
        with torch.no_grad():
            valid_mask = (mask != 0).repeat(1, 3) if mask is not None else None
            psnr_unmasked = psnr(results[f"rgb_{typ}"], rgbs, valid_mask=valid_mask)
            log["val_psnr_unmasked"] = psnr_unmasked

        self.validation_step_psnr.append(psnr_)
        log["val_psnr"] = psnr_

        return log

    def on_validation_epoch_end(self):
        mean_loss = torch.stack(self.validation_step_loss).mean()
        mean_psnr = torch.stack(self.validation_step_psnr).mean()

        self.validation_step_loss.clear()
        self.validation_step_psnr.clear()

        self.log("val/loss", mean_loss)
        self.log("val/psnr", mean_psnr, prog_bar=True)


def main(hparams):
    system = NeRFSystem(hparams)

    ckpt_callback_dir = os.path.join("ckpts", f"{hparams.exp_name}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_callback_dir,
        monitor="val/psnr",
        mode="max",
        save_top_k=-1,
        save_last=True,
    )

    # get last version number and increment
    version_num = "0"
    if os.path.isdir(os.path.join("./logs", hparams.exp_name)):
        version_list = sorted(
            [
                int(item.split("_")[-1])
                for item in os.listdir(os.path.join("./logs", hparams.exp_name))
                if os.path.isdir(os.path.join("./logs", hparams.exp_name, item))
                and item != "summaries"
            ]
        )
        version_num = str(version_list[-1] + 1) if version_list else "0"

    tensorboard = pl_loggers.TensorBoardLogger(
        save_dir="logs",
        name=hparams.exp_name,
        version=f"{hparams.exp_name}_{version_num}",
    )

    trainer = Trainer(
        max_epochs=hparams.num_epochs,
        callbacks=[checkpoint_callback],
        logger=tensorboard,
        num_sanity_val_steps=1,
        benchmark=True,
        profiler="simple" if hparams.num_gpus == 1 else None,
    )

    last_ckpt = glob.glob(os.path.join("ckpts", f"{hparams.exp_name}", "last.ckpt"))

    if hparams.ckpt_path and not hparams.test:
        ckpt_path = hparams.ckpt_path
    else:
        ckpt_path = last_ckpt[0] if last_ckpt else None

    trainer.fit(system, ckpt_path=ckpt_path)


if __name__ == "__main__":
    hparams = get_opts()
    main(hparams)
