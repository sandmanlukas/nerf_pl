import csv
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image
from torchvision import transforms as T
import cv2

from .ray_utils import *
from .colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary


class PhototourismDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        img_downscale=1,
        val_num=1,
        use_cache=False,
        exp_name="",
        white_back=False,
        mask_path="",
        **kwargs,
    ):
        """
        img_downscale: how much scale to downsample the training images.
                       The original image sizes are around 500~100, so value of 1 or 2
                       are recommended.
                       ATTENTION! Value of 1 will consume large CPU memory,
                       about 40G for brandenburg gate.
        val_num: number of val images (used for multigpu, validate same image for all gpus)
        use_cache: during data preparation, use precomputed rays (useful to accelerate
                   data loading, especially for multigpu!)
        """
        self.root_dir = root_dir
        self.split = split

        self.exp_name = exp_name

        self.mask_path = mask_path
        self.mask = torch.tensor([])

        self.tsv_file = kwargs.get("tsv_file", "")

        # testing
        self.poses_test = []
        self.test_K = []
        self.test_img_h = None
        self.test_img_w = None
        self.test_appearance_idx = None

        assert (
            img_downscale >= 1
        ), "image can only be downsampled, please set img_downscale>=1!"

        self.img_downscale = img_downscale

        if split == "val":  # image downscale=1 will cause OOM in val mode
            self.img_downscale = max(2, self.img_downscale)

        self.val_num = max(1, val_num)  # at least 1
        self.use_cache = use_cache
        self.define_transforms()

        self.read_meta()
        self.white_back = white_back

    def create_tsv(self):
        images = sorted(glob.glob(os.path.join(self.root_dir, "images/*.png")))

        test_holdout = 8
        test_images = images[::test_holdout]

        print(f"Creating {os.path.join(self.root_dir, self.exp_name)}.tsv...")
        with open(
            os.path.join(self.root_dir, "tsvs", f"{self.exp_name}.tsv"), "w"
        ) as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["filename", "id", "split", "dataset"])

            for image in images:
                filename = image[-9:]
                id = filename[:5]
                split = "test" if image in test_images else "train"
                dataset = self.exp_name
                row = [filename, id, split, dataset]
                writer.writerow(row)

    def read_meta(self):
        tsv_files = [
            os.path.basename(x)
            for x in glob.glob(os.path.join(self.root_dir, "tsvs", "*.tsv"))
        ]

        dataset_tsv = self.tsv_file if self.tsv_file else f"{self.exp_name}.tsv"
        if not dataset_tsv in tsv_files:
            self.create_tsv()

        # read all files in the tsv first (split to train and test later)
        self.scene_name = self.exp_name
        self.files = pd.read_csv(
            os.path.join(self.root_dir, "tsvs", dataset_tsv), sep="\t"
        )
        self.files = self.files[~self.files["id"].isnull()]  # remove data without id
        self.files.reset_index(inplace=True, drop=True)

        # Step 1. load image paths
        # Attention! The 'id' column in the tsv is BROKEN, don't use it!!!!
        # Instead, read the id from images.bin using image file name!
        if self.use_cache:
            with open(os.path.join(self.root_dir, f"cache/img_ids.pkl"), "rb") as f:
                self.img_ids = pickle.load(f)
            with open(os.path.join(self.root_dir, f"cache/image_paths.pkl"), "rb") as f:
                self.image_paths = pickle.load(f)
        else:
            imdata = read_images_binary(
                os.path.join(self.root_dir, "sparse/0/images.bin")
            )
            img_path_to_id = {}
            self.image_to_cam = {}
            for v in imdata.values():
                img_path_to_id[v.name] = v.id
                self.image_to_cam[v.id] = v.camera_id
            self.img_ids = []
            self.image_paths = {}  # {id: filename}
            for filename in list(self.files["filename"]):
                id_ = img_path_to_id[filename]
                self.image_paths[id_] = filename
                self.img_ids += [id_]

        # Step 2: read and rescale camera intrinsics
        if self.use_cache:
            with open(
                os.path.join(self.root_dir, f"cache/Ks{self.img_downscale}.pkl"), "rb"
            ) as f:
                self.Ks = pickle.load(f)
        else:
            self.Ks = {}  # {id: K}
            camdata = read_cameras_binary(
                os.path.join(self.root_dir, "sparse/0/cameras.bin")
            )
            for id_ in self.img_ids:
                K = np.zeros((3, 3), dtype=np.float32)
                cam_id = self.image_to_cam[id_]
                cam = camdata[cam_id]
                img_w, img_h = int(cam.width), int(cam.height)
                img_w_, img_h_ = (
                    img_w // self.img_downscale,
                    img_h // self.img_downscale,
                )

                if cam.model in [
                    "OPENCV",
                    "OPENCV_FISHEYE",
                    "FULL_OPENCV",
                    "PINHOLE",
                    "FOV",
                    "THIN_PRISM",
                ]:  # radial-tangential distortion
                    K[0, 0] = cam.params[0] * img_w_ / img_w  # fx
                    K[1, 1] = cam.params[1] * img_h_ / img_h  # fy
                    K[0, 2] = cam.params[2] * img_w_ / img_w  # cx
                    K[1, 2] = cam.params[3] * img_h_ / img_h  # cy
                    K[2, 2] = 1
                elif cam.model in [
                    "RADIAL_FISHEYE",
                    "SIMPLE_PINHOLE",
                    "SIMPLE_RADIAL",
                    "SIMPLE_RADIAL_FISHEYE",
                    "RADIAL",
                ]:  # fisheye distortion
                    K[0, 0] = cam.params[0] * img_w_ / img_w  # fx
                    K[1, 1] = cam.params[0] * img_h_ / img_h  # fy
                    K[0, 2] = cam.params[1] * img_w_ / img_w  # cx
                    K[1, 2] = cam.params[2] * img_h_ / img_h  # cy
                    K[2, 2] = 1
                else:
                    raise NotImplementedError(
                        f"Camera model {cam.model} not implemented"
                    )

                self.Ks[cam_id] = K

        # Step 3: read c2w poses (of the images in tsv file only) and correct the order
        if self.use_cache:
            self.poses = np.load(os.path.join(self.root_dir, "cache/poses.npy"))
        else:
            w2c_mats = []
            bottom = np.array([0, 0, 0, 1.0]).reshape(1, 4)
            for id_ in self.img_ids:
                im = imdata[id_]
                R = im.qvec2rotmat()
                t = im.tvec.reshape(3, 1)
                w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
            w2c_mats = np.stack(w2c_mats, 0)  # (N_images, 4, 4)
            self.poses = np.linalg.inv(w2c_mats)[:, :3]  # (N_images, 3, 4)
            # Original poses has rotation in form "right down front", change to "right up back"
            self.poses[..., 1:3] *= -1

        # Step 4: correct scale
        if self.use_cache:
            self.xyz_world = np.load(os.path.join(self.root_dir, "cache/xyz_world.npy"))
            with open(os.path.join(self.root_dir, f"cache/nears.pkl"), "rb") as f:
                self.nears = pickle.load(f)
            with open(os.path.join(self.root_dir, f"cache/fars.pkl"), "rb") as f:
                self.fars = pickle.load(f)
        else:
            pts3d = read_points3d_binary(
                os.path.join(self.root_dir, "sparse/0/points3D.bin")
            )
            self.xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d])
            xyz_world_h = np.concatenate(
                [self.xyz_world, np.ones((len(self.xyz_world), 1))], -1
            )
            # Compute near and far bounds for each image individually
            self.nears, self.fars = {}, {}  # {id_: distance}
            for i, id_ in enumerate(self.img_ids):
                xyz_cam_i = (xyz_world_h @ w2c_mats[i].T)[
                    :, :3
                ]  # xyz in the ith cam coordinate
                xyz_cam_i = xyz_cam_i[
                    xyz_cam_i[:, 2] > 0
                ]  # filter out points that lie behind the cam
                self.nears[id_] = np.percentile(xyz_cam_i[:, 2], 0.1)
                self.fars[id_] = np.percentile(xyz_cam_i[:, 2], 99.9)

            max_far = np.fromiter(self.fars.values(), np.float32).max()
            scale_factor = max_far / 5  # so that the max far is scaled to 5
            self.poses[..., 3] /= scale_factor
            for k in self.nears:
                self.nears[k] /= scale_factor
            for k in self.fars:
                self.fars[k] /= scale_factor
            self.xyz_world /= scale_factor
        self.poses_dict = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids)}

        # Step 5. split the img_ids (the number of images is verfied to match that in the paper)
        self.img_ids_train = [
            id_
            for i, id_ in enumerate(self.img_ids)
            if self.files.loc[i, "split"] == "train"
        ]

        self.img_ids_test = [
            id_
            for i, id_ in enumerate(self.img_ids)
            if self.files.loc[i, "split"] == "test"
        ]

        self.img_ids_val = [
            id_
            for i, id_ in enumerate(self.img_ids)
            if self.files.loc[i, "split"] == "val"
        ]
        self.N_images_train = len(self.img_ids_train)
        self.N_images_test = len(self.img_ids_test)
        self.N_images_val = len(self.img_ids_val)

        if self.split == "train":  # create buffer of all rays and rgb data
            if self.use_cache:
                all_rays = np.load(
                    os.path.join(self.root_dir, f"cache/rays{self.img_downscale}.npy")
                )
                self.all_rays = torch.from_numpy(all_rays)
                all_rgbs = np.load(
                    os.path.join(self.root_dir, f"cache/rgbs{self.img_downscale}.npy")
                )
                self.all_rgbs = torch.from_numpy(all_rgbs)
            else:
                self.all_rays = []
                self.all_rgbs = []
                self.all_masks = []

                # If mask dir passed, create mask
                if self.mask_path:
                    mask = Image.open(self.mask_path).convert("L")
                    mask_w, mask_h = mask.size

                    if self.img_downscale > 1:
                        mask_w = mask_w // self.img_downscale
                        mask_h = mask_h // self.img_downscale
                        mask = mask.resize((mask_w, mask_h), Image.Resampling.LANCZOS)
                    mask = self.transform(mask).to(torch.uint8)

                    # mask = mask.view(3, -1).permute(1, 0) # (h*w, 1)

                    self.mask = mask

                for id_ in self.img_ids_train:
                    c2w = torch.FloatTensor(self.poses_dict[id_])

                    img = Image.open(
                        os.path.join(self.root_dir, "images", self.image_paths[id_])
                    ).convert("RGB")

                    img_w, img_h = img.size
                    if self.img_downscale > 1:
                        img_w = img_w // self.img_downscale
                        img_h = img_h // self.img_downscale
                        img = img.resize((img_w, img_h), Image.Resampling.LANCZOS)

                    img = self.transform(img)  # (3, h, w)

                    if len(self.mask) != 0:
                        img = img * self.mask  # Mask image
                        self.all_masks += [self.mask.view(1, -1).permute(1, 0)]

                    img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB

                    self.all_rgbs += [img]

                    directions = get_ray_directions(
                        img_h, img_w, self.Ks[self.image_to_cam[id_]]
                    )
                    rays_o, rays_d = get_rays(directions, c2w)
                    rays_t = id_ * torch.ones(len(rays_o), 1)

                    self.all_rays += [
                        torch.cat(
                            [
                                rays_o,
                                rays_d,
                                self.nears[id_] * torch.ones_like(rays_o[:, :1]),
                                self.fars[id_] * torch.ones_like(rays_o[:, :1]),
                                rays_t,
                            ],
                            1,
                        )
                    ]  # (h*w, 8)

                self.all_rays = torch.cat(self.all_rays, 0)  # ((N_images-1)*h*w, 8)
                self.all_rgbs = torch.cat(self.all_rgbs, 0)  # ((N_images-1)*h*w, 3)
                if self.all_masks:
                    self.all_masks = torch.cat(
                        self.all_masks, 0
                    )  # ((N_images-1)*h*w,1)

        elif self.split in [
            "val",
            "test_train",
        ]:  # use the first image as val image (also in train)
            self.val_id = self.img_ids_train[0]
        elif self.split == "val_test":
            self.val_id = self.img_ids_test[0]

        else:  # for testing, create a parametric rendering path
            self.all_rays_top = []
            self.all_rays_bottom = []

            self.all_rgbs_top = []
            self.all_rgbs_bottom = []

            self.all_masks_top = []
            self.all_masks_bottom = []

            # If mask dir passed, create mask
            if self.mask_path:
                mask = Image.open(self.mask_path).convert("L")
                mask_w, mask_h = mask.size

                if self.img_downscale > 1:
                    mask_w = mask_w // self.img_downscale
                    mask_h = mask_h // self.img_downscale
                    mask = mask.resize((mask_w, mask_h), Image.Resampling.LANCZOS)
                mask = self.transform(mask).to(torch.uint8)

                self.mask = mask

            for id_ in self.img_ids_test:
                c2w = torch.FloatTensor(self.poses_dict[id_])

                img = Image.open(
                    os.path.join(self.root_dir, "images", self.image_paths[id_])
                ).convert("RGB")

                img_w, img_h = img.size

                if self.img_downscale > 1:
                    img_w = img_w // self.img_downscale
                    img_h = img_h // self.img_downscale
                    img = img.resize((img_w, img_h), Image.Resampling.LANCZOS)

                img = self.transform(img)  # (3, h, w)

                # Split image in left and right half
                if len(self.mask) != 0:
                    img = img * self.mask  # Mask image
                    mask_top, mask_bottom = torch.split(
                        self.mask, [img_h // 2, img_h // 2], dim=1
                    )
                    # Store top side of mask
                    self.all_masks_top += [
                        mask_top.reshape(1, -1).permute(1, 0)
                    ]  # (h*w, 1) Grayscale

                    # Store bottom side of mask
                    self.all_masks_bottom += [
                        mask_bottom.reshape(1, -1).permute(1, 0)
                    ]  # (h*w, 1) Grayscale

                img_top, img_bottom = torch.split(img, [img_h // 2, img_h // 2], dim=1)
                img_top = img_top.reshape(3, -1).permute(1, 0)  # (h*w, 3) RGB
                img_bottom = img_bottom.reshape(3, -1).permute(1, 0)  # (h*w, 3) RGB

                # Store the left and right side of image
                self.all_rgbs_top += [img_top]
                self.all_rgbs_bottom += [img_bottom]

                # calculate direction of left side of image
                directions = get_ray_directions(
                    img_h, img_w, self.Ks[self.image_to_cam[id_]]
                )

                rays_o_top, rays_o_bottom, rays_d_top, rays_d_bottom = get_rays(
                    directions, c2w, test=True
                )
                rays_t_top = id_ * torch.ones(len(rays_o_top), 1)
                rays_t_bottom = id_ * torch.ones(len(rays_o_bottom), 1)

                self.all_rays_top += [
                    torch.cat(
                        [
                            rays_o_top,
                            rays_d_top,
                            self.nears[id_] * torch.ones_like(rays_o_top[:, :1]),
                            self.fars[id_] * torch.ones_like(rays_o_top[:, :1]),
                            rays_t_top,
                        ],
                        1,
                    )
                ]  # (h*w, 9)

                self.all_rays_bottom += [
                    torch.cat(
                        [
                            rays_o_bottom,
                            rays_d_bottom,
                            self.nears[id_] * torch.ones_like(rays_o_bottom[:, :1]),
                            self.fars[id_] * torch.ones_like(rays_o_bottom[:, :1]),
                            rays_t_bottom,
                        ],
                        1,
                    )
                ]  # (h*w, 9)

            self.all_rays_top = torch.cat(
                self.all_rays_top, 0
            )  # ((N_images-1)*h*w, 9)

            self.all_rays_bottom = torch.cat(
                self.all_rays_bottom, 0
            )  # ((N_images-1)*h*w, 9)

            self.all_rgbs_top = torch.cat(
                self.all_rgbs_top, 0
            )  # ((N_images-1)*h*w, 3)

            self.all_rgbs_bottom = torch.cat(
                self.all_rgbs_bottom, 0
            )  # ((N_images-1)*h*w, 3)

            if self.all_masks_top and self.all_masks_bottom:
                self.all_masks_top = torch.cat(
                    self.all_masks_top, 0
                )  # ((N_images-1)*h*w,1)

                self.all_masks_bottom = torch.cat(
                    self.all_masks_bottom, 0
                )  # ((N_images-1)*h*w,1)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == "train":
            return len(self.all_rays)
        if self.split == "test_train":
            return self.N_images_train
        if self.split in ["val", "val_test"]:
            return self.val_num
        if self.split == "test" and len(self.poses_test):
            return len(self.poses_test)
        return len(self.all_rays_top)

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            if len(self.mask) != 0:
                sample = {
                    "rays": self.all_rays[idx, :8],
                    "ts": self.all_rays[idx, 8].long(),
                    "rgbs": self.all_rgbs[idx],
                    "mask": self.all_masks[idx],
                }
            else:
                sample = {
                    "rays": self.all_rays[idx, :8],
                    "ts": self.all_rays[idx, 8].long(),
                    "rgbs": self.all_rgbs[idx],
                }

        elif self.split in ["val", "val_test", "test_train"]:
            sample = {}

            if self.split in ["val", "val_test"]:
                id_ = self.val_id
            else:
                # id_ = self.img_ids_train[idx]
                id_ = idx

            sample["c2w"] = c2w = torch.FloatTensor(self.poses_dict[id_])

            img = Image.open(
                os.path.join(self.root_dir, "images", self.image_paths[id_])
            ).convert("RGB")

            if self.mask_path:
                mask = Image.open(self.mask_path).convert("L")

            img_w, img_h = img.size
            if self.img_downscale > 1:
                img_w = img_w // self.img_downscale
                img_h = img_h // self.img_downscale
                img = img.resize((img_w, img_h), Image.Resampling.LANCZOS)
                mask = (
                    mask.resize((img_w, img_h), Image.Resampling.LANCZOS)
                    if self.mask_path
                    else torch.ones((1, img_h, img_w))
                )

            img = self.transform(img)  # (3, h, w)
            mask = (
                mask if torch.is_tensor(mask) else self.transform(mask).to(torch.uint8)
            )

            self.mask = mask

            img = img * mask

            valid_mask = (img[-1] > 0).flatten()
            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB

            sample["rgbs"] = img
            sample["mask"] = mask

            directions = get_ray_directions(
                img_h, img_w, self.Ks[self.image_to_cam[id_]]
            )
            rays_o, rays_d = get_rays(directions, c2w)

            rays = torch.cat(
                [
                    rays_o,
                    rays_d,
                    self.nears[id_] * torch.ones_like(rays_o[:, :1]),
                    self.fars[id_] * torch.ones_like(rays_o[:, :1]),
                ],
                1,
            )  # (h*w, 8)

            sample["rays"] = rays
            sample["ts"] = id_ * torch.ones(len(rays), dtype=torch.long)
            sample["img_wh"] = torch.LongTensor([img_w, img_h])
            sample["valid_mask"] = valid_mask

        else:
            # if generate novel views
            if (
                np.any(self.test_K)
                and self.test_img_h
                and self.test_img_w
                and self.test_appearance_idx
            ):
                sample = {}
                sample["c2w"] = c2w = torch.FloatTensor(self.poses_test[idx])
                directions = get_ray_directions(
                    self.test_img_h, self.test_img_w, self.test_K
                )
                rays_o, rays_d = get_rays(directions, c2w)
                near, far = 0, 5

                rays = torch.cat(
                    [
                        rays_o,
                        rays_d,
                        near * torch.ones_like(rays_o[:, :1]),
                        far * torch.ones_like(rays_o[:, :1]),
                    ],
                    1,
                )

                sample["rays"] = rays
                sample["ts"] = self.test_appearance_idx * torch.ones(
                    len(rays), dtype=torch.long
                )

                sample["img_wh"] = torch.LongTensor([self.test_img_w, self.test_img_h])
            else:
                sample = {
                    "rays_top": self.all_rays_top[idx, :8],
                    "rays_bottom": self.all_rays_bottom[idx, :8],
                    "ts_top": self.all_rays_top[idx, 8].long(),
                    "ts_bottom": self.all_rays_bottom[idx, 8].long(),
                    "rgbs_top": self.all_rgbs_top[idx],
                    "rgbs_bottom": self.all_rgbs_bottom[idx],
                }

                if len(self.mask) != 0:
                    sample["mask_top"] = self.all_masks_top[idx]
                    sample["mask_bottom"] = self.all_masks_bottom[idx]

        return sample
