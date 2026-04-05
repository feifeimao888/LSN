import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2
import albumentations as A
import random

class HMCnet_Dataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes,
                 dataEnhance, dataset_path, mean, std, seed=6, need_enhance_txt_path=os.path.join("cityscape", "all_train_needEnhance.txt")):
        super(HMCnet_Dataset, self).__init__()

        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = dataEnhance
        self.dataset_path = dataset_path
        self.mean = mean
        self.std = std
        self.seed = seed

        # 【修改1】读取需要增强的图片名（存为set，查询速度快）
        self.need_enhance_names = set()
        if need_enhance_txt_path and os.path.exists(need_enhance_txt_path):
            with open(need_enhance_txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.need_enhance_names.add(line)

        # 【修改2】拆分Transform：先定义"无增强的基础Transform"
        self.no_aug_transform = A.Compose([
            A.Resize(
                self.input_shape[0],
                self.input_shape[1],
                interpolation=cv2.INTER_CUBIC
            ),
            A.Normalize(
                mean=self.mean,
                std=self.std,
                max_pixel_value=255.0
            )
        ])
        print(f"需要增强的数据集为{need_enhance_txt_path}")
        # 再定义"带增强的Train Transform"
        if self.train:
            self.transform = A.Compose([
                A.OneOf([
                    A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        A.Affine(
                            scale=(0.8, 1.2),
                            rotate=(-45, 45),
                            shear=(-10, 10),
                            mode=cv2.BORDER_CONSTANT,
                            cval=0,
                            mask_mode=cv2.BORDER_CONSTANT,
                            mask_cval=0,
                            p=0.6
                        ),
                        A.ElasticTransform(
                            alpha=50,
                            sigma=5,
                            p=0.15
                        ),
                    ]),
                    A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.RandomScale(scale_limit=0.2, p=0.5),
                        A.PadIfNeeded(
                            min_height=input_shape[0],
                            min_width=input_shape[1],
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=0
                        ),
                        A.RandomCrop(
                            height=input_shape[0],
                            width=input_shape[1]
                        ),
                    ])
                ], p=0.8),

                A.OneOf([
                    A.Compose([
                        A.RandomBrightnessContrast(
                            brightness_limit=0.25,
                            contrast_limit=0.25,
                            p=1.0
                        ),
                        A.CLAHE(
                            clip_limit=4.0,
                            tile_grid_size=(8, 8),
                            p=0.5
                        ),
                    ]),
                    A.Compose([
                        A.GaussNoise(
                            var_limit=(10.0, 30.0),
                            p=0.5
                        ),
                        A.OneOf([
                            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                            A.MotionBlur(blur_limit=5, p=0.2),
                        ], p=0.3),
                    ]),
                    A.Sharpen(alpha=(0.2, 0.5), p=0.3),
                ], p=0.6),

                A.Resize(
                    self.input_shape[0],
                    self.input_shape[1],
                    interpolation=cv2.INTER_CUBIC
                ),

                A.Normalize(
                    mean=self.mean,
                    std=self.std,
                    max_pixel_value=255.0
                )
            ])
        else:
            self.transform = self.no_aug_transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        # ----------------------------------
        # 关键：让每个 index 的增强确定
        # ----------------------------------
        seed = self.seed + index
        random.seed(seed)
        np.random.seed(seed)

        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        jpg_path = os.path.join(self.dataset_path, "imgdata", name + ".jpg")
        png_path = os.path.join(self.dataset_path, "segdata", name + ".png")

        jpg = np.array(Image.open(jpg_path).convert("RGB"))
        png = np.array(Image.open(png_path).convert("L"))

        # 【修改3】按条件选择Transform：只有训练模式+在名单里才增强
        if self.train and (name in self.need_enhance_names):
            augmented = self.transform(image=jpg, mask=png)
        else:
            augmented = self.no_aug_transform(image=jpg, mask=png)
            
        jpg = augmented['image']
        png = augmented['mask']

        # HWC -> CHW
        jpg = np.transpose(jpg, [2, 0, 1])

        png = np.array(png)
        png[png >= self.num_classes] = 0  # 二分类无需 +1

        # 二分类 One-hot
        seg_labels = np.eye(self.num_classes)[png.reshape(-1)]
        seg_labels = seg_labels.reshape(
            (self.input_shape[0],
             self.input_shape[1],
             self.num_classes)
        )

        return jpg, png, seg_labels


# DataLoader中collate_fn使用
def unet_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels