import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dowload import download_and_extract, get_datasets

# -----------------------------
# Augmentation
# -----------------------------


def get_augmentations(img_size=256, augment=True):
    if augment:
        return A.Compose([
            A.Rotate(limit=(0, 90), p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ElasticTransform(
                alpha=1, sigma=50, alpha_affine=10,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                approximate=True, same_dxdy=True, p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.Resize(img_size, img_size),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            ToTensorV2()
        ])


# -----------------------------
# Dataset cho binary segmentation (background, polyp)
# -----------------------------
class PolypDatasets(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=256, augment=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))
        self.transform = get_augmentations(img_size=img_size, augment=augment)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # --- Load image ---
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- Load mask (grayscale: 0 or 255) ---
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # --- Chuyển mask về 0/1 ---
        mask = (mask > 127).astype("float32")   # (H, W)

        # --- Apply augmentation ---
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]              # (3, H, W)
        mask = augmented["mask"]                # (H, W)

        # --- Ép mask thành (1, H, W) ---
        mask = mask.unsqueeze(0)                # thêm channel = 1

        return image, mask


# -----------------------------
# DataLoader cho train/val/test
# -----------------------------
def get_dataloaders(dataset_class, images_dir, masks_dir,
                    img_size=256, batch_size=4,
                    split_ratio=(0.7, 0.15, 0.15), num_workers=2):

    full_dataset = dataset_class(images_dir, masks_dir,
                                 img_size=img_size, augment=True)

    n_total = len(full_dataset)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])
    n_test = n_total - n_train - n_val

    train_dataset, val_dataset_idx, test_dataset_idx = random_split(
        full_dataset, [n_train, n_val, n_test]
    )

    # val dataset (augment=False)
    val_dataset = dataset_class(images_dir, masks_dir,
                                img_size=img_size, augment=False)
    val_dataset.image_files = [full_dataset.image_files[i]
                               for i in val_dataset_idx.indices]
    val_dataset.mask_files = [full_dataset.mask_files[i]
                              for i in val_dataset_idx.indices]

    # test dataset (augment=False)
    test_dataset = dataset_class(images_dir, masks_dir,
                                 img_size=img_size, augment=False)
    test_dataset.image_files = [full_dataset.image_files[i]
                                for i in test_dataset_idx.indices]
    test_dataset.mask_files = [full_dataset.mask_files[i]
                               for i in test_dataset_idx.indices]

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True, persistent_workers=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True, persistent_workers=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader


def dataloader(dataset_name="cvc-clinicdb",
               dataset_dir="./datasets",
               dataset_class=PolypDatasets,
               img_size=256,
               batch_size=4,
               split_ratio=(0.7, 0.15, 0.15),
               num_workers=2,
               download=True):
    """
    Download (nếu cần), load dataset và trả về train/val/test dataloaders.

    Args:
        dataset_name (str): tên dataset trong get_datasets()
        dataset_dir (str): thư mục lưu dataset
        dataset_class (Dataset): class dataset dùng để load
        img_size (int): resize ảnh
        batch_size (int): batch size
        split_ratio (tuple): tỉ lệ train/val/test
        num_workers (int): số worker cho DataLoader
        download (bool): có tải dataset nếu chưa có không

    Returns:
        train_loader, val_loader, test_loader
    """
    # --- Download dataset nếu cần ---
    datasets = get_datasets()
    dataset_dict = {d["name"].lower(): d for d in datasets}

    name_lower = dataset_name.lower()
    if name_lower not in dataset_dict:
        raise ValueError(f"Dataset '{dataset_name}' No exits!")

    if download:
        download_and_extract(dataset_dict[name_lower]["name"],
                             dataset_dict[name_lower]["url"],
                             out_dir=dataset_dir)

    # --- Thư mục images và masks ---
    data_path = os.path.join(dataset_dir, dataset_dict[name_lower]["name"])
    data_path = os.path.join(data_path, dataset_dict[name_lower]["name"])
    images_dir = os.path.join(data_path, "images")
    masks_dir = os.path.join(data_path, "masks")

    # --- Trả về dataloaders ---
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_class, images_dir, masks_dir,
        img_size=img_size, batch_size=batch_size,
        split_ratio=split_ratio, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader

# if __name__ == "__main__":
#     # Example: load dataloaders
#     train_loader, val_loader, test_loader = dataloader(
#         dataset_name="cvc-clinicdb",
#         dataset_dir="./datasets",
#         dataset_class=PolypDatasets,
#         img_size=256,
#         batch_size=4,
#         split_ratio=(0.7, 0.15, 0.15),
#         num_workers=2,
#         download=True
#     )

#     print(f"Train batches: {len(train_loader)}")
#     print(f"Val batches: {len(val_loader)}")
#     print(f"Test batches: {len(test_loader)}")
