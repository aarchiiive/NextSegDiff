
import os
import cv2
import glob
import time
import shutil
import nibabel
import random

import torch
import torch.nn
from torchvision import transforms
import numpy as np

from pathlib import Path
from tqdm import tqdm
from PIL import Image
import SimpleITK as sitk

class AMOSDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform, phase="Tr", save_path="amos", save=False, target_class=None, pre=False, test_flag=False):
        super().__init__()
        self.phase = phase
        self.transform = transform
        self.save_path = save_path
        self.save = save
        self.pre = pre
        self.target_class = target_class
        self.image_dir = Path(os.path.expanduser(os.path.join(data_dir, f"images{self.phase}")))
        self.label_dir = Path(os.path.expanduser(os.path.join(data_dir, f"labels{self.phase}")))
        self.nii_image_files = sorted(glob.glob(str(self.image_dir / Path("*.nii.gz"))))
        self.nii_label_files = sorted(glob.glob(str(self.label_dir / Path("*.nii.gz"))))
        self.image_files = sorted(glob.glob(str(self.image_dir / Path("*.png"))))
        self.label_files = sorted(glob.glob(str(self.label_dir / Path("*.png"))))

        assert len(self.image_files) == len(self.label_files), "Number of image and label files should match"

    def __len__(self):
        return len(self.nii_image_files) if self.pre else len(self.image_files)

    def __getitem__(self, x):
        img_path = self.image_files[x]
        msk_path = self.label_files[x]

        img = Image.open(img_path).convert('RGB') # 3
        mask = Image.open(msk_path).convert('L') # 1
        
        # print(img.size, mask.size)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)
            
        # img = transforms.ToTensor()(img)
        # mask = transforms.ToTensor()(img)
        # print(type(img))
        # print(type(mask))
        # print(img.shape, mask.shape)

        return (img, mask, os.path.basename(img_path))
    
    def preprocess(self, x):
        # nii.gz 파일 열기
        nii_img = nibabel.load(self.nii_image_files[x])
        nii_label = nibabel.load(self.nii_label_files[x])

        # 데이터 읽기
        img = nii_img.get_fdata()
        label = nii_label.get_fdata()
        
        # torch ver.
        img = torch.from_numpy(img).unsqueeze(0).cuda()  # (1, H, W, C)
        label = torch.from_numpy(label).unsqueeze(0).cuda()  # (1, H, W, C)
        
        if self.target_class is None: label = torch.where(label != 0, 255, label)

        # numpy ver.
        """label = np.where(label != 0, 255, label)
        
        # img = cv2.GaussianBlur(img, (5, 5), 0)
        # label = cv2.GaussianBlur(label, (5, 5), 0)
        
        # _, img = cv2.threshold(img, 32, 255, cv2.THRESH_TOZERO)
        # _, label = cv2.threshold(label, 32, 255, cv2.THRESH_BINARY)
        
        # img = img / np.max(img) * 255
        # label = label / np.max(label) * 255
        
        # img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        # label = (label - np.min(label)) / (np.max(label) - np.min(label)) * 255
        
        # img = (img - np.mean(img)) / np.std(img) * 255
        # label = (label - np.mean(label)) / np.std(label) * 255
        
        # img = self.histogram_equalization(img) / np.max(img) * 255
        # label = self.histogram_equalization(label) / np.max(img) * 255"""
        
        for i in range(img.shape[3]):
            slice_img = img[:, :, :, i].cuda()  # (C, H, W)
            slice_label = label[:, :, :, i].cuda()  # (C, H, W)
            
            if self.target_class is not None:
                if target_class in torch.unique(slice_label):
                    slice_label = torch.where(slice_label == target_class, 255, 0)
                    self.save = True
                else:
                    slice_label = torch.zeros_like(slice_label)

            img_path = f'{self.save_path}/images{self.phase}/{os.path.basename(self.nii_image_files[x]).split(".")[0]}_slice_{i}.png'
            label_path = f'{self.save_path}/labels{self.phase}/{os.path.basename(self.nii_label_files[x]).split(".")[0]}_slice_{i}.png'

            if os.path.isfile(img_path) and os.path.isfile(label_path):
                continue
            elif self.phase == "Ts" and self.save:
                transforms.ToPILImage()(slice_img.byte()).save(img_path)
                transforms.ToPILImage()(slice_label.byte()).save(label_path)
            elif torch.any(slice_label != 0) and self.save:
                transforms.ToPILImage()(slice_img.byte()).save(img_path)
                transforms.ToPILImage()(slice_label.byte()).save(label_path)
                if self.target_class is not None: self.save = False

    def histogram_equalization(self, image):
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized)
        equalized_image = equalized_image.reshape(image.shape)
        return equalized_image
    
class AMOSDataset3D(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform, crop_size=10, phase="Tr", test_flag=False):
        super().__init__()
        random.seed(int(time.time()) % 9999)
        self.phase = phase
        self.transform = transform
        self.crop_size = crop_size if self.phase == "Tr" else crop_size # - 1
        self.image_dir = Path(os.path.expanduser(os.path.join(data_dir, f"images{self.phase}")))
        self.label_dir = Path(os.path.expanduser(os.path.join(data_dir, f"labels{self.phase}")))
        self.image_files = sorted(glob.glob(str(self.image_dir / Path("*.nii.gz"))))
        self.label_files = sorted(glob.glob(str(self.label_dir / Path("*.nii.gz"))))
        
        assert len(self.image_files) == len(self.label_files), "Number of image and label files should match"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, x):
        img_path = self.image_files[x]
        msk_path = self.label_files[x]
        
        img = nibabel.load(img_path).get_fdata()
        mask = nibabel.load(msk_path).get_fdata()
        
        d = np.shape(img)[2]
        n = random.randint(0, d - self.crop_size - 1)
        
        # crop
        img = torch.tensor(img)[:,:,n : n + self.crop_size]
        mask = torch.tensor(mask)[:,:,n : n + self.crop_size]
        
        # (H, W, D) -> (D, W, H)
        img = torch.transpose(img, 0, 2).contiguous()
        mask = torch.transpose(mask, 0, 2).contiguous()
        
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)
            
        return (img, mask, os.path.basename(img_path))

if __name__ == "__main__":
    data_dir = "AMOS" # nii.gz 파일이 있는 폴더
    phase = "Va"
    
    class_names = {0: "background", 1: "spleen", 2: "right kidney", 3: "left kidney", 
                   4: "gall bladder", 5: "esophagus", 6: "liver", 7: "stomach", 
                   8: "arota", 9: "postcava", 10: "pancreas", 11: "right adrenal gland", 
                   12: "left adrenal gland", 13: "duodenum", 14: "bladder", 15: "prostate,uterus"}
    target_class = 10
    
    save_path = f"dataset/{data_dir}2D/{class_names[target_class]}" if target_class is not None else f"dataset/{data_dir}2D"
    
    if not os.path.isdir(f"{save_path}/images{phase}"):
        os.makedirs(f"{save_path}/images{phase}", exist_ok=True)
        os.makedirs(f"{save_path}/labels{phase}", exist_ok=True)
    
    dataset = AMOSDataset(os.path.join("dataset", data_dir), 
                          None, 
                          phase, 
                          save_path=save_path, 
                          target_class=target_class,
                          save=False, 
                          pre=True)
    
    for i in tqdm(range(len(dataset))):
        dataset.preprocess(i)