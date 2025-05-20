import cv2
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from models.Attention_UNet import AttU_Net
import glob
from natsort import natsorted
import torch.nn as nn
import functools
from torchvision import transforms
import os
import argparse

Tensor = torch.cuda.FloatTensor

parser = argparse.ArgumentParser(description="Test MR to CT image generation.")
parser.add_argument("--mr_dir", type=str, required=True, help="Directory containing MR images.")
parser.add_argument("--ct_dir", type=str, required=True, help="Directory containing CT images.")
parser.add_argument("--size", type=int, default=256, help="Size to resize images (default: 256).")
parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save generated CT images.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint.")
args = parser.parse_args()


# Dataset class for TEST MR AND CT SLICES
class testDataset(Dataset):
    def __init__(self, mr_dir, ct_dir, size=256, mr_transforms=None, ct_transforms=None):
        self.mr_files = natsorted(glob.glob(f"{mr_dir}/*.jpg"))
        self.ct_files = natsorted(glob.glob(f"{ct_dir}/*.jpg"))
        self.size = size
        self.mr_transforms = mr_transforms
        self.ct_transforms = ct_transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        mr = Image.open(self.mr_files[idx]).convert('L')
        ct = Image.open(self.ct_files[idx]).convert('L')
        mr = mr.resize((self.size, self.size), Image.BILINEAR)
        ct = ct.resize((self.size, self.size), Image.BILINEAR)
        if self.mr_transforms:
            mr = self.mr_transforms(mr)
            ct = self.ct_transforms(ct)
        return mr, ct


# 设置设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 加载测试数据集

mr_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.1996, std=0.1619),
])
ct_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.238, std=0.1465),
])

test_dataset = testDataset(mr_dir=args.mr_dir, ct_dir=args.ct_dir, size=args.size, mr_transforms=mr_transforms, ct_transforms=ct_transforms)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Load model
attunet = AttU_Net()
attunet = attunet.to(device)
checkpoint = torch.load(args.model_path)
attunet.load_state_dict(checkpoint)
attunet.eval()

# 测试
all_psnr = []
all_ssim = []
all_mae = []

# 存储MR，真实CT和合成CT图像用于显示
real_ct_images = []
generated_ct_images = []
mr_images = []

with torch.no_grad():
    for idx, (data, target) in enumerate(tqdm(test_dataloader)):
        data = data.to(device)
        mr_image = data.squeeze(0).cpu().numpy()
        target_image = target.squeeze(0).squeeze(0).cpu().numpy()

        generated_ct_image = attunet(data).squeeze(0).cpu().numpy()

        target_image.squeeze()

        mean = np.array([0.238])
        std = np.array([0.1465])
        generated_ct_image = generated_ct_image * std[:] + mean[:]
        generated_ct_image = np.clip(generated_ct_image, 0, 1)
        generated_ct_image = (generated_ct_image * 255).astype(np.uint8)
        generated_ct_image = generated_ct_image.squeeze(0)
        generated_ct_image = cv2.threshold(generated_ct_image, 25, 255, cv2.THRESH_TOZERO)[1]

        target_image = target_image * std[:] + mean[:]
        target_image = np.clip(target_image, 0, 1)
        target_image = (target_image * 255).astype(np.uint8)

        mean = np.array([0.1996])
        std = np.array([0.1619])
        mr_image = mr_image * std[:] + mean[:]
        mr_image = np.clip(mr_image, 0, 1)
        mr_image = (mr_image * 255).astype(np.uint8)

        psnr_value = psnr(generated_ct_image, target_image, data_range=255)
        ssim_value = ssim(generated_ct_image, target_image, data_range=255)
        mae_value = np.mean(np.abs(generated_ct_image - target_image))

        all_mae.append(mae_value)
        all_psnr.append(psnr_value)
        all_ssim.append(ssim_value)

        output_path = os.path.join(args.output_dir, f'generated_ct_{idx}.png')
        cv2.imwrite(output_path, generated_ct_image)


        real_ct_images.append(target_image)
        generated_ct_images.append(generated_ct_image)
        mr_images.append(mr_image)

# Calculate average PSNR and SSIM
mean_psnr = np.mean(all_psnr)
std_psnr = np.std(all_psnr)

mean_ssim = np.mean(all_ssim)
std_ssim = np.std(all_ssim)

mean_mae = np.mean(all_mae)
std_mae = np.std(all_mae)

print(f'Mean PSNR: {mean_psnr:.4f}, Std PSNR: {std_psnr:.4f}')
print(f'Mean SSIM: {mean_ssim:.4f}, Std SSIM: {std_ssim:.4f}')
print(f'Mean MAE: {mean_mae:.4f}, Std MAE: {std_mae:.4f}')

# Visualization
plt.close('all')
plt.figure(figsize=(15, 10))
for i in range(min(5, len(mr_images))):
    # MR Image
    plt.subplot(3, 5, i + 1)
    mr_img = mr_images[i+0].squeeze()
    if isinstance(mr_img, np.ndarray):
        if mr_img.shape[0] == 3:
            mr_img = np.transpose(mr_img, (1, 2, 0))
    else:
        mr_img = mr_img.cpu().numpy()
        if mr_img.shape[0] == 3:
            mr_img = np.transpose(mr_img, (1, 2, 0))
    plt.imshow(mr_img, cmap='gray')
    plt.title('MR Image')
    plt.axis('off')

    # Real CT Image
    plt.subplot(3, 5, i + 6)
    real_ct_img = real_ct_images[i+0].squeeze()
    if isinstance(real_ct_img, np.ndarray):
        if real_ct_img.shape[0] == 3:
            real_ct_img = np.transpose(real_ct_img, (1, 2, 0))
    else:
        real_ct_img = real_ct_img.cpu().numpy()
        if real_ct_img.shape[0] == 3:
            real_ct_img = np.transpose(real_ct_img, (1, 2, 0))
    plt.imshow(real_ct_img, cmap='gray')
    plt.title('Real CT Image')
    plt.axis('off')

    # Generated CT Image
    plt.subplot(3, 5, i + 11)
    gen_ct_img = generated_ct_images[i+0].squeeze()
    if isinstance(gen_ct_img, np.ndarray):
        if gen_ct_img.shape[0] == 3:
            gen_ct_img = np.transpose(gen_ct_img, (1, 2, 0))
    else:
        gen_ct_img = gen_ct_img.cpu().numpy()
        if gen_ct_img.shape[0] == 3:
            gen_ct_img = np.transpose(gen_ct_img, (1, 2, 0))
    plt.imshow(gen_ct_img, cmap='gray', vmax=255, vmin=0)
    plt.title('Generated CT Image')
    plt.axis('off')

plt.tight_layout()
plt.show()

