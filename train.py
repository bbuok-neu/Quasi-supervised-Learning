from models.Attention_UNet import AttU_Net
from models.as_loss import approximately_supervised_loss_exp
import argparse
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn
import numpy as np
import random
import os
from PIL import Image
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser(description="Training Resnet Model")

parser.add_argument('--batch_size', type=int, default=200, help='Batch size for training and validation')
parser.add_argument('--train_size', type=int, default=64, help='train image size for training and validation')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train the model')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='Initial learning rate')
parser.add_argument('--num_workers', type=int, default=2, help='dataloader num workers')
# if continue training
parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint')
parser.add_argument('--last_epoch', type=int, default=0, help='Checkpoint for last epoch when resume training')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/checkpoint_to_load.pt',
                    help='Path to the checkpoint file')

parser.add_argument('--val_ratio', type=float, default=0.001, help='Validation split ratio')
parser.add_argument('--dataset_path', type=str, default='your_patch_pairs_dataframe.pkl',
                    help='Path to the dataset pickle file')
parser.add_argument('--save_path', type=str, default='./checkpoints/', help='Path to save model weights')
parser.add_argument('milestones', nargs='*', default=[1, 20, 50, 70, 100], help='MultiStepLR milestones')

args = parser.parse_args()


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# Load train dataframe pickle
with open(args.dataset_path, 'rb') as f:
    train_df = pickle.load(f)


class trainDataset(Dataset):
    def __init__(self, df, size=64, mr_transforms=None, ct_transforms=None, data_augmentation=None):
        self.df = df
        self.size = size
        self.mr_transforms = mr_transforms
        self.ct_transforms = ct_transforms
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        mr = Image.open(self.df.iloc[idx, 0]).convert('L')
        ct = Image.open(self.df.iloc[idx, 1]).convert('L')
        mr = mr.resize((self.size, self.size), Image.Resampling.BILINEAR)
        ct = ct.resize((self.size, self.size), Image.Resampling.BILINEAR)
        if self.data_augmentation:
            mr = self.data_augmentation(mr)
            ct = self.data_augmentation(ct)
        if self.mr_transforms:
            mr = self.mr_transforms(mr)
            ct = self.ct_transforms(ct)
        return mr, ct


crop_resize = transforms.Compose([
    transforms.RandomApply([transforms.RandomCrop(24),
                            transforms.Resize((args.train_size, args.train_size))], p=0.2),
])
mr_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.1996, std=0.1619),
])
ct_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.238, std=0.1465),
])

train_dataset = trainDataset(df=train_df, size=args.train_size, paired=args.paired, mr_transforms=mr_transforms,
                             ct_transforms=ct_transforms)

# split dataset
val_ratio = args.val_ratio
num_val_samples = int(train_dataset.__len__() * val_ratio)
num_train_samples = train_dataset.__len__() - num_val_samples
train_dataset, val_dataset = random_split(train_dataset, [num_train_samples, num_val_samples])

train_dataloader = DataLoaderX(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True,
                               num_workers=args.num_workers)
val_dataloader = DataLoaderX(val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True,
                             num_workers=args.num_workers)

# Instantiating Neural Networks
attunet = AttU_Net()
attunet = attunet.to(device)
epoch_add = args.last_epoch
if args.resume:
    checkpoint = torch.load(args.checkpoint_path)
    attunet.load_state_dict(checkpoint)
    epoch_add += 1


def train_model(model_name, model, train_loader, val_loader, optimizer, lr_scheduler, num_epochs):
    print(model_name)
    l1 = nn.L1Loss(reduction='none')

    # viz = Visdom(port=18097)  # visdom window
    # loss_window = viz.line(X=np.array([0]), Y=np.array([0]),
    #                        opts=dict(xlabel='Epoch', ylabel='Loss', title='Training and Validation Loss'))
    #
    # image_window = viz.images(np.random.randn(3, 1, 256, 256),
    #                           opts=dict(title='Input, Target, Output', caption='Epoch 0'))

    for epoch in tqdm(range(num_epochs), desc='epoch', position=1):

        model.train()  # Enter train mode
        losses = []

        for i_step, (data, target) in enumerate(tqdm(train_loader, leave=False, desc='train iter', position=0)):
            i_step += 1
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)


            optimizer.zero_grad()
            outputs = model(data)
            l1lo = l1(outputs, target)
            l1loss = approximately_supervised_loss_exp(l1lo, gamma=0.5)
            total_loss = l1loss
            total_loss.backward()
            losses.append(total_loss.item())
            optimizer.step()

            # if i_step % 50 == 0:  # visdom show image
            #     images = torch.cat([data[:10]*0.1619+0.1996, target[:10]*0.1465+0.238, outputs[:10]*0.1465+0.238], 0)  # 选择前3个样本
            #     images = images.cpu().detach().numpy()
            #     viz.images(images, nrow=10, win=image_window,
            #                opts=dict(title='Input, Target, Output', caption=f'Epoch {epoch + epoch_add}'))

        model.eval()

        all_loss = []
        all_iou = []
        for i_step, (data, target) in enumerate(val_loader):
            val_style_loss = 0
            with torch.no_grad():
                data = data.to(device)
                target = target.to(device)
                outputs = model(data)
                vl1lo = l1(outputs, target)
                vl1loss = approximately_supervised_loss_exp(vl1lo, gamma=0.5)
                val_total_loss = vl1loss
                all_loss.append(val_total_loss.item())

        mean_val_loss = sum(all_loss) / len(all_loss)
        tqdm.write(f"Epoch [{epoch + epoch_add}] - train loss: {np.mean(losses):.5f}, val loss: {mean_val_loss:.5f}")

        # viz.line(X=np.array([epoch + epoch_add]), Y=np.array([np.mean(losses)]),
        #          win=loss_window, name='Train Loss', update='append')
        # viz.line(X=np.array([epoch + epoch_add]), Y=np.array([mean_val_loss]),
        #          win=loss_window, name='Val Loss', update='append')
        # viz.line(X=np.array([epoch + epoch_add]), Y=np.array([np.mean(iou_losses)]),
        #          win=loss_window, name='Train iou Loss', update='append')
        # viz.line(X=np.array([epoch + epoch_add]), Y=np.array([mean_val_iou]),
        #          win=loss_window, name='Val iou Loss', update='append')

        if (epoch + epoch_add) % 1 == 0:
            path = os.path.join(args.save_path, f'as64_attunet_{str(epoch + epoch_add)}_epoch.pt')
            torch.save(model.state_dict(), path)

        if lr_scheduler:
            lr_scheduler.step(mean_val_loss)


if __name__ == '__main__':
    set_seed(42)
    optimizer = torch.optim.AdamW(attunet.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=0.5, last_epoch=-1)
    train_model("Attention Unet", attunet, train_dataloader, val_dataloader, optimizer=optimizer, lr_scheduler=scheduler, num_epochs=args.num_epochs)
