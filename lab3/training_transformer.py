import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers()
        self.prepare_training()
        self.args = args
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)


    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}")
        scaler = torch.cuda.amp.GradScaler()

        for batch_idx, images in enumerate(progress_bar):
            images = images.to(self.args.device)
            self.optim.zero_grad()


            with torch.cuda.amp.autocast():
                predictions, targets = self.model(images)
                predictions = predictions.view(-1, predictions.size(-1))
                targets = targets.view(-1)
                loss = F.cross_entropy(predictions, targets)

            scaler.scale(loss).backward()
            scaler.step(self.optim)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.3f}")

        average_loss = total_loss / len(train_loader)
        return average_loss

    def eval_one_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0
        progress_bar = tqdm(val_loader, total=len(val_loader), desc="Validation")

        with torch.no_grad():
            for batch_idx, images in enumerate(progress_bar):
                images = images.to(self.args.device)

                # 混合精度前向傳播
                with torch.cuda.amp.autocast():
                    predictions, targets = self.model(images)
                    predictions = predictions.view(-1, predictions.size(-1))
                    targets = targets.view(-1)

                    # 計算損失
                    loss = F.cross_entropy(predictions, targets)

                total_loss += loss.item()

                # 更新進度條
                progress_bar.set_postfix(loss=f"{loss.item():.3f}")

        average_loss = total_loss / len(val_loader)
        return average_loss
    #####################下面沒有用FP16###############
    # def train_one_epoch(self, train_loader):
    #     # pass
    #     self.model.train()
    #     total_loss = 0
    #     progress_bar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}")

    #     for batch_idx, images in enumerate(progress_bar):
    #         images = images.to(self.args.device)
    #         self.optim.zero_grad()

    #         # Forward pass
    #         predictions, targets = self.model(images)
    #         predictions = predictions.view(-1, predictions.size(-1))
    #         targets = targets.view(-1)

    #         # Compute loss
    #         loss = F.cross_entropy(predictions, targets)
    #         total_loss += loss.item()

    #         # Backward pass and optimization
    #         loss.backward()
    #         self.optim.step(self.scheduler)

    #         # Update progress bar
    #         progress_bar.set_postfix(loss=f"{loss.item():.3f}")

    #     average_loss = total_loss / len(train_loader)
    #     return average_loss
    def eval_one_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0
        progress_bar = tqdm(val_loader, total=len(val_loader), desc="Validation")

        with torch.no_grad():
            for batch_idx, images in enumerate(progress_bar):
                images = images.to(self.args.device)

                # Forward pass
                predictions, targets = self.model(images)
                predictions = predictions.view(-1, predictions.size(-1))
                targets = targets.view(-1)

                # Compute loss
                loss = F.cross_entropy(predictions, targets)
                total_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix(loss=f"{loss.item():.3f}")

        average_loss = total_loss / len(val_loader)
        return average_loss
        # pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.transformer.parameters(), lr=args.learning_rate)
        scheduler = None
        return optimizer,scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab5_dataset/train/", help='Training Dataset Path')
    parser.add_argument('--valid_path', type=str, default="./lab5_dataset/val/", help='Validation Dataset Path')
    parser.add_argument('--save_path', type=str, default='./transformer_checkpoints/', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(default: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=1, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.valid_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:    
    loss_file_path = os.path.join(args.save_path, "fp16_loss.txt")
    os.makedirs(args.save_path, exist_ok=True)  # 確保 save_path 目錄存在
    with open(loss_file_path, "w") as loss_file:  # 開啟檔案以寫入模式
        for epoch in range(args.start_from_epoch, args.epochs + 1):
            train_loss = []
            valid_loss = []

            epoch_train_loss = train_transformer.train_one_epoch(train_loader)
            epoch_valid_loss = train_transformer.eval_one_epoch(val_loader)

            train_loss.append(epoch_train_loss)
            valid_loss.append(epoch_valid_loss)

            log_message = f"Epoch {epoch} , Train loss : {epoch_train_loss}, Valid loss : {epoch_valid_loss}\n"
            print(log_message.strip())  # 印出到終端機
            loss_file.write(log_message)  # 寫入到 loss.txt

            torch.save(train_transformer.model.transformer.state_dict(), os.path.join(args.save_path, f"fp16_epoch_{epoch}.pt"))