import numpy as np  
import torch
import matplotlib.pyplot as plt
import re
def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    num = len(gt_mask)
    smooth = 1e-5
    m1 = torch.round(torch.sigmoid(pred_mask)) #sigmoid完再round 就會是binary
    m1 = m1.view(num, -1)  
    m2 = gt_mask.view(num, -1)  
    intersection = m1 * m2
    score = (2 * intersection.sum() + smooth) / (m1.sum() + m2.sum() + smooth)
    return score
    
def show_result(input_img , label_mask, output_mask,i):
    # 轉換 Tensor 為 NumPy 格式以便 Matplotlib 顯示
    input_img = input_img.permute(1, 2, 0).numpy()  # 轉換為 (H, W, C)
    label_mask = label_mask.squeeze().numpy()  # 移除 batch 和 channel 維度
    output_mask = torch.round(torch.sigmoid(output_mask.squeeze())).numpy()  # 預測結果
    # 繪製三張圖像並合併成一張
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 1 行 3 列
    axes[0].imshow(input_img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(label_mask, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(output_mask, cmap="gray")
    axes[2].set_title("Model Output")
    axes[2].axis("off")

    plt.savefig(f'./result{str(i)}.jpg')

def plot_dice_score__batch_curve():
    batch_sizes = [64, 256]
    dice_data = {}
    for bs in batch_sizes:
        file_path = f"/home/winston/dlp/lab/lab2/saved_models/resnet34_unet_lr0.00200_epoch0_batch{bs}_dice.txt"
        epochs = []
        dice_scores = []
        with open(file_path, "r") as f:
            for line in f:
                # 每行格式類似：
                # "Epoch 51, Dice score: 0.594"
                match = re.search(r"Epoch (\d+), Dice score: ([0-9.]+)", line)
                if match:
                    epoch = int(match.group(1))
                    dice_score = float(match.group(2))
                    epochs.append(epoch)
                    dice_scores.append(dice_score)
        dice_data[bs] = (epochs, dice_scores)

    # 畫圖
    plt.figure(figsize=(10, 6))
    for bs in batch_sizes:
        epochs, dice_scores = dice_data[bs]
        plt.plot(epochs, dice_scores, label=f"Batch size {bs}")

    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Dice Score Curves for Different Batch Sizes")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig("dice_score_curves.png")

def plot_dice_score_lr_curve():
    learning_rates = ["0.00010", "0.00200"]
    dice_data = {}
    batch_size = 64  # 固定的 batch size
    for lr in learning_rates:
        file_path = f"/home/winston/dlp/lab/lab2/saved_models/resnet34_unet_lr{lr}_epoch0_batch{batch_size}_dice.txt"
        epochs = []
        dice_scores = []
        with open(file_path, "r") as f:
            for line in f:
                # 每行格式類似：
                # "Epoch 51, Dice score: 0.594"
                match = re.search(r"Epoch (\d+), Dice score: ([0-9.]+)", line)
                if match:
                    epoch = int(match.group(1))
                    dice_score = float(match.group(2))
                    epochs.append(epoch)
                    dice_scores.append(dice_score)
        dice_data[lr] = (epochs, dice_scores)

    # 畫圖
    plt.figure(figsize=(10, 6))
    for lr in learning_rates:
        epochs, dice_scores = dice_data[lr]
        plt.plot(epochs, dice_scores, label=f"Learning rate {lr}")

    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Dice Score Curves for Different Learning Rates")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig("dice_score_lr_curves.png")

if __name__ == '__main__':
    plot_dice_score_lr_curve()