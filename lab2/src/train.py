import argparse ,albumentations as A
import torch,os , random, numpy as np

from oxford_pet import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from models.resnet34_unet import ResNet34_UNet
from evaluate import evaluate
from utils import show_result
from albumentations.pytorch import ToTensorV2
from inference import test


def train(args):
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Using device: {device}")



    train_transform = A.Compose(
        [
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(size=(256, 256), scale=(0.8, 1)), 
            A.Rotate(limit=30, p=0.5),
            # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
    )
    valid_transform = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
    )

    # Load datasets
    train_dataset = load_dataset(args.data_path, mode='train', transform=train_transform)
    valid_dataset = load_dataset(args.data_path, mode='valid', transform=valid_transform)
    test_dataset = load_dataset(args.data_path, mode='test', transform= valid_transform) # valid_transform is same as test_transform
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")
    print(f'Test dataset size: {len(test_dataset)}')

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=4, pin_memory=True)

    # Load model
    if args.network == 'unet':
        from models.unet import UNet
        model = UNet()
    elif args.network == 'resnet34_unet':
        print("using resnet")
        model = ResNet34_UNet()
    else:
        raise ValueError(f"Unknown network architecture: {args.network}")

    # Move model to device
    model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Initialize GradScaler
    scaler = torch.amp.GradScaler('cuda')

    # Initialize lists to store loss values
    train_losses = []
    valid_losses = []
    dice_scores = []
    best_dice = 0
    best_epoch = 0
    no_improvement_epochs = 0
    model_save_path = f"./../saved_models/best_{args.network}_lr{args.learning_rate:.5f}_epoch{best_epoch}.pth"
    saved = False
    # Training loop
    for epoch in range(args.epochs):
        train_loss = 0
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # Training phase
        model.train()
        for batch in tqdm(train_dataloader, desc="Training", leave=False):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            optimizer.zero_grad()

            # Forward pass and loss computation within autocast context
            with torch.amp.autocast('cuda'):#torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            # Backward pass and optimization within scaler context
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # Validation phase
        avg_valid_loss, _, _ = evaluate(model, valid_dataloader, criterion, device)

        # Compute average losses
        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        tmp_score = test(model, test_dataloader, device)  
        dice_scores.append(tmp_score)  
        print("dice score : ",tmp_score)
        print(f"Epoch {epoch + 1}/{args.epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_valid_loss}")

        # # Check if current epoch's dice score is the best
        # if epoch % 5 == 0 : 
        #     tmp_score = test(model, test_dataloader, device)
        #     print("dice score : ",tmp_score)
        #     if tmp_score > best_dice:
        #         if saved ==True and os.path.exists(model_save_path):
        #             os.remove(model_save_path)
        #         best_dice = tmp_score
        #         best_epoch = epoch
        #         # print(f'Best model saved at epoch {epoch}')
        #         # model_save_path = f"./../saved_models/best_{args.network}_lr{args.learning_rate:.5f}_epoch{best_epoch}_batch{args.batch_size}.pth"
        #         # torch.save(model.state_dict(), model_save_path)
        #         no_improvement_epochs = 0
        #         saved = True
        #     else:
        #         no_improvement_epochs += 1
    model_save_path = f"./../saved_models/best_{args.network}_lr{args.learning_rate:.5f}_epoch{args.epochs}_batch{args.batch_size}.pth"
    torch.save(model.state_dict(), model_save_path)
        # Early stopping if no improvement in dice score for 15 consecutive epochs
        # if no_improvement_epochs >= 3:
        #     print(f"No improvement in dice score for 15 consecutive epochs. Stopping training at epoch {epoch + 1}.")
        #     break

    # Save loss records
    # loss_record_path = f"./../saved_models/{args.network}_lr{args.learning_rate:.5f}_epoch{best_epoch}_batch{args.batch_size}_loss_record.txt"
    # dice_record_path = f"./../saved_models/{args.network}_lr{args.learning_rate:.5f}_epoch{best_epoch}_batch{args.batch_size}_dice.txt"
    # with open(loss_record_path, 'w') as f:
    #     for epoch, (train_loss, valid_loss) in enumerate(zip(train_losses, valid_losses), 1):
    #         f.write(f"Epoch {epoch}, Training Loss: {train_loss}, Validation Loss: {valid_loss}\n")
    # with open(dice_record_path, 'w') as f:
    #     for epoch, dice_score in enumerate(dice_scores, 1):
    #         f.write(f"Epoch {epoch}, Dice score: {dice_score}\n")

def get_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--data_path', '-p', type=str, default='./../dataset/oxford-iiit-pet', required=False, help='Path to input data')
    parser.add_argument('--epochs', '-e', type=int, default=325, help='Number of training epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--network', '-n', type=str, default='resnet34_unet', help='Network architecture')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU device number')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)
