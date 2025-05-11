import argparse
import torch
import numpy as np
from oxford_pet import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from evaluate import evaluate  
from models.unet import UNet 
from models.resnet34_unet import ResNet34_UNet 
from utils import dice_score, show_result
import albumentations as A
from albumentations.pytorch import ToTensorV2
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model','-m', default='MODEL.pth', help='path to the stored model weight')
    parser.add_argument('--data_path','-d', type=str, default = "./../dataset/oxford-iiit-pet",help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='batch size')
    parser.add_argument('--network','-n', type=str, default='ResNet34_UNet', help='model type')  
    return parser.parse_args()

def load_model(model_path, device, model_type):
    model = UNet() if model_type == 'unet' else ResNet34_UNet()
    model.load_state_dict(torch.load(model_path, map_location=device))  
    model.to(device)
    model.eval()
    return model

def test(model, test_dataloader, device):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ])
    dice = 0
    model.eval()
    with torch.no_grad() :  
        for i, data in enumerate(test_dataloader):
            image = data['image'].to(device).to(torch.float)
            mask = data['mask'].to(device)
            outputs = model(image)
            dice += float(dice_score(outputs, mask))
    # print('Dice Score : ', round(dice/len(test_dataloader), 4))
    return round(dice/len(test_dataloader), 4)


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("using device:", device)
    transform = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
    )
    test_dataset = load_dataset(args.data_path, mode='test', transform=transform)
    # print(f'Test dataset size: {len(test_dataset)}')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=4, pin_memory=False)
    model = load_model("./../saved_models/"+args.model, device, args.network)
    dice = 0
    model.eval()
    with torch.no_grad() :  
        for i, data in enumerate(test_dataloader):
            image = data['image'].to(device).to(torch.float)
            mask = data['mask'].to(device)
            outputs = model(image)
            # if i < 5:
            #     input_img = image[0].cpu()  
            #     label_mask = mask[0].cpu()
            #     output_mask = outputs[0].cpu()
            #     show_result(input_img, label_mask, output_mask,i)
            dice += float(dice_score(outputs, mask))
            # if i == 0 :
            #     utils.show_predict(args.network, data['image'][0], data['mask'][0], outputs[0])
    # print(args.model)
    print('Dice Score : ', round(dice/len(test_dataloader), 4))
