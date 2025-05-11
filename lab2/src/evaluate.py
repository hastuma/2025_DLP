import torch
from tqdm import tqdm

def evaluate(model, dataloader, criterion, device):
    model.eval()
    valid_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            valid_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(masks.cpu().numpy())

    avg_valid_loss = valid_loss / len(dataloader)
    return avg_valid_loss, all_preds, all_labels