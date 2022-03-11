import torch
from torchmetrics import F1Score
import numpy as np
from sklearn.metrics import classification_report

def classification_reports(model, loader, device):
    y_pred = []
    y_true = []
    for batch_idx, (face, mouth, eye, target) in enumerate(loader):
        # Input
        face = face.to(device)
        mouth = mouth.to(device)
        eye = eye.to(device)
        target = target.type(torch.LongTensor).to(device)
        
        output = model(face, mouth, eye)

        _, preds = torch.max(output, 1)

        y_pred.append(preds.cpu().numpy())
        y_true.append(target.cpu().numpy())

        y_pred = [i[0][0][0] for i in y_pred]
        y_true = [i[0] for i in y_true]
    print(classification_report(y_true, y_pred))

def score(model, criterion, loader, device):
    losses = []
    f1scores = []
    for batch_idx, (face, mouth, eye, target) in enumerate(loader):
        # Input
        face = face.to(device)
        mouth = mouth.to(device)
        eye = eye.to(device)
        target = target.type(torch.LongTensor).to(device)
        
        output = model(face, mouth, eye)

        # Cross_Entropy_Loss
        loss = criterion(output, target)
        losses.append(loss.item())

        # F1_Score
        _, preds = torch.max(output, 1)
        f1 = F1Score(num_classes=3).to(device)
        f1score = f1(preds, target)
        f1scores.append(f1score.item())
    
    print(f"Test Loss: {np.mean(losses)}, Test f1: {np.mean(f1scores)}")

def save_checkpoint(state, filename="new_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])