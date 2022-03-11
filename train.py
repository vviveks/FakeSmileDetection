import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from utils import save_checkpoint, load_checkpoint, score
from get_loader import get_loader
from model import SmileClassifier
import numpy as np
from torchmetrics import F1Score

def load():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    data_transforms = {
        'train':
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299,299)),
            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'test':
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299,299)),
            transforms.ToTensor(),
            normalize
        ]),
    }

    annotation_file = {
        'train': "./data/data/train.csv",
        'test': "./data/data/test.csv"
    }

    root_folder = "./data/data/happy_images/happy_images"

    train_loader, _ = get_loader(root_folder, annotation_file['train'], data_transforms['train'])
    test_loader, _ = get_loader(root_folder, annotation_file['test'], data_transforms['test'])

    return train_loader, test_loader

def train():
    train_loader, test_loader = load()

    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = True
    save_model = True

    # Hyperparameters
    hidden_size = 64
    learning_rate = 0.001
    num_classes = 3
    num_epochs = 20

    # Initialize model, loss and optim
    model = SmileClassifier(hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        load_checkpoint(torch.load("checkpoint.pth.tar"), model, optimizer)

    for epoch in range(num_epochs):
        print(f"[Epoch {epoch+20} / {num_epochs+20}]")

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)
        
        losses = []
        f1scores = []
        for batch_idx, (face, mouth, eye, target) in enumerate(train_loader):
            # Input
            face = face.to(device)
            mouth = mouth.to(device)
            eye = eye.to(device)
            target = target.type(torch.LongTensor).to(device)

            output = model(face, mouth, eye)

            # Cross_Entropy_Loss
            loss = criterion(output, target)
            losses.append(loss.item())

            # # F1_Score
            model.eval()
            with torch.no_grad():
                _, preds = torch.max(output, 1)
                f1 = F1Score(num_classes=3).to(device)
                f1score = f1(preds, target)
                f1scores.append(f1score.item())
            model.train()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Train Loss: {np.mean(losses)}, Train f1: {np.mean(f1scores)}")
        model.eval()
        with torch.no_grad():
            score(model, criterion, test_loader, device)
        model.train()

if __name__ == "__main__":
    train()








