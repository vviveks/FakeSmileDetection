import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from utils import load_checkpoint, score
from get_loader import get_loader
from model import SmileClassifier

def load():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        normalize
    ])

    annotation_file = "./data/data/train.csv"

    root_folder = "./data/data/happy_images/happy_images"

    test_loader, _ = get_loader(root_folder, annotation_file, transform)

    return test_loader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = load()

    # Hyperparameters
    hidden_size = 64
    learning_rate = 0.001
    num_classes = 3
    num_epochs = 20

    model = SmileClassifier(hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    load_checkpoint(torch.load("checkpoint/checkpoint_40_epoch.pth.tar"), model, optimizer)

    model.eval()
    with torch.no_grad():
        score(model, criterion, test_loader, device)
    model.train()