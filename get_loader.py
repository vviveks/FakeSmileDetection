import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import cv2
import torchvision.transforms as transforms
import numpy as np

def detectMouth(img):
    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    mouths = mouth_cascade.detectMultiScale(img)
    
    bbox = np.zeros([len(mouths),4,2])
    for i, (x,y,w,h) in enumerate(mouths):
        bbox[i,:,:] = np.array([[y,x],[y,x+w],[y+h,x],[y+h,x+w]])

    if bbox.shape[0] == 0: # if no mouth is detected 
        temp = np.zeros([4,2])
        temp[0] = np.array([230,90])
        temp[1] = np.array([230,175])
        temp[2] = np.array([260,90])
        temp[3] = np.array([260,175])
        return temp.astype(int)
    else: # if only a few mouths detected
        return bbox[-1,:,:].astype(int) # returning the first one

def detectEyes(img):
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(img, 1.1, 1)
    
    bbox = np.zeros([len(eyes),4,2])
    for i, (x,y,w,h) in enumerate(eyes):
        bbox[i,:,:] = np.array([[y,x],[y,x+w],[y+h,x],[y+h,x+w]])
    
    if bbox.shape[0] == 0: # if no eyes are detected
        temp = np.zeros([4,2])
        temp[0] = np.array([80,70])
        temp[1] = np.array([80,130])
        temp[2] = np.array([130,70])
        temp[3] = np.array([130,130])
        return temp.astype(int) # returning a rectangle which we think should contain an eye for most images
    else: # if only a few eyes are detected
        return bbox[0,:,:].astype(int) # returning the first one

class SmileDataset(Dataset):
    def __init__(self, rootdir, annotation_file, transform=None):
        self.rootdir = rootdir
        self.df = pd.read_csv(annotation_file, names=['image', 'type'])
        self.transform = transform

        self.img = self.df["image"]
        self.type = self.df["type"]
        self.classes = ['negative smile', 'positive smile', 'NOT smile']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_id = self.img[index]+'.jpg'
        type = self.classes.index(self.type[index])
        img = cv2.imread(os.path.join(self.rootdir, img_id))
        img = cv2.resize(img, (300, 300))

        # detect mouth
        bbox = detectMouth(img)
        mouth = img[bbox[0,0]:bbox[2,0],bbox[0,1]:bbox[1,1]]
        # mouth = cv2.resize(mouth, (300, 300))

        # detect eyes
        bbox = detectEyes(img)
        eye = img[bbox[0,0]:bbox[2,0],bbox[0,1]:bbox[1,1]]
        # eye = cv2.resize(eye, (300, 300))

        if self.transform is not None:
            img = self.transform(img)
            mouth = self.transform(mouth)
            eye = self.transform(eye)
        
        type = torch.tensor(type, dtype=torch.int32)
        return img, mouth, eye, type

class MyCollate:
    def __init__(self):
        pass

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        imgs = torch.Tensor(imgs)

        mouths = [item[1].unsqueeze(0) for item in batch]
        mouths = torch.cat(mouths, dim=0)
        mouths = torch.Tensor(mouths)

        eyes = [item[2].unsqueeze(0) for item in batch]
        eyes = torch.cat(eyes, dim=0)
        eyes = torch.Tensor(eyes)

        targets = [item[3] for item in batch]
        targets = torch.tensor(targets, dtype=torch.int32)

        return imgs, mouths, eyes, targets

def get_loader(root_folder, annotation_file, transform, batch_size=128, shuffle=True):
    dataset = SmileDataset(root_folder, annotation_file, transform)

    loader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        collate_fn=MyCollate()
    )

    return loader, dataset

if __name__ == "__main__":
    root_folder = "./data/happy_images/happy_images"
    annotation_file = "./data/test.csv"
    transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    loader, dataset = get_loader(root_folder, annotation_file, transform)

    for idx, (img, mouth, eye, type) in enumerate(loader):
        print(img.shape)
        print(mouth.shape)
        print(eye.shape)
        print(type.shape)