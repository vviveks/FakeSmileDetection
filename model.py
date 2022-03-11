import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SmileClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(SmileClassifier, self).__init__()

        ######################### Resnet50 ########################
        # Faces
        self.res_faces = models.resnet50(pretrained=True)
        for param in self.res_faces.parameters():
            param.requires_grad = False
        self.res_faces.fc = nn.Linear(2048, hidden_size)

        # Mouths
        self.res_mouths = models.resnet50(pretrained=True)
        for param in self.res_mouths.parameters():
            param.requires_grad = False  
        self.res_mouths.fc = nn.Linear(2048, hidden_size)

        # Eyes
        self.res_eyes = models.resnet50(pretrained=True)
        for param in self.res_eyes.parameters():
            param.requires_grad = False
        self.res_eyes.fc = nn.Linear(2048, hidden_size)

        ######################### Densenet201 #######################
        # Faces
        self.den_faces = models.densenet201(pretrained=True)
        for param in self.den_faces.parameters():
            param.requires_grad = False 
        self.den_faces.classifier = nn.Linear(self.den_faces.classifier.in_features, hidden_size)
        
        # Mouths
        self.den_mouths = models.densenet201(pretrained=True)
        for param in self.den_mouths.parameters():
            param.requires_grad = False 
        self.den_mouths.classifier = nn.Linear(self.den_mouths.classifier.in_features, hidden_size)

        # Eyes
        self.den_eyes = models.densenet201(pretrained=True)
        for param in self.den_eyes.parameters():
            param.requires_grad = False 
        self.den_eyes.classifier = nn.Linear(self.den_eyes.classifier.in_features, hidden_size)

        ######################### AlexNet ############################
        # Faces
        self.alex_faces = models.alexnet(pretrained=True)
        for param in self.alex_faces.parameters():
            param.requires_grad = False
        self.ale_faces = nn.Linear(1000, hidden_size)
        # Mouths
        self.alex_mouths = models.alexnet(pretrained=True)
        for param in self.alex_mouths.parameters():
            param.requires_grad = False 
        self.ale_mouths = nn.Linear(1000, hidden_size)
        # Eyes
        self.alex_eyes = models.alexnet(pretrained=True)
        for param in self.alex_eyes.parameters():
            param.requires_grad = False
        self.ale_eyes = nn.Linear(1000, hidden_size)

        ##################################################################

        # ReLU layers
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        self.relu7 = nn.ReLU()
        self.relu8 = nn.ReLU()
        self.relu9 = nn.ReLU()

        self.relu_fc1 = nn.ReLU()
        self.relu_fc2 = nn.ReLU()
        self.relu_fc3 = nn.ReLU()

        self.d1 = nn.Dropout(p=0.2)
        self.d2 = nn.Dropout(p=0.2)
        self.d3 = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(3*hidden_size, hidden_size)
        self.fc2 = nn.Linear(3*hidden_size, hidden_size)
        self.fc3 = nn.Linear(3*hidden_size, hidden_size)

        self.output = nn.Linear(3*hidden_size, num_classes)
    
    def forward(self, faces, mouths, eyes):
        # Resnet50
        res_faces = self.relu1(self.res_faces(faces))
        res_mouths = self.relu2(self.res_mouths(mouths))
        res_eyes = self.relu3(self.res_eyes(eyes))

        # DenseNet201
        den_faces = self.relu4(self.den_faces(faces))
        den_mouths = self.relu5(self.den_mouths(mouths))
        den_eyes = self.relu6(self.den_eyes(eyes))

        # AlexNet
        alex_faces = self.relu7(self.alex_faces(faces))
        ale_faces = self.ale_faces(alex_faces)
        alex_mouths = self.relu8(self.alex_mouths(mouths))
        ale_mouths = self.ale_mouths(alex_mouths)
        alex_eyes = self.relu9(self.alex_eyes(eyes))
        ale_eyes = self.ale_eyes(alex_eyes)

        # features
        out1 = self.relu_fc1(self.fc1(torch.cat((res_faces, res_mouths, res_eyes), dim=1)))
        out1 = self.d1(out1)

        out2 = self.relu_fc2(self.fc2(torch.cat((den_faces, den_mouths, den_eyes), dim=1)))
        out2 = self.d2(out2)

        out3 = self.relu_fc3(self.fc3(torch.cat((ale_faces, ale_mouths, ale_eyes), dim=1)))
        out3 = self.d3(out3)

        # Output
        output = self.output(torch.cat((out1, out2, out3), dim=1))

        return output
