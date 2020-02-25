import torch
import torch.nn.functional as F
import torchvision


class MNISTEmbeddingNet(torch.nn.Module):
    """Convolutional network for embedding MNIST image data."""
    def __init__(self, embed_dim=128):
        super(MNISTEmbeddingNet, self).__init__()
        self.embed_dim = embed_dim
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size = 5) 
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size = 3) 
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size = 3)
        self.conv4 = torch.nn.Conv2d(256, embed_dim, kernel_size = 2)
        self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.conv4(x) 
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.squeeze(x)
        return x
    
    
class CIFAREmbeddingNet(torch.nn.Module):
    """ResNet50 for embedding 3 channel image data."""
    def __init__(self, embed_dim=1024):
        super(CIFAREmbeddingNet, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=False)
        self.resnet.fc = torch.nn.Linear(2048, embed_dim)

    def forward(self, x):
        return self.resnet(x)
    

class RowNet(torch.nn.Module):

    def __init__(self, input_size, embed_dim=1024):
        # Language (BERT): 3072, Vision+Depth (ResNet152): 2048 * 2.
        super(RowNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, input_size)
        self.fc2 = torch.nn.Linear(input_size, input_size)
        self.fc3 = torch.nn.Linear(input_size, embed_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




    
