import numpy as np
import scipy
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, SVHN
import pickle
from collections import Counter
import random


class DataDomain(Dataset):    
    # Adapted from: https://github.com/adambielski/siamese-triplet/blob/master/datasets.py

    def __init__(self, dataset, datasetname):
        self.dataset = dataset
        self.datasetname = datasetname
        if datasetname == 'svhn':
            self.labels = np.array(self.dataset.labels)
        else:
            self.labels = np.array(self.dataset.targets)
        self.data = self.dataset.data
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}

    def __getitem__(self, index):
        
        # Determines what portion of dataset to sample from.
        mid = len(self.dataset)//2
        if index < mid:
            marker = 'a'  # Domain A is the anchor.
        else:
            marker = 'b'  # Domain B is the anchor.
        
        img1, label1 = self.data[index], self.labels[index]
        positive_index = index
        if marker == 'a':
            while (positive_index == index):
                positive_index = np.random.choice([e for e in self.label_to_indices[label1] if e >= mid])                                    
        elif marker == 'b':
            while (positive_index == index):
                positive_index = np.random.choice([e for e in self.label_to_indices[label1] if e < mid])            
            
        img2 = self.data[positive_index]        
        
        
        if self.datasetname == 'mnist':
            img1 = torch.unsqueeze(img1,0).float()
            img2 = torch.unsqueeze(img2,0).float()             
        elif self.datasetname == 'cifar10':
            img1 = torch.tensor(img1).permute((2,0,1)).float()
            img2 = torch.tensor(img2).permute((2,0,1)).float()                        
        elif self.datasetname == 'svhn':
            img1 = torch.tensor(img1).float()
            img2 = torch.tensor(img2).float()                        
        return (img1, img2, label1, marker)  

    def __len__(self):
        return len(self.dataset)


def cifar10_loaders():
    """Setup CIFAR10 data loaders."""
    
    # Load datasets.
    train_dataset = CIFAR10('data/pytorch_data/CIFAR10', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))
    test_dataset = CIFAR10('data/pytorch_data/CIFAR10', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))

    train_dataset = DataDomain(train_dataset, 'cifar10')
    test_dataset = DataDomain(test_dataset, 'cifar10')

    kwargs = {'num_workers': 8, 'pin_memory': True, 'batch_size': 32, 'shuffle': False} 
    train_loader = DataLoader(train_dataset, **kwargs)
    test_loader = DataLoader(test_dataset, **kwargs)
    
    return train_loader, test_loader


def svhn_loaders():
    """Setup SVHN data loaders."""
    
    # Load datasets.
    train_dataset = SVHN('data/pytorch_data/SVHN', split='train', download=True, transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))
    test_dataset = SVHN('data/pytorch_data/SVHN', split='test', download=True, transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

    train_dataset = DataDomain(train_dataset, 'svhn')
    test_dataset = DataDomain(test_dataset, 'svhn')

    kwargs = {'num_workers': 8, 'pin_memory': True, 'batch_size': 32, 'shuffle': False} 
    train_loader = DataLoader(train_dataset, **kwargs)
    test_loader = DataLoader(test_dataset, **kwargs)
    
    return train_loader, test_loader


def mnist_loaders():
    """Setup MNIST data loaders."""
    
    # Load datasets.
    train_dataset = MNIST('data/pytorch_data/MNIST', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor()
                             ]))
    test_dataset = MNIST('data/pytorch_data/MNIST', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor()
                            ]))

    train_dataset = DataDomain(train_dataset, 'mnist')
    test_dataset = DataDomain(test_dataset, 'mnist')

    kwargs = {'num_workers': 8, 'pin_memory': True, 'batch_size': 32, 'shuffle': False} 
    train_loader = DataLoader(train_dataset, **kwargs)
    test_loader = DataLoader(test_dataset, **kwargs)
    
    return train_loader, test_loader

class UWData(Dataset):
    """Train/Test split for the preprocessed UW dataset."""    
    def __init__(self, dataloc='./data/rgbd_processed_data/preproc_rgbl_data_wdepth.pkl', train=True):        
        self.train = train                        
        # Load all the data.
        with open(dataloc, 'rb') as f:
            data = pickle.load(f)   
        language_data = data['language_data']
        vision_data = np.concatenate((np.squeeze(data['depth_data']),np.squeeze(data['vision_data'])),axis=1)
        object_names = data['object_names']
        instance_names = data['instance_names']

        # Only keep the top classes. Set to all for now.
        classes_to_keep = [a for a,_ in Counter(data['object_names']).most_common(None)]  # None is all.
        indices_to_keep = [idx for idx in range(len(data['object_names'])) if data['object_names'][idx] in classes_to_keep]

        # Train test split.      
        indices_to_keep_train, indices_to_keep_test = train_test_split(indices_to_keep,test_size=0.3,random_state=42,stratify=object_names)

        # Train data.
        self.language_data_train = [language_data[i] for i in indices_to_keep_train]
        self.vision_data_train = [vision_data[i] for i in indices_to_keep_train] 
        self.object_names_train = [object_names[i] for i in indices_to_keep_train] 
        self.instance_names_train = [instance_names[i] for i in indices_to_keep_train] 

        # Test data.
        self.language_data_test = [language_data[i] for i in indices_to_keep_test]
        self.vision_data_test = [vision_data[i] for i in indices_to_keep_test] 
        self.object_names_test = [object_names[i] for i in indices_to_keep_test] 
        self.instance_names_test = [instance_names[i] for i in indices_to_keep_test]                 
        
    def __len__(self):
        if self.train:
            return len(self.object_names_train)
        else:
            return len(self.object_names_test)

    def __getitem__(self, index):
        if self.train:
            return self.language_data_train[index], self.vision_data_train[index], self.object_names_train[index], self.instance_names_train[index]
        else:
            return self.language_data_test[index], self.vision_data_test[index], self.object_names_test[index], self.instance_names_test[index]        

class GLDData(Dataset):
    def __init__(self, dataloc, train=True):
        self.train = train
        with open(dataloc, 'rb') as f:
            data = pickle.load(f)
        language_data = data['language_data']
        vision_data = data['vision_data']
        object_names = data['object_names']
        instance_names = data['instance_names']

        classes_to_keep = [a for a,_ in Counter(data['object_names']).most_common(None)]
        indices_to_keep = [idx for idx in range(len(data['object_names'])) if data['object_names'][idx] in classes_to_keep]

        indices_to_keep_train, indices_to_keep_test = train_test_split(indices_to_keep, test_size=0.3, random_state=42, stratify=object_names)

        self.language_data_train = [language_data[i] for i in indices_to_keep_train]
        self.vision_data_train = [vision_data[i] for i in indices_to_keep_train]
        self.object_names_train = [object_names[i] for i in indices_to_keep_train]
        self.instance_names_train = [instance_names[i] for i in indices_to_keep_train]

        self.language_data_test = [language_data[i] for i in indices_to_keep_test]
        self.vision_data_test = [vision_data[i] for i in indices_to_keep_test]
        self.object_names_test = [object_names[i] for i in indices_to_keep_test]
        self.instance_names_test = [instance_names[i] for i in indices_to_keep_test]

    def __len__(self):
        if self.train:
            return len(self.object_names_train)
        else:
            return len(self.object_names_test)

    def __getitem__(self, index):
        if self.train:
            return self.language_data_train[index], self.vision_data_train[index], self.object_names_train[index], self.instance_names_train[index]
        else:
            return self.language_data_test[index], self.vision_data_test[index], self.object_names_test[index], self.instance_names_test[index]


def uw_loaders(uw_data):
    """Setup UW data loaders."""
    
    # Select language base embedding.
    if uw_data == 'bert':  # BERT.
        dataloc = './data/rgbd_processed_data/preproc_rgbl_data_wdepth.pkl'    
        
    # Load datasets.
    train_dataset = UWData(dataloc=dataloc, train=True) 
    test_dataset = UWData(dataloc=dataloc, train=False) 

    # Setup data loaders.
    kwargs = {'num_workers': 8, 'pin_memory': True, 'batch_size': 1, 'batch_sampler': None, 'shuffle': False}
    train_loader = DataLoader(train_dataset, **kwargs)
    test_loader = DataLoader(test_dataset, **kwargs)
    
    return train_loader, test_loader
    
def gl_loaders(gld_data_location, num_workers=8, pin_memory=True, batch_size=1, batch_sampler=None, shuffle=False):
    with open(gld_data_location, 'rb') as fin:
        data = pickle.load(fin)

    train, test = gl_train_test_split(data, train_percentage=0.8)

    train_data = GLData(train)
    test_data = GLData(test)

    kwargs = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'batch_size': batch_size,
        'batch_sampler': batch_sampler,
        'shuffle': shuffle
    }

    return DataLoader(train_data, **kwargs), DataLoader(test_data, **kwargs)
 
def gl_train_test_split(data, train_percentage=0.8):
    """
    Splits a grounded language dictionary into training and testing sets.

    data needs the following keys:
    language_data
    vision_data
    object_names
    instance_names
    """
    train = {}
    test = {}

    # ensure test and train have some of every object
    train_indices = []
    unique_object_names = list(set(data['object_names']))
    for object_name in unique_object_names:
        train_indices += random.sample(
            [i for i, name in enumerate(data['object_names']) if name == object_name],
            int(train_percentage * data['object_names'].count(object_name))
        )
    test_indices = [i for i in range(len(data['object_names'])) if i not in train_indices]

    train['language_data'] = [data['language_data'][i] for i in train_indices]
    train['vision_data'] = [data['vision_data'][i] for i in train_indices]
    train['object_names'] = [data['object_names'][i] for i in train_indices]
    train['instance_names'] = [data['instance_names'][i] for i in train_indices]

    test['language_data'] = [data['language_data'][i] for i in test_indices]
    test['vision_data'] = [data['vision_data'][i] for i in test_indices]
    test['object_names'] = [data['object_names'][i] for i in test_indices]
    test['instance_names'] = [data['instance_names'][i] for i in test_indices]

    return train, test

class GLData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['object_names'])

    def __getitem__(self, i):
        return self.data['language_data'][i], self.data['vision_data'][i], self.data['object_names'][i], self.data['instance_names'][i]
