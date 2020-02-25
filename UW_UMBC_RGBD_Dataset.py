
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import re
#import cv2
from skimage import io

class UW_UMBC_RGBD_Dataset(Dataset):

    def __init__(self, csv_file, root_dir, transform_depth=None, transform_rgb=None):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        dataset = pd.read_csv(csv_file)
        self.images = dataset["image"]
        self.descriptions = dataset["description"]
        self.root_dir = root_dir
        self.transform_depth = transform_depth
        self.transform_rgb = transform_rgb

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        description = self.descriptions [index]
        rgb_image_name = self.images[index]
        depth_image_name = self.images[index].replace("crop","depthcrop")
        object_name = self.getObjectName(rgb_image_name)
        instance_name = self.getInstanceName(rgb_image_name)

        depth_image_loc = self.root_dir + "/" + object_name + "/" + instance_name + "/" + depth_image_name
        rgb_image_loc = self.root_dir + "/" + object_name + "/" + instance_name + "/" + rgb_image_name

        #print(rgb_image_loc)
        depth_image = io.imread(depth_image_loc, as_gray=True) #cv2.imread(depth_image_loc, cv2.IMREAD_GRAYSCALE)
        rgb_image =  io.imread(rgb_image_loc, as_gray=False)#cv2.imread(rgb_image_loc)

        if self.transform_depth:
            depth_image = self.transform_depth(depth_image)
        if self.transform_rgb:
            rgb_image = self.transform_rgb(rgb_image)


        return description, rgb_image, depth_image, object_name, instance_name

    def __len__(self):
        return len(self.images)

    def getObjectName(self,picture_name):
        pattern = "([a-z].*[a-z])_\d+"
        return re.search(pattern,picture_name).group(1)

    def getInstanceName(self,picture_name):
        pattern = "([a-z].*[a-z])_\d+"
        return re.search(pattern,picture_name).group(0)


def main():
    dataset = UW_UMBC_RGBD_Dataset("data/UW_UMBC_RGBD_dataset_language_images_processed.csv", "data/rgbd-dataset")
    print(dataset[1027])
    print(len(dataset))

if __name__ == '__main__':
    main()
