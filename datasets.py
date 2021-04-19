import pickle
import numpy as np
from PIL import Image
import cv2

import torchvision, torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from cfg import *

"""
- CreateDatasets - returns trainloader and testloader for the entire original CIFAR10 dataset
- CreateSubDatasets - takes in a list of positive classes and negative classes (or None) and 
                      returns trainloader and testloader
- (classes) TrainDataset, TestDataset - used in CreateSubDatasets
- CheckSubDatasetsCount - count number of pos vs neg images in a data loader (should be 0)
- CheckSubDatasetsImage - visualise images
"""

# Create normal datasets with all class data
def CreateDatasets():
    transform_train = transforms.Compose([
        transforms.ColorJitter(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='../cifar', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='../cifar', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=50, shuffle=False, num_workers=2)

    return trainloader, testloader

# Create sub datasets, defining positive classes (1) and negative classes (0)
def CreateSubDatasets(positive_classes, negative_classes = None):
    """
    positive_classes: list containing strings of classes to be marked as positive
    negative_classes: list containing strings of classes to given as negative train data
    if negative_classes = None, all other classes are given as train data
    """
    assert type(positive_classes) == list, "positive_classes must be a list"
    assert negative_classes == None or type(negative_classes) == list, "negative_classes must be None or a list"

    CreateDatasets() # to ensure that the CIFAR-10 image data is present on computer

    num_positive = len(positive_classes)
    if negative_classes == None:
        num_negative = 10 - num_positive
    else:
        num_negative = len(negative_classes)
    num_classes = num_positive + num_negative
    
    ### SORT THROUGH TRAIN IMAGES
    # at the end of this, 
    # images: 50000 * 32 * 32 * 3 np array
    # labels: list of length 50000, values 0-9, matching the images

    all_train_images = None
    all_train_labels = None

    for i in range(1, 6):
        with open(f"../cifar/cifar-10-batches-py/data_batch_{i}", "rb") as f:
            batch_data = pickle.load(f, encoding='bytes')

        if all_train_images is None: # first batch
            all_train_images = batch_data["data".encode()] # 10000x3072 nd array
            all_train_labels = batch_data["labels".encode()] # 10000d list
        else: # concatenate / extend
            all_train_images = np.concatenate((all_train_images, batch_data["data".encode()]))
            all_train_labels.extend(batch_data["labels".encode()])

    all_train_images = np.reshape(all_train_images, (-1, 3, 32, 32))
    all_train_images = np.transpose(all_train_images, (0, 2, 3, 1)) # transpose to 32*32*3

    ### CREATE TRAIN DATASET
    num_train = num_classes * 5000

    sub_train_images = np.zeros((num_train, 32, 32, 3), dtype=np.uint8)
    sub_train_labels = np.zeros((num_train))
    sub_train_weights = np.zeros((num_train))

    count = 0 # to keep track of how many images have been found
    
    # loop through all images/labels
    for image, label in zip(all_train_images, all_train_labels):

        # positive class
        if classes[label] in positive_classes:
            sub_train_images[count] = image
            sub_train_labels[count] = 1
            sub_train_weights[count] = num_negative
            count += 1

        # negative class
        elif negative_classes == None or classes[label] in negative_classes:
            sub_train_images[count] = image
            sub_train_labels[count] = 0
            sub_train_weights[count] = num_positive
            count += 1

        if count == num_train:
            break

    train_dataset = TrainDataset(sub_train_images, sub_train_labels)
    train_sampler = WeightedRandomSampler(sub_train_weights, num_classes * 5000)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=256, num_workers=2)

    ### CREATE TEST DATASET
    # note all images are included in test set, not just specified negative classes, but imbalanced!
    with open("../cifar/cifar-10-batches-py/test_batch", "rb") as f:
        test_data = pickle.load(f, encoding='bytes')
    
    test_images = test_data["data".encode()]
    test_labels = test_data["labels".encode()]
    test_images = np.reshape(test_images, (-1, 3, 32, 32))
    test_images = np.transpose(test_images, (0, 2, 3, 1))

    # change all test labels to either 0 or 1
    test_labels = [1 if classes[label] in positive_classes else 0 for label in test_labels]

    test_dataset = TestDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=100, num_workers=2)
    
    return train_loader, test_loader

class TrainDataset(Dataset):
  def __init__(self, X, y):
    self.X = X
    self.y = y
    self.transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
 
  def __len__(self):
    return len(self.X)
 
  def __getitem__(self, index):
    image = self.X[index]
    image = self.transform(Image.fromarray(image))
    y = self.y[index]
    return (image, y)
 
class TestDataset(Dataset):
  def __init__(self, X, y):
    self.X = X
    self.y = y
    self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
 
  def __len__(self):
    return len(self.X)
 
  def __getitem__(self, index):
    image = self.X[index]
    # image = self.transform(Image.fromarray(image))
    y = self.y[index]
    return (image, y)



def CheckSubDatasetsCount(positive, negative = None, mode="train"):
    # show proportion between number of positive elements : number of negative elements
    # for train: should be roughly 0.5, no matter the actual ratio between number of classes
    # for test: exactly in the ratio of pos : 10-pos classes
    
    trainloader, testloader = CreateSubDatasets(positive, negative)

    if mode == "train":
        loader = trainloader
    elif mode == "test":
        loader = testloader
    else:
        raise Exception("Mode must be either train or test")

    total_num_positive = 0
    total = 0

    for inputs, targets in loader:
        num_positive = sum(targets).item()
        num_total = len(targets)

        total_num_positive += num_positive
        total += num_total

        print(f"{num_positive}/{num_total} ({num_positive/num_total})")

    print(f"{total_num_positive}/{total} ({total_num_positive/total})")

def CheckSubDatasetsImages(positive, negative = None, mode="train"):
    # display images with their label (make sure labels are correct and correct classes are included)
    # in datasets, need to disable self.transform lines when returning images (to prevent augmentation and normalisation)

    trainloader, testloader = CreateSubDatasets(positive, negative)

    if mode == "train":
        loader = trainloader
    elif mode == "test":
        loader = testloader
    else:
        raise Exception("Mode must be either train or test")

    for inputs, targets in loader:
        for i in range(len(inputs)):
            img = inputs[i].numpy()
            label = str(int(targets[i].item()))

            img = cv2.resize(img, (128, 128))

            cv2.imshow(label, img)

            cv2.waitKey(0) # waits until a key is pressed