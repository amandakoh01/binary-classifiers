import pickle
import numpy as np
from PIL import Image
import cv2
import fire

import torchvision, torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

import cfg

############### CREATE DATASETS ###############

def MultiwayDatasets():
    """
    Create normal datasets with all class data
    Uses the built in torch vision datasets, so this downloads data if not present yet

    Returns: trainloader, testloader
    """
    transform_train = transforms.Compose([
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
        root='cifar', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=cfg.TRAIN_BATCH_SIZE, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(
        root='cifar', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=cfg.TEST_BATCH_SIZE, shuffle=False, num_workers=4)

    return trainloader, testloader

def BinaryDatasets(positive_classes, negative_classes=None, transform=True):
    """
    Create binary sub datasets, defining positive classes (1) and negative classes (0)

    positive_classes: list containing strings of classes to be marked as positive
    negative_classes: list containing strings of classes to given as negative train data
    if negative_classes = None, all other classes are given as train data

    All classes, regardless of what negative_classes is defined as, will be given as test data
    
    Trainloader is balanced, testloader is imbalanced

    Returns: trainloader, testloader
    """
    assert isinstance(positive_classes, list), "positive_classes must be a list"
    assert negative_classes is None or isinstance(negative_classes, list), "negative_classes must be None or a list"

    # ensure that the CIFAR-10 image data is present on computer
    MultiwayDatasets()

    # set up number of classes variables (for weightage of train set)
    num_positive = len(positive_classes)
    if negative_classes == None:
        num_negative = 10 - num_positive
    else:
        num_negative = len(negative_classes)
    num_classes = num_positive + num_negative

    ### CREATE TRAIN DATASET
    all_train_images, all_train_labels = get_all_train_data()
    num_train = num_classes * 5000

    sub_train_images = np.zeros((num_train, 32, 32, 3), dtype=np.uint8)
    sub_train_labels = np.zeros((num_train))
    sub_train_weights = np.zeros((num_train))

    count = 0 # to keep track of how many images have been found
    
    for image, label in zip(all_train_images, all_train_labels):

        # positive class
        if cfg.CLASSES[label] in positive_classes:
            sub_train_images[count] = image
            sub_train_labels[count] = 1
            sub_train_weights[count] = num_negative
            count += 1

        # negative class
        elif negative_classes is None or cfg.CLASSES[label] in negative_classes:
            sub_train_images[count] = image
            sub_train_labels[count] = 0
            sub_train_weights[count] = num_positive
            count += 1

        if count == num_train:
            break

    train_dataset = TrainDataset(sub_train_images, sub_train_labels, transform=transform)
    train_sampler = WeightedRandomSampler(sub_train_weights, num_classes * 5000)
    trainloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=cfg.TRAIN_BATCH_SIZE, num_workers=4)

    ### CREATE TEST DATASET
    all_test_images, all_test_labels = get_all_test_data()

    num_test = num_classes * 1000

    sub_test_images = np.zeros((num_test, 32, 32, 3), dtype=np.uint8)
    sub_test_labels = np.zeros((num_test))

    count = 0 # to keep track of how many images have been found
    
    for image, label in zip(all_test_images, all_test_labels):

        # positive class
        if cfg.CLASSES[label] in positive_classes:
            sub_test_images[count] = image
            sub_test_labels[count] = 1
            count += 1

        # negative class
        elif negative_classes is None or cfg.CLASSES[label] in negative_classes:
            sub_test_images[count] = image
            sub_test_labels[count] = 0
            count += 1

        if count == num_test:
            break

    test_dataset = TestDataset(sub_test_images, sub_test_labels, transform=transform)
    testloader = DataLoader(test_dataset, batch_size=cfg.TEST_BATCH_SIZE, num_workers=4)
    
    return trainloader, testloader

def SampledTestSets(positive_classes, negative_classes=None, transform=True):
    assert isinstance(positive_classes, list), "positive_classes must be a list"
    assert negative_classes is None or isinstance(negative_classes, list), "negative_classes must be None or a list"

    # ensure that the CIFAR-10 image data is present on computer
    MultiwayDatasets()

    # set up number of classes variables (for weightage of train set)
    num_positive = len(positive_classes)
    if negative_classes is None:
        num_negative = 10 - num_positive
    else:
        num_negative = len(negative_classes)
    num_classes = num_positive + num_negative

    ### CREATE TEST DATASET
    all_test_images, all_test_labels = get_all_test_data()
    num_test = num_classes * 1000

    sub_test_images = np.zeros((num_test, 32, 32, 3), dtype=np.uint8)
    sub_test_labels = np.zeros((num_test))
    sub_test_weights = np.zeros((num_test))

    count = 0 # to keep track of how many images have been found
    
    for image, label in zip(all_test_images, all_test_labels):

        # positive class
        if cfg.CLASSES[label] in positive_classes:
            sub_test_images[count] = image
            sub_test_labels[count] = 1
            sub_test_weights[count] = num_negative
            count += 1

        # negative class
        elif negative_classes is None or cfg.CLASSES[label] in negative_classes:
            sub_test_images[count] = image
            sub_test_labels[count] = 0
            sub_test_weights[count] = num_positive
            count += 1

        if count == num_test:
            break

    test_dataset = TestDataset(sub_test_images, sub_test_labels, transform=transform)
    test_sampler = WeightedRandomSampler(sub_test_weights, num_positive * 2 * 1000)
    testloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=cfg.TEST_BATCH_SIZE, num_workers=4)
    
    # return None to keep it in line with all other datasets (train, test)
    return None, testloader

def MultiwaySubDatasets(classes_to_include, transform=True):
    """
    Create multiway sub datasets, renumbering labels (e.g. 4 classes with labels 0-3)
    Basically pretends that all other classes do not exist

    classes_to_include: list

    Returns: trainloader, testloader
    """

    assert isinstance(classes_to_include, list), "classes_to_include must be a list"

    # ensure that the CIFAR-10 image data is present on computer
    MultiwayDatasets()

    # create a dictionary mapping the original class labels to their new numerical label
    num_classes = len(classes_to_include)
    class_to_new_label = dict()
    for i, class_ in enumerate(classes_to_include):
        assert class_ in cfg.CLASSES, "class to include must be in original class list"
        class_to_new_label[class_] = i

    ### CREATE TRAIN DATASET
    all_train_images, all_train_labels = get_all_train_data()
    num_train = num_classes * 5000
    sub_train_images = np.zeros((num_train, 32, 32, 3), dtype=np.uint8)
    sub_train_labels = np.zeros((num_train))

    count = 0 # to keep track of how many images have been found

    # loop through all images/labels
    for image, label in zip(all_train_images, all_train_labels):
        if cfg.CLASSES[label] in classes_to_include:
            sub_train_images[count] = image
            sub_train_labels[count] = class_to_new_label[cfg.CLASSES[label]]
            count += 1

        if count == num_train:
            break

    train_dataset = TrainDataset(sub_train_images, sub_train_labels, transform=transform)
    trainloader = DataLoader(train_dataset, batch_size=cfg.TRAIN_BATCH_SIZE, num_workers=4)

    ### CREATE TEST DATASET
    all_test_images, all_test_labels = get_all_test_data()
    num_test = num_classes * 1000
    sub_test_images = np.zeros((num_test, 32, 32, 3), dtype=np.uint8)
    sub_test_labels = np.zeros((num_test))

    count = 0 # to keep track of how many images have been found

    # loop through all images/labels
    for image, label in zip(all_test_images, all_test_labels):
        if cfg.CLASSES[label] in classes_to_include:
            sub_test_images[count] = image
            sub_test_labels[count] = class_to_new_label[cfg.CLASSES[label]]
            count += 1

        if count == num_test:
            break

    test_dataset = TestDataset(sub_test_images, sub_test_labels, transform=transform)
    testloader = DataLoader(test_dataset, batch_size=cfg.TEST_BATCH_SIZE, num_workers=4)

    return trainloader, testloader



############### DATASET CLASSES ###############

class TrainDataset(Dataset):
    def __init__(self, X, y, transform=True):
        self.X = X
        self.y = y
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.apply_transform = transform
 
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, index):
        image = self.X[index]
        if self.apply_transform:
            image = self.transform(Image.fromarray(image))
        y = self.y[index]
        return (image, y)
    
class TestDataset(Dataset):
    def __init__(self, X, y, transform=True):
        self.X = X
        self.y = y
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.apply_transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        image = self.X[index]
        if self.apply_transform:
            image = self.transform(Image.fromarray(image))
        y = self.y[index]
        return (image, y)



############### CHECKING FUNCTIONS ###############

def CheckSubDatasetsCount(positive, negative=None, mode="train"):
    """
    for use only with CreateBinarySubDatasets

    prints out proportion between number of positive elements : number of negative elements

    for train: should be roughly 0.5, no matter the actual ratio between number of classes
    for test: exactly in the ratio of pos : (10-pos) classes
    """

    trainloader, testloader = BinaryDatasets(positive, negative)

    if mode == "train":
        loader = trainloader
    elif mode == "test":
        loader = testloader
    else:
        raise Exception("Mode must be either train or test")

    total_num_positive = 0
    total = 0

    for _, targets in loader:
        num_positive = sum(targets).item()
        num_total = len(targets)

        total_num_positive += num_positive
        total += num_total

        print(f"{num_positive}/{num_total} ({num_positive/num_total})")

    print(f"{total_num_positive}/{total} ({total_num_positive/total})")

def CheckSubDatasetsImages(positive, negative=None, mode="train"):
    # display images with their label (make sure labels are correct and correct classes are included)

    trainloader, testloader = BinaryDatasets(positive, negative, transform=False)
    # trainloader, testloader = MultiwaySubDatasets(positive, transform=False)

    if mode == "train":
        loader = trainloader
    elif mode == "test":
        loader = testloader
    else:
        raise Exception("Mode must be either train or test")

    for inputs, targets in loader:
        for i in range(len(inputs)):
            img = inputs[i].numpy()
            img = img[:,:,::-1] # convert RGB to BGR
            img = cv2.resize(img, (128, 128)) # scale up from 32x32 to 128x128

            label = str(int(targets[i].item()))

            cv2.imshow(label, img)
            cv2.waitKey(0) # waits until a key is pressed



############### HELPER FUNCTIONS ###############
# process original CIFAR10 data

def get_all_train_data():
    """
    process original CIFAR10 train set images from the 5 data batches into numpy lists (images, labels)

    returns:
    - images: 50000 * 32 * 32 * 3 np array
    - labels: list of length 50000, values 0-9, matching the images
    """

    all_train_images = None
    all_train_labels = None

    for i in range(1, 6):
        with open(f"cifar/cifar-10-batches-py/data_batch_{i}", "rb") as f:
            batch_data = pickle.load(f, encoding='bytes')

        if all_train_images is None: # first batch
            all_train_images = batch_data["data".encode()] # 10000x3072 nd array
            all_train_labels = batch_data["labels".encode()] # 10000d list
        else: # concatenate / extend
            all_train_images = np.concatenate((all_train_images, batch_data["data".encode()]))
            all_train_labels.extend(batch_data["labels".encode()])

    all_train_images = np.reshape(all_train_images, (-1, 3, 32, 32))
    all_train_images = np.transpose(all_train_images, (0, 2, 3, 1)) # transpose to 32*32*3

    return all_train_images, all_train_labels

def get_all_test_data():
    """
    process original CIFAR10 test set images from the data batches into numpy lists (images, labels)

    returns:
    - images: 10000 * 32 * 32 * 3 np array
    - labels: list of length 10000, values 0-9, matching the images
    """

    with open("cifar/cifar-10-batches-py/test_batch", "rb") as f:
        test_data = pickle.load(f, encoding='bytes')

    test_images = test_data["data".encode()]
    test_labels = test_data["labels".encode()]
    test_images = np.reshape(test_images, (-1, 3, 32, 32))
    test_images = np.transpose(test_images, (0, 2, 3, 1))

    return test_images, test_labels



if __name__ == '__main__':
    fire.Fire()
