import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

from model import ResNet50, ResNet50Features
from datasets import BinaryDatasets, MultiwayDatasets, MultiwaySubDatasets, SampledTestSets


def get_features(class_):
    net = ResNet50Features(2)
    net = net.to("cuda")
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    ckpt = torch.load(f"results/{class_}/last_ckpt.pth")
    net.load_state_dict(ckpt['net'])
    net.eval()
    
    _, testloader = BinaryDatasets([class_])
    n_images = len(testloader.dataset)

    # arrays to hold all features from the model
    all_features = torch.zeros((n_images, 2048)).to("cpu")
    all_labels = np.zeros((n_images))

    count = 0
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(testloader)):
            images = images.to("cuda")
            features = net(images)

            n_images_batch = len(images) # number of images in this batch
            all_features[count : count + n_images_batch, :] = features.cpu() 
            all_labels[count : count + n_images_batch] = labels

            count += n_images_batch

    # reduce dimensionality using PCA first
    pca = PCA(n_components=50)
    all_features_pca = pca.fit_transform(all_features)
    print(pca.explained_variance_ratio_.sum())

    # add embedding
    writer = SummaryWriter(f"tsne/{class_}")
    writer.add_embedding(all_features_pca, metadata=all_labels)
    writer.close()


if __name__ == '__main__':
    get_features("automobile")
