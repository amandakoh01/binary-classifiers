import torch
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

from model import ResNet50
from datasets import CreateDatasets

import numpy as np

from tqdm import tqdm

def get_features(class_):
    net = ResNet50Features(2)

    net = net.to("cuda")
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    ckpt = torch.load(f"results/{class_}/last_ckpt.pth")
    net.load_state_dict(ckpt['net'])

    net.eval()

    _, testloader = CreateSubDatasets([class_])

    all_features = torch.zeros((10000, 2048)).to("cpu")
    all_labels = np.zeros((10000))

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(testloader)):
            images = images.to("cuda")

            features = net(images)

            n = len(num_images)
            all_features[i*n : (i+1)*n, :] = features.cpu()
            all_labels[i*n : (i+1)*n] = labels

    writer = SummaryWriter(f"tsne/{class_}")
    writer.add_embedding(all_features, metadata=all_labels)
    writer.close()