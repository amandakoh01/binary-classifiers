import torch
import torch.backends.cudnn as cudnn

from model import ResNet50
from datasets import BinaryDatasets, MultiwaySubDatasets, MultiwayDatasets
from engine import test

_, testloader = BinaryDatasets(['cat'])
# _, testloader = BinaryDatasets(['bird'], ['cat', 'deer', 'dog', 'frog', 'horse'])
# # _, testloader = MultiwaySubDatasets(['airplane', 'automobile', 'ship', 'truck'])

net = ResNet50(2)
net.to("cuda")
net = torch.nn.DataParallel(net)
cudnn.benchmark = True
net.eval()

for class_ in cfg.CLASSES_TO_RUN:
    ckpt = torch.load(f"checkpoints/10way/{class_}/last_ckpt.pth")

    net.load_state_dict(ckpt['net'])

    criterion = torch.nn.CrossEntropyLoss()

    # neg = cfg.CLASSES_TO_RUN.copy()
    # neg.remove(class_)
    # _, testloader = BinaryDatasets([class_], neg)

    test_loss, acc = test(net, criterion, testloader, "cuda")

    print(f"{acc:.4f}")