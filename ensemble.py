from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn

from datasets import BinaryDatasets, MultiwayDatasets, MultiwaySubDatasets, SampledTestSets
from model import ResNet50

from cfg import *

# load one model -> run through the entire test dataset -> next model

classes = ['airplane', 'automobile', 'ship', 'truck']

def ensemble_eval():
    _, testloader = MultiwayDatasets()
    # _, testloader = MultiwaySubDatasets(['airplane', 'automobile', 'ship', 'truck'])

    net = ResNet50(2)
    net.to("cuda")
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    net.eval()

    n_images = len(testloader.dataset)

    scores = torch.zeros((n_images, len(classes)))
    gt = torch.zeros((n_images))

    # get target results
    for batch_idx, (_, targets) in enumerate(testloader):
        gt[batch_idx * len(targets) : (batch_idx + 1) * len(targets)] = targets

    # get results from each classifier
    for i, class_ in tqdm(enumerate(classes)):
        ckpt = torch.load(f"results/{class_}/last_ckpt.pth")
        net.load_state_dict(ckpt['net'])
        print(f"{class_} best acc: {ckpt['acc']}")

        scores = predict_with_one_net(net, i, scores, testloader)

    # consolidate results
    _, predicted = scores.max(1)
    correct = predicted.eq(gt).sum().item()
    accuracy = 100. * correct / n_images

    # ERROR ANALYSIS

    # actual_classes = torch.zeros((10))
    # misclassified_classes = torch.zeros((10))

    # error_counts = [0, 0, 0]

    # for i in range(len(predicted)):
    #     if gt[i] == predicted[i]:
    #         actual_class = int(gt[i])
    #         predicted_class = predicted[i]
    #         actual_class_score = scores[i, actual_class]
    #         predicted_class_score = scores[i, predicted_class]
    #         if actual_class_score < 0:
    #             print(scores[i])
    #             error_counts[0] += 1

    #         print(f"actual: {actual_class} ({actual_class_score:.4f}), predicted: {predicted_class} ({predicted_class_score:.4f})")

    #         if actual_class_score < 0:
    #             print(scores[i])

    #         if actual_class_score < 0 and predicted_class_score > 0:
    #             error_counts[0] += 1
    #         elif actual_class_score > 0 and predicted_class_score > 0:
    #             error_counts[1] += 1
    #         elif actual_class_score < 0 and predicted_class_score < 0:
    #             error_counts[2] += 1

    #         actual_classes[int(gt[i])] += 1
    #         misclassified_classes[predicted[i]] += 1

    # print(actual_classes)
    # print(misclassified_classes)

    return accuracy

def predict_with_one_net(net, class_number, scores, testloader):
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(testloader):
            inputs.to("cuda")

            outputs = net(inputs)

            n = len(inputs)

            scores[batch_idx * n : (batch_idx + 1) * n, class_number] = outputs[:, 1]

    return scores


if __name__ = '__main__':
    print(ensemble_eval())
