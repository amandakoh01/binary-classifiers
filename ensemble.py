from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn

from datasets import BinaryDatasets, MultiwayDatasets, MultiwaySubDatasets, SampledTestSets
from model import ResNet50

import cfg

# load one model -> run through the entire test dataset -> next model

def ensemble_eval():
    classes = cfg.CLASSES_TO_RUN
    
    # _, testloader = MultiwayDatasets()
    _, testloader = MultiwaySubDatasets(classes)

    net = ResNet50(2)
    net.to("cuda")
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    net.eval()

    softmax = torch.nn.Softmax(1)

    n_images = len(testloader.dataset)

    scores = torch.zeros((n_images, len(classes)))
    gt = torch.zeros((n_images))

    # get target results
    for batch_idx, (_, targets) in enumerate(testloader):
        gt[batch_idx * len(targets) : (batch_idx + 1) * len(targets)] = targets

    # get results from each classifier
    for i, class_ in tqdm(enumerate(classes)):
        ckpt = torch.load(f"checkpoints/10way/{class_}/last_ckpt.pth")
        net.load_state_dict(ckpt['net'])

        scores = predict_with_one_net(net, i, scores, testloader, softmax)

    # consolidate results
    _, predicted = scores.max(1)
    correct = predicted.eq(gt).sum().item()
    accuracy = 100. * correct / n_images

    # ERROR ANALYSIS

    actual_classes = torch.zeros((len(classes)))
    misclassified_classes = torch.zeros((len(classes)))

    counts = [0, 0, 0, 0]

    with open("results.txt", "a") as f:

        for i in range(len(predicted)):
            actual_class = int(gt[i])
            predicted_class = predicted[i]
            actual_class_score = scores[i, actual_class]
            predicted_class_score = scores[i, predicted_class]

            if gt[i] != predicted[i]:
                f.write(f"actual: {actual_class} ({actual_class_score:.4f}), predicted: {predicted_class} ({predicted_class_score:.4f})\n")

                actual_classes[int(gt[i])] += 1
                misclassified_classes[predicted[i]] += 1

                # Actual class classifier predicted wrongly (<0.5) and predicted class classifier predicted wrongly (>0.5)
                if actual_class_score < 0.5 and predicted_class_score > 0.5: 
                    counts[0] += 1

                # Actual class classifier predicted correctly (>0.5), but predicted class classifier predicted wrongly and was more confident (>>0.5, closer to 1)
                elif actual_class_score > 0.5 and predicted_class_score > 0.5:
                    counts[1] += 1

                # Actual class classifier predicted wrongly (<<0.5) and predicted class classifier predicted correctly but was a bit more confident (<0.5, closer to 0.5)
                elif actual_class_score < 0.5 and predicted_class_score < 0.5:
                    counts[2] += 1


            else:
                # All classifiers return a negative score, but class is still predicted correctly (because every class predicted negatively)
                if actual_class_score < 0.5 and predicted_class_score < 0.5:
                    counts[3] += 1
            
    print(counts)
    print(actual_classes)
    print(misclassified_classes)

    return accuracy

def predict_with_one_net(net, class_number, scores, testloader, softmax):
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(testloader):
            inputs.to("cuda")

            outputs = net(inputs)
            probs = softmax(outputs)

            n = len(inputs)

            scores[batch_idx * n : (batch_idx + 1) * n, class_number] = probs[:, 1]

    return scores


if __name__ == '__main__':
    print(ensemble_eval())
