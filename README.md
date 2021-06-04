# Binary Classifiers
*Code based on [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar), especially model.py, utils.py and engine.py*

Experiment on whether a n binary classifiers, ensembled (right), leads to better accuracy than a single n-way classifier (left).

![One-vs-rest](https://miro.medium.com/max/700/1*4Ii3aorSLU50RV6V5xalzg.png)

For instance, if we wanted to be able to distinguish between 3 classes, we have the option of:
1. A 3-way classifier that immediately outputs one of the 3 classes
2. 3 binary classifiers, one for each class, that individually outputs the probability that the image contains that specific class. After that, choose the class that returns the highest probability returned.

While our hypothesis was that the binary classifiers would perform better at the expense of computation requirements since each model is more specialised to a task, the results are mixed.

## Methods / Notes

- All classifiers are ResNet-50 (see [models.py](https://github.com/amandakoh01/binary-classifiers/blob/main/model.py)). 
- CIFAR-10 dataset is used, with classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
- For ensembling, we run a softmax layer on top of the final output to obtain the probability that the image contains that specific class. Ensembling is done by seeing which class returns the highest probability. See [ensemble.py](https://github.com/amandakoh01/binary-classifiers/blob/main/ensemble.py) for ensembling code.

A note on datasets:
- For training, we can vary what data we give to the binary classifiers to train on. For instance, say we are trying to differentiate 3 classes - airplane, automobile and bird. When training the "airplane" class, we can give it exactly the 3 classes of data (with airplane images marked as 1 and all other classes marked as 0). We can also give it less data (only "airplane" and "bird") or more ("airplane" and all 9 CIFAR classes).
- For training, we balance the positive/negative classes so that the ratio is approximately 1:1. We can also balance the test set, but using the original non-balanced test set will provide more consistent results for the strength of the model.
- Due to all these variables, there are a number of different types of datasets available - MultiwayDatasets, MultiwaySubDatasets, BinaryDatasets, SampledTestSets. See them in [datasets.py](https://github.com/amandakoh01/binary-classifiers/blob/main/datasets.py), they are (somewhat) documented.

## Results
Here, a summary of findings are provided. Click through each link to find more details and the full results.

In [experiment 1](https://www.notion.so/experiment-1-1-v-9-20026deb276f44bb9ae94772e13d5202), we tested on all 10 classes. 10-way classifier gave an overall accuracy of 94.75, while the ensembled test set gives an accuracy of only 92.66. Per-class accuracy on the binary classifiers are actually quite high (average 98.16), but ensembling seems to introduce errors and results in an overall lower accuracy. t-SNE visualisations (using feature vectors) show good separation on the 10-way classifier, but were unable to separate the points in the binary classifier.

In [experiment 2](https://www.notion.so/experiment-2-1-v-3-5-7da473af9a27456fb8efbbfb11a0f9d3), we split the classes into 2 groups (animals, containing 6 classes; vehicles, containing 4 classes). This time, the binary classifiers outperformed the multiway classifiers. This could potentially be due to the fewer number of classes and thus there are less areas where errors are introduced. 

Also, we find that the binary classifiers trained in experiment 1 with all negative classes data always perform better than the newly trained binary classifiers with only classes in the same group. In other words, even when distinguishing between just vehicles, training "airplane" with all other classes including animals performs better than "airplane" with only other vehicles. This suggests that more data is always better, even if the classes are not being distinguished between.

| Group      | Multiway   | Binary (full dataset) | Binary (only group data provided) |
| ---------- | ---------- | --------------------- | --------------------------------- |
| Animals    | 91.5167    | 91.7000               | 91.0167                           |
| Vehicles   | 95.1500    | 96.5750               | 95.3500                           |

In [experiment 3](https://www.notion.so/experiment-3-1-v-9-c3430ef5c8f94ab0badf337ba353b665), we tested on even more subgroups of classes. In all cases, ensembled binary classifiers perform better than the multiway classifier. Generally it seems that binary classifiers perform better relative to n-way classifiers if there are fewer classes and the classes are closer to each other, but many more experiments will need to be conducted to fully establish this.

| Classes                             | n   | n-way   | Binary  | Delta  |
| ----------------------------------- | --- | ------- | ------- | ------ |
| Bird, cat, deer, dog, frog, horse   | 6   | 91.5167 | 91.7000 | 0.1833 |
| Airplane, automobile, ship, truck   | 4   | 95.1500 | 96.5750 | 1.4250 |
| Automobile, airplane, frog, cat     | 4   | 96.5000 | 97.0000 | 0.5000 |
| Cat, dog, deer, horse               | 4   | 91.1500 | 91.4250 | 0.2750 |
| Airplane, automobile, frog          | 3   | 98.8333 | 99.2000 | 0.3667 |
| Airplane, automobile, ship          | 3   | 97.0333 | 97.8666 | 0.8333 |

## Possible further experiments
1. Do more extensive testing, varying n from 3 to 9, to see how n affects the performance of the classifiers
2. Do more extensive testing with different groups of classes to see how the similarity of classes affects the performance
3. Try different model architectures (instead of ResNet-50)
4. Try a different dataset entirely (instead of CIFAR-10)
5. Try the 10-way test again, this time providing even more data with other classes outside of CIFAR-10.
