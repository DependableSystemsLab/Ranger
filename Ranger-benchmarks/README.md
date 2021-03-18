# This folder contains the benchmarks used in the Ranger paper.

For each benchmark, you can either train the model by yourself; or use the pre-trained weights. We provide the pre-trained weights of all the models except ResNet and VGG16 whose models are too big (the link to download for these two are provided). We provide a link to the GTSRB dataset (in pickle format) and the pre-trained weights for ResNet-18: https://drive.google.com/file/d/1chWSXom-lLfmwgxVxP2jU-SCD8FUx5H-/view?usp=sharing

The config files for fault injection are provided in each model's directory.


- Link to the pre-trained weights from the original repos.
    - VGG16 - http://www.cs.toronto.edu/~frossard/post/vgg16/
    - ResNet-18 https://github.com/dalgu90/resnet-18-tensorflow

- Link to the dataset. 
    - LeNet-4 - Mnist dataset (http://yann.lecun.com/exdb/mnist/) 
    - AlexNet - Cifar-10 (https://www.cs.toronto.edu/~kriz/cifar.html)
    - VGG16 - ImageNet (http://image-net.org/download)
    - SqueezeNet - ImageNet (http://image-net.org/download)
    - ResNet-18 - ImageNet (http://image-net.org/download)
    - VGG11 - German traffic sign (http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
    - Comma.ai's steering model - real-world driving frame (https://github.com/SullyChen/driving-datasets)
    - Nvidia Dave steering model - real-world driving frame (https://github.com/SullyChen/driving-datasets)


The restriction values in each model in the Ranger paper is as below:

1. LeNet: upper bound: 3, 4, 8; lower bound: 0, 0, 0
2. AlexNet: upper bound: 1, 2, 2, 3, 4, 9; lower bound: 0, 0, 0, 0, 0
3. VGG11: upper bound: 5, 10, 17, 27, 44, 108, 147, 292, 432, 200; lower bound: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
4. VGG16: upper bound: 956, 5118, 9870, 16892, 27154, 20696, 25586, 19959, 9975, 5791, 4118, 2379, 869, 145, 42; lower bound: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
5. ResNet-18: upper bound: 7, 8, 7, 5, 11, 5, 12, 6, 11, 5, 12, 5, 14, 5, 12, 5, 66; lower bound: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
6. Dave-driving (using steering angle as output): upper bound: 2.8, 26, 39, 94, 164, 562, 2538, 2368, 2431; lower bound: 0, 0, 0, 0, 0, 0, 0, 0, 0
7. Comma-driving: upper bound: 2, 3, 1, 1; lower bound: -1, -1, -1, -1
8. SqueezeNet: upper bound: 958, 2018, 1587, 2470, 1892, 1228, 2003, 3024, 1736, 3031, 2687, 1408, 1903, 3329, 1661, 2285, 2749, 1041, 1884, 2126, 823, 1772, 2306, 748, 1086, 700; lower bound: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0


**/auto-trans/** contains the files to automatically transform the TensorFlow graph to insert the restriction ops.


## How to perform fault injection for evaluation

There are two ways to evaluate the models depending on how you implement Ranger:

1. If you implement Ranger using the automation script, you'll just need to evaluate the operator under the new graph (make sure you're using the new graph and evaluating the ops from the new graph). An working example is in */LeNet-mnist/FI-lenet-autoTrans.py*

2. If you implement Ranger via TensorFI (in injectFault.py), you can replace the injectFault.py file with the one in the TensorFI directory. Each benchmark folder has a injectFault.py file where Ranger is implemented. *NOTE*: for SqueezeNet model, use the TensorFI module under ./squeezenet-model/squeezenet-TensorFI/*. 
./squeezenet-model/squeezenet-TensorFI/ranger-injectFault.py is implemented with Ranger. 

How to run: run the file with prefix `FI` (e.g., FI-lenet.py, FI-alexnet.py). For ResNet-18, run FI-eval.sh (which triggers FI-eval.py).

To see how Ranger is implemented in injectFault.py, you can see line-596, line-611 and line-489 of LeNet-mnist/injectFault.py for an example.







