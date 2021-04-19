# Experimental artifacts in the paper “A Low-cost Fault Corrector for Deep Neural Networks through Range Restriction”

This repo lists the benchmarks and fault injection tool used in the Ranger paper.

- /TensorFI includes the fault injection tool (TensorFI). We've made some changes to the tool (mainly in /TensorFI/injectFault.py), to support the capability of injection in all the benchmarks we've used. 
- /Ranger-benchmarks contains all the DNN benchmarks and experimental infrastructure.
- /dave-driving contains a sample DNN benchmark where we've provided the scripts to automate the deployment of Ranger. More details are available in README file in the /dave-driving directory.


## Installation

  * You'll first need to install TensorFI (see https://github.com/DependableSystemsLab/TensorFI).
  * Then also install opencv and keras.
  * We recommend using conda for installation.

## Implementation of Ranger

We provide different ways to implement Ranger.

1. You can use the script /Ranger-benchmarks/auto-trans/auto-transform.py to automatically transform the TensorFlow graph to insert the operators for range restriction. Template for each model is also provided under the same directory.

2. You can also manually insert the restriction operators into the source program where you define the model. See */Ranger-benchmarks/vgg16-Imagenet/bounded-vgg16-model-def.py* for an example.

3. If you don't want to insert the script for automation into your source program, you can also evaluate the effectiveness of Ranger by *simulating Ranger in the fault injection tool*. This is because the tool (TensorFI) *duplicates* the original TensorFlow graph (for the purpose of fault injection), where we can insert range check during fault injection process. Below is an example:

```
1	def relu(input): # this is the customized implementation of ReLu from the TensorFI tool in injectFault.py
2		output = relu(input)
3		output = tf.minimum(output, upBound) # insert range check
4		output = tf.maximum(output, lowBound) # insert range check
5		return output
```

In the above example, we can simulate Ranger in TensorFI, but not on the source TensorFlow program. This is done in /TensorFI/injectFault.py. See an example in /Ranger-benchmarks/LeNet-mnist/injectFault.py

Currently, the injectFault.py is customized to different models and they can be found in each model's directory.

In order to perform fault injections, you can use the script starting with `FI` under each benchmark directory. 

- To perform fault injection *without* Ranger enabled, run the Python program in the following format: `FI_model_org.py` (under each benchmark's directory)
- To perform the experiment *with* ranger enabled, run the Python program in the following format: `FI_model_ranger.py` (under each benchmark's directory)


To calculate the SDC rates you can use the log files written during the fault injection. Otherwise, you can put counters in the fault injection code to measure the average SDC.


## Evaluation of Ranger
(The following commands use the LeNet model and assume you're under the following directory: /Ranger-benchmarks/LeNet-mnist/)


### 1. To evaluate the effectiveness in improving the DNN's error resilience
```
python FI-lenet-org.py
python FI-lenet-ranger.py
```
You can compare the SDC rate of the model before and after deploying Ranger.

In the LeNet example, you can compare the SDC rates in lenet-randomFI-org.csv and lenet-randomFI-ranger.csv

### 2. To evaluate the impact of Ranger on accuracy

Evaluate the accuracy of the model (with Ranger inserted) on the validation set. The same way as you do to evaluate the common model.

```
python lenet-accuracy-org.py
python lenet-accuracy-ranger.py # NOTE: you'll need to set the fault type as "None" in LeNet-mnist/confFiles/default.yaml
```

You can compare the accuracy of the model before and after deploying Ranger.

### 3. To evaluate the overhead of Ranger

We evaluate it in terms of FLOPs. The script to evaluate the FLOPs of model with and without Ranger is in /Ranger-benchmarks/auto-trans/overhead-measurement.py. 

```
python lenet-ranger-overhead.py
```
You can compare the FLOPs of the model before and after deploying Ranger.

### 4. To evaluate the effectiveness of Ranger under models using reduced-precision datatype

Replace the current benchmark_folder/TensorFI/faultTypes.py with benchmark_folder/TensorFI/16-bit-faultTypes.py and then perform fault injection as before.

```
# In the LeNet example, the default setup is provided so you don't need to replace the file manually. 

python FI-lenet-org-16bit.py
python FI-lenet-ranger-16bit.py
```

### 5. To evaluate the effectiveness of Ranger against multi-bit flips

Change the variable ```numOfCurrFault``` in benchmark_folder/TensorFI/faultTypes.py and benchmark_folder/TensorFI_ranger/faultTypes.py, e.g., ```numOfCurrFault=2``` to enable 2-bit injection per inference. And then you can perform injection experiment as before.

```
# In the LeNet example, the default setup for 2-bit injection is provided so you don't need to modify the variable manually. 

python FI-lenet-org-multi-bit.py
python FI-lenet-ranger-multi-bit.py
```

## Paper
If you find the repo useful, please cite the following paper: 

*Zitao Chen, Guanpeng Li, Karthik Pattabiraman “A Low-cost Fault Corrector for Deep Neural Networks through Range Restriction” The 51st Annual IEEE/IFIP International Conference on Dependable Systems and Networks (DSN), 2021*
 





