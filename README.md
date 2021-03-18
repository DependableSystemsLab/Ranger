# Experimental-artifacts-Ranger

This repo lists the benchmarks and fault injection tool used in the Ranger paper.

- /TensorFI includes the fault injection tool (TensorFI). We've made some changes to the tool (mainly in /TensorFI/injectFault.py), to support the capability of injection in all the benchmarks we've used. 
- /Ranger-benchmarks contains all the DNN benchmarks and experimental infrastructure.
- /dave-driving contains a sample DNN benchmark where we've provided the scripts to automate the deployment of Ranger.


## Installation

  * You'll first need to install TensorFI (see https://github.com/DependableSystemsLab/TensorFI).
  * Then also install opencv and keras.
  * We recommend using condo for installation.

## Implementation of Ranger

We provide different ways to implement Ranger.

1. You can use the script */Ranger-benchmarks/auto-trans/auto-transoform.py* to automatically transform the TensorFlow graph to insert the operators for range restriction. Template for each model is also provided under the same directory.

2. You can also manually insert the restriction operators into the source program where you define the model. See */Ranger-benchmarks/vgg16-Imagenet/bounded-vgg16-model-def.py* for an example.

3. If you don't want to insert the script for automation into your source program, you can also evaluate the effectiveness of Ranger by *simulating Ranger in the fault injection tool*. This is because the tool (TensorFI) *duplicates* the original TensorFlow graph (for the purpose of fault injection), where we can insert range check during fault injection process. Below is an example:

```
1	def relu(input): # this is the customized implementation of ReLu from the TensorFI tool in injectFault.py
2		output = relu(input)
3		output = tf.minimum(output, upBound) # insert range check
4		output = tf.maximum(output, lowBound) # insert range check
5		return output
```

In the above example, we can simulate Ranger in TensorFI, but not on the source TensorFlow program. This is done in */TensorFI/injectFault.py*. See an example in */Ranger-benchmarks/LeNet-mnist/injectFault.py*

Currently, the injectFault.py is customized to different models and they can be found in each model's directory.

In order to perform fault injections, you can use the script starting with `FI` under each benchmark directory. 

- To perform fault injection *without* Ranger enabled, just use the original injectFault.py (this is the one under /TensorFI). 
- To perform the experiment *with* ranger enabled just replace the injectFault.py in the TensorFI module with the one under the benchmark's directory.

To calculate the SDC rates you can use the log files written during the fault injection. Otherwise, you can put counters in the fault injection code to measure the average SDC.


## How to evaluate Ranger

### To evaluate the effectiveness in improving the DNN's error resilience

1. Measure the SDC rate of the original model.
2. Measure the SDC rate of the model deployed with Ranger.
3. Compare the SDC rate of model with and without Ranger

Please refer to method 3 in "Implementation of Ranger" for more details.

You can inject control fault into both models to evaluate the effectiveness of Ranger. Or you can inject multiple faults to get a statistical estimate of the SDC rate on both models.

### To evaluate the impact of Ranger on accuracy

Evaluate the accuracy of the model (with Ranger inserted) on the validation set. The same way as you do to evaluate the common model.

### To evaluate the overhead of Ranger

We evaluate it in terms of FLOPs. The script to evaluate the FLOPs of model with and without Ranger is in /Ranger-benchmarks/auto-trans/overhead-measurement.py. A working example is in /Ranger-benchmarks/LeNet-mnist/overhead-measurement.py

### To evaluate the effectiveness of Ranger under models using reduced-precision datatype

Replace the current *TensorFI/faultTypes.py* with */TensorFI/16-bit-faultTypes.py* and then perform fault injection as before.

### To evaluate the effectiveness of Ranger against multi-bit flips

Change the variable ```numOfCurrFault``` in *TensorFI/faultTypes.py*, e.g., ```numOfCurrFault=2``` to enable 2-bit injection per inference.


## Paper
If you find the repo useful, please cite the following paper: 

*Zitao Chen, Guanpeng Li, Karthik Pattabiraman “A Low-cost Fault Corrector for Deep Neural Networks through Range Restriction” The 51st Annual IEEE/IFIP International Conference on Dependable Systems and Networks (DSN), 2021*
 





