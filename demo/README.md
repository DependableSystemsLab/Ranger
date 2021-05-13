# Autopilot-TensorFlow (copy from https://github.com/SullyChen/Autopilot-TensorFlow)
A TensorFlow implementation of this [Nvidia paper](https://arxiv.org/pdf/1604.07316.pdf) with some changes. For a summary of the design process and FAQs, see [this medium article I wrote](https://medium.com/@sullyfchen/how-a-high-school-junior-made-a-self-driving-car-705fa9b6e860).


# This folder contains the script to automatically insert range restriction in a self-driving DNN.

The first phase is to derive the restriction bounds, and the second phase is to insert range restriction operators in the networks.

## 1. To derive the restriction bounds:


1.1.
Run ```./profiling.sh profiling.yaml get_act_dist.py 25 default.yaml```

This will generate the *code* for deriving the restriction bounds from a specific DNN.

- The first argument: the config file to derive the restriction bounds (e.g., how many data to be used for profiling).
- The second argument: a simple TF program, in which a snippet of code will be inserted for deriving the restriction bounds.
- The third argument: line number in the TF program, after which the new snippet of code will be added.
- The forth argument: The template config file used in the **second** phase to insert Ranger.


1.2.

Run ```python get_act_dist.py```

This will output the restriction values for different layers and also automatically populate these values into the config file for the second phase deployment - this will be outputed as a new file *new.yaml*.

NOTE: You can skip the profiling step and directly go to the second step by using the parameters in *default.yaml*.

## 2. Insert range restriction operators

Run ```./insert.sh default.yaml dave-new.py 38```

*dave-new.py* is a copy of *dave-org.py*, and we'll deploy Ranger into *dave-new.py* and compare it with the original model (*dave-org.py*).

The above script will insert range restriction operators into the network based on the configuration specified in *default.yaml* 

- The first argument: config file 
- The second argument: the original TF program that contains the unprotected DNN
- The third argument: line number in the TF program, after which a snippet of code will be added to insert range restriction operators.

The code for automated insertion is written in *ranger.py*.

## 3. Evaluate the effectiveness of Ranger

```python dave-org.py --input sample_inputs/1000.jpg --isFI true``` 
```python dave-new.py --input sample_inputs/1000.jpg --isFI true```

You can compare the output from *dave-new.py* and that from *Dave-org.py* by injecting the same fault (default setup is provided).
