# Automated transformation of TensorFlow graph to insert range checking

This folder contains the code to *automatically* transform the TensorFlow graph to insert range checking. See "illustration.png" for an illustrative example. Check our Ranger paper for more details.

We provide working examples for all of the 8 benchmarks used in the Ranger paper. **You'll need to place the file under the corresponding benchmark folders and run it** (e.g., place *lenet.py* under *path_to_Ranger/Ranger-benchmarks/LeNet-mnist*). Please read the comments in the beginning in auto-transform.py if you want to use it in your own TF program.





