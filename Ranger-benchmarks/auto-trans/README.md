# Automated transformation of TensorFlow graph to insert range checking

This folder contains the code to *automatically* transform the TensorFlow graph to insert range checking. See "illustration.png" for an illustrative example. Check our Ranger paper for more details.


The script for automated transformation is in *auto-transform.py*, you can insert this into your own TensorFlow program at the place after you've loaded the model and are ready to run the model. Below is a pseudo code:

```
1	model = model() # define your model 
2	sess = restore_TF_model # restore the trained model into the session (optional)
3	Insert the script from *auto-transform.py*
4	Evaluate the model (e.g., sess.run(model.output)) # the sess has been replaced with the new graph
```


We also provide working examples for all of the 8 benchmarks used in the Ranger paper. You just need to place the file under the corresponding folders and run it. Please read the comments in the beginning in auto-transform.py if you want to use it in your own TF program.

NOTE: the current script might use the *untrained* model as the purpose is to insert range checking into the model. You can import the trained model file into the program as well (see the above pseudo code).




