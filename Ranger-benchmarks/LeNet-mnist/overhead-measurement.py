#!/usr/bin/python

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import argparse
import gzip
import os
import sys
import time
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import TensorFI as ti

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = int(60000 * 0.1)  # Size of the validation set.
                                        # consider 70% train; 20% val and 10% test
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 256


NUM_EPOCHS = 30

EVAL_BATCH_SIZE = 1
EVAL_FREQUENCY = 20  # Number of steps between evaluations.


FLAGS = None


def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  if FLAGS.use_fp16:
    return tf.float16
  else:
    return tf.float32


def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(WORK_DIRECTORY):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels


def fake_data(num_images):
  """Generate a fake dataset that matches the dimensions of MNIST."""
  data = numpy.ndarray(
      shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
      dtype=numpy.float32)
  labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels


def error_rate(predictions, labels, isPrint=False):
  """Return the error rate based on dense predictions and sparse labels."""
#  if(isPrint):
#    print("pred :", numpy.argmax(predictions, 1))
#    print("label:", labels) 

  res = (numpy.argmax(predictions, 1) == labels)
  indexOfCorrectSample = numpy.where(res == True)[0]
 

  return (100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0]))  , indexOfCorrectSample


def main(_):
  if FLAGS.self_test:
    print('Running self-test.')
    train_data, train_labels = fake_data(256)
    validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE)
    test_data, test_labels = fake_data(EVAL_BATCH_SIZE)
    num_epochs = 1
  else:
    # Get the data.
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
 
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000) 


    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]
    num_epochs = NUM_EPOCHS

  train_size = int(train_labels.shape[0] )
  print ("Training size ", train_size)

  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  train_data_node = tf.placeholder(
                                    data_type(),
                                    shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
  eval_data = tf.placeholder(
                              data_type(),
                              shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when we call:
  # {tf.global_variables_initializer().run()}



  depth1 = 32
  conv1_weights = tf.Variable(
                              tf.truncated_normal([5, 5, NUM_CHANNELS, depth1],  # 5x5 filter, depth 32.
                                                  stddev=0.1,
                                                  seed=SEED, dtype=data_type()))
  conv1_biases = tf.Variable(tf.zeros([depth1], dtype=data_type()))

  depth2 = 64
  conv2_weights = tf.Variable(
                              tf.truncated_normal([5, 5, depth1, depth2], stddev=0.1,
                              seed=SEED, dtype=data_type()))
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[depth2], dtype=data_type()))
  

  depth3 = 512
  fc1_weights = tf.Variable(  # fully connected, depth 512.
                            tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * depth2, depth3],
                                                stddev=0.1,
                                                seed=SEED,
                                                dtype=data_type()))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[depth3], dtype=data_type()))

  fc2_weights = tf.Variable(tf.truncated_normal([depth3, NUM_LABELS],
                                                stddev=0.1,
                                                seed=SEED,
                                                dtype=data_type()))
  fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=data_type()))




  # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.
  def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    # conv output : (samples, rows, cols, channels) 
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    tanh = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(tanh,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

    
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    tanh = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(tanh,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    

    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

  # Training computation: logits + cross-entropy loss.
  logits = model(train_data_node, True)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=train_labels_node, logits=logits))

  # L2 regularization for the fully connected parameters.
  regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                  tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
  # Add the regularization term to the loss.
  loss += 5e-4 * regularizers

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  batch = tf.Variable(0, dtype=data_type())
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      0.01,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)
  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(loss, global_step=batch)

  # Predictions for the current training minibatch.
  train_prediction = tf.nn.softmax(logits)

  global eval_prediction
  # Predictions for the test and validation, which we'll compute less often.
  eval_prediction = tf.nn.softmax(model(eval_data))



  " Use one batch for the whole test dataset"
  # Small utility function to evaluate a dataset by feeding batches of data to
  # {eval_data} and pulling the results from {eval_predictions}.
  # Saves memory and enables this to run on smaller GPUs.
  def eval_in_batches(data, sess, eval_batch_size=EVAL_BATCH_SIZE):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < eval_batch_size:
      raise ValueError("batch size for evals larger than dataset: %d" % size)

    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, eval_batch_size):
      end = begin + eval_batch_size
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-eval_batch_size:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions





  checkpointer = tf.train.Saver()
  _SAVE_PATH_ = 'checkpoint/lenet.ckpt'

  # Create a local session to run the training.
  start_time = time.time() 
  sess = tf.Session() 

  '''
  # train the model
  learning = open("learning_progress.csv" , "a")
  # Run all the initializers to prepare the trainable parameters.
  tf.global_variables_initializer().run()
  print('Initialized!')
  # Loop through training steps.
  for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
    # Compute the offset of the current minibatch in the data.
    # Note that we could use better randomization across epochs.
    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
    batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
    batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
    # This dictionary maps the batch data (as a numpy array) to the
    # node in the graph it should be fed to.
    feed_dict = {train_data_node: batch_data,
                 train_labels_node: batch_labels}
    # Run the optimizer to update weights. 

    sess.run(optimizer, feed_dict=feed_dict)
    
    # print some extra information once reach the evaluation frequency
    if step % EVAL_FREQUENCY == 0:
      # fetch some extra nodes' data
      l, lr, predictions = sess.run([loss, learning_rate, train_prediction],
                                    feed_dict=feed_dict)
      elapsed_time = time.time() - start_time
      start_time = time.time()
      print('Step %d (epoch %.2f), %.1f ms' %
            (step, float(step) * BATCH_SIZE / train_size,
             1000 * elapsed_time / EVAL_FREQUENCY))

#        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
#        print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
      val_error , _= error_rate(eval_in_batches(validation_data, sess), validation_labels)

      print('Val error: %.1f%%' % val_error)
      learning.write(`float(step) * BATCH_SIZE / train_size` + "," + `step` + "," + `val_error` + "\n" )

      sys.stdout.flush()

      checkpointer.save(sess, save_path=_SAVE_PATH_)
  '''

  "restore the trained model"
  checkpointer.restore(sess, save_path=_SAVE_PATH_)

 
  "============================================================================================================"
  "=================  Begin to insert restriction on selective layers  ========================================"
  "============================================================================================================"

  # get all the operators in the graph
  ops = [tensor for op in sess.graph.get_operations() for tensor in op.values()]
  graph_def = sess.graph.as_graph_def()

  def get_op_dependency(op):
    "get all the node that precedes the target op"
    cur_op = []
    #op = sess.graph.get_tensor_by_name("ranger_11/ranger_10/ranger_9/ranger_8/ranger_7/ranger_6/ranger_5/ranger_4/ranger_3/ranger_2/ranger_1/ranger/Relu_5:0").op
    a = open('alex-dep.txt', "w")
    cur_op.append(op)
    next_op=[]
    while(not (next_op==[] and cur_op==[])):
        next_op = []
        for each in cur_op:
            printline = False
            for inp in each.inputs:
                printline = True
                #print(inp)
                a.write(str(inp) + "\n")
                next_op.append(inp.op) 
            if(printline): 
              #print('')
              a.write("\n\n")
        cur_op = next_op 

  def get_target_scope_prefix(scope_name, dup_cnt, dummy_scope_name, dummy_graph_dup_cnt):
    "get the scope prefix of the target path (the latest duplicated path)"
    target_graph_prefix = "" # the scope prefix of the latest path
    if(dup_cnt==0):
        target_graph_prefix = "" #
    elif(dup_cnt==1):
        target_graph_prefix = str(scope_name + "/" ) # e.g., ranger/relu:0
        if(dummy_graph_dup_cnt==1):
          target_graph_prefix = dummy_scope_name + "/" + target_graph_prefix # e.g., dummy/ranger/relu:0
    else:
        target_graph_prefix = str(scope_name + "/" ) 
   
        if(dummy_graph_dup_cnt>0):  # e.g., dummy/ranger/relu:0
          target_graph_prefix = dummy_scope_name + "/" + target_graph_prefix
          dummy_graph_dup_cnt -= 1 

        for i in range(1, dup_cnt):
            target_graph_prefix = scope_name  + "/" + target_graph_prefix  # e.g., ranger/dummy/ranger/ relu 
            if(dummy_graph_dup_cnt>0):
              target_graph_prefix = dummy_scope_name + "/" + target_graph_prefix # e.g., dummy/ranger/dummy/ranger/relu:0
              dummy_graph_dup_cnt-=1

    return target_graph_prefix

  def restore_trainable_var(sess, scope_name, dup_cnt, train_var, dummy_scope_name, dummy_graph_dup_cnt, OLD_SESS):
    "need to map back the variable values to the ones under the new scope" 

    target_graph_prefix = get_target_scope_prefix(scope_name,dup_cnt,dummy_scope_name,dummy_graph_dup_cnt)

    tmp = []
    for each in train_var: 
      tmp.append( tf.assign(sess.graph.get_tensor_by_name(target_graph_prefix + each.name), OLD_SESS.run( OLD_SESS.graph.get_tensor_by_name( each.name ) )) )
    sess.run(tmp) 

  def get_op_with_prefix(op_name, dup_cnt, scope_name, dummy_graph_dup_cnt, dummy_scope_name):
    "Need to call this function to return the name of the ops under the NEW graph (with scope prefix)"
    "return the name of the duplicated op with prefix, a new scope prefix upon each duplication"
    op_name = get_target_scope_prefix(scope_name,dup_cnt,dummy_scope_name,dummy_graph_dup_cnt) + op_name

    return op_name

  import re
  def modify_graph(sess, dup_cnt, scope_name, prefix_of_bound_op, dummy_graph_dup_cnt, dummy_scope_name): 
    "Modify the graph def to: 1) remove the nodes from older paths (we only need to keep the latest path)"
    " and 2) modify the input dependency to only associate with the latest path" 
    graph_def = sess.graph.as_graph_def() 
   
    target_graph_prefix = get_target_scope_prefix(scope_name,dup_cnt,dummy_scope_name,dummy_graph_dup_cnt)

    #print('target prefix ==> ', target_graph_prefix, dup_cnt)


    # Delete nodes from the redundant paths, we only want the most recent path, otherwise the size of graph will explode
    nodes = [] 
    for node in graph_def.node: 
        if target_graph_prefix in node.name and prefix_of_bound_op not in node.name: # ops to be kept, otherwise removed from graph        
          nodes.append(node)  

        elif(prefix_of_bound_op in node.name):

          if( dup_cnt != graph_dup_cnt ):
            "this part should keep the new op from the most recent duplication (with lesser prefix)"
            if( target_graph_prefix not in node.name ): # remove dummy nodes like dummy/op
              nodes.append(node) 

          else: 
            nodes.append(node)

          # remove dummy nodes like dummy/dummy/relu
          if( dummy_scope_name+"/"+dummy_scope_name+"/" in node.name ):
            nodes.remove(node)


    #print(' ', dup_cnt, dummy_graph_dup_cnt)

    mod_graph_def = tf.GraphDef()
    mod_graph_def.node.extend(nodes) 
     

    "For the newly created op, we need to rewire the input dependency so that it only relies on the latest graph"
    "because we've only kpet the latest graph in the modified graphdef. "
    "This is for the restriction op, e.g., tf.maximum(relu_1, 100), where relu_1 is from the PREVIOUS graph"
    # Delete references to deleted nodes, 
    for node in mod_graph_def.node: 
        inp_names = []
        if(prefix_of_bound_op in node.name): # only for the restriction op
            for inp in node.input:
                if prefix_of_bound_op in inp or target_graph_prefix in inp:
                    inp_names.append(inp)
                else: 
                    #print(node.name, inp, ' ---> ', (scope_name + "_" + str(dup_cnt-1) + "/" + inp) )
                    "here because we copy the graghdef from the PREVIOUS graph, it has dependency to the PREVIOUS graph"
                    "so we need to remove this redepency by using input from only the latest path, e.g., test/x3, test_1/test/x3, the"
                    "former will be removed in the above pruning, so we need to replace x3 input as test_1/test/x3 from the current graph"
                    # change the scope prefix to be the one from the latest path
                    bfname = inp
                    if(scope_name in inp):
                      regexp = re.escape(scope_name) + "_\d+/|" + re.escape(scope_name) + "/|" + \
                                re.escape(dummy_scope_name) + "_\d+/|" + re.escape(dummy_scope_name) + "/" # pattern for "ranger_1/" or "ranger"
                      inp_names.append( target_graph_prefix + re.sub(regexp, "", inp) )
                      afname = target_graph_prefix + re.sub(regexp, "", inp)
                    else:
                      inp_names.append(target_graph_prefix + inp)
                      afname = target_graph_prefix + inp
   

            del node.input[:] # delete all the inputs
            node.input.extend(inp_names) # keep the modified input dependency 

    return mod_graph_def

  def printgraphdef(graphdef):
    for each in graphdef.node: 
        print(each.name) 

  def printgraph(sess):
      ops = [tensor for op in sess.graph.get_operations() for tensor in op.values()]
      #a = open("op.txt", "a")
      for n in ops:   
        #a.write(n.name + "\n")
        print(n.name)


  "NOTE: if you rename the name of the opeartor, you'll need to sepcify it in the following"
  "below gives default op name from TensorFlow"

  act = "Relu" 
  op_follow_act = ["MaxPool", "Reshape", "AvgPool"]
  special_op_follow_act = "concat"
  up_bound = [3.,4.,9.] # upper bound for restriction
  low_bound = [0.,0.,0.] # low bound for restriction

  PREFIX = 'ranger' # scope name in the graph
  DUMMY_PREFIX = 'dummy' # 
  graph_dup_cnt = 0 # count the number of iteration for duplication, used to track the scope prefix of the new op
  dummy_graph_dup_cnt = 0 # count the num of dummy graph duplication (for resetting the default graph to contain only the latest path)

  op_cnt = 0 # count num of op
  act_cnt = 0 # count num of act
  check_follow = False # flag for checking the following op (when the current op is ACT) 
  op_to_keep = [] # ops to keep while duplicating the graph (we remove the irrelevant ops before duplication, otherwise the graph size will explode)
  new_op_prefix = "bound_op_prefix" # prefix of the newly created ops for range restriction
  OLD_SESS = sess # keep the old session
  train_var = tf.trainable_variables() # all vars before duplication

  # get all the operators in the graph
  ops = [tensor for op in sess.graph.get_operations() for tensor in op.values()]
  graph_def = sess.graph.as_graph_def() 

   
  "iterate each op in the graph and insert bounding ops"
  for cur_op in ops:   

      if( act in cur_op.name and ("gradients" not in cur_op.name)  ): 
        # bounding
        with tf.name_scope(new_op_prefix) as scope: # the restricion ops will have the special scope prefix name
          bound_tensor = sess.graph.get_tensor_by_name( get_op_with_prefix(cur_op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)  )
          print("bounding: ", bound_tensor, up_bound[act_cnt])
          rest = tf.maximum(bound_tensor, low_bound[act_cnt] )
          rest = tf.minimum(rest, up_bound[act_cnt] )
         
        op_to_be_replaced = get_op_with_prefix(cur_op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX) 

        # delete redundant paths in graphdef and modify the input dependency to be depending on the latest path only
        truncated_graphdef = modify_graph(sess, graph_dup_cnt, PREFIX, new_op_prefix, dummy_graph_dup_cnt, DUMMY_PREFIX)
        # import the modified graghdef (inserted with bouding ops) into the current graph
        tf.import_graph_def(truncated_graphdef, name=PREFIX, input_map={op_to_be_replaced: rest}) 
        graph_dup_cnt += 1


        "reset the graph to contain only the duplicated path"
        truncated_graphdef = modify_graph(sess, graph_dup_cnt, PREFIX, new_op_prefix, dummy_graph_dup_cnt, DUMMY_PREFIX)
        tf.reset_default_graph()
        sess = tf.Session()
        sess.as_default()  
        tf.import_graph_def(truncated_graphdef, name=DUMMY_PREFIX)   
        dummy_graph_dup_cnt += 1  

      

        check_follow = True # this is a ACT, so we need to check the following op
        act_cnt= (act_cnt+1)%len(up_bound) # count the number of visited ACT (used for the case where there are two copies of ops (e.g., LeNet), one for training and one testing)

      # this will check the next operator that follows the ACT op
      elif(check_follow):
        keep_rest = False # check whether the following op needs to be bounded 

        # this is the case for Maxpool, Avgpool and Reshape
        for each in op_follow_act: 
          if(each in cur_op.name and "/shape" not in cur_op.name ): #the latter condition is for checking case like "Reshape_1/shape:0", this shouldn't be bounded
            keep_rest=True 
            low = low_bound[act_cnt-1]
            up = up_bound[act_cnt-1]
            break
        # this is the case for ConCatV2, "axis" is the parameter to the actual op concat
        if(special_op_follow_act in cur_op.name and ("axis" not in cur_op.name) and ("values" not in cur_op.name) ):  
          keep_rest=True
          low = np.minimum(low_bound[act_cnt-1], low_bound[act_cnt-2])
          up = np.maximum(up_bound[act_cnt-1], up_bound[act_cnt-2]) 

        "bound the values, using either float (default) or int"
        if(keep_rest):
          try:
            with tf.name_scope(new_op_prefix) as scope: # the restricion ops will have the special scope prefix name
              bound_tensor = sess.graph.get_tensor_by_name( get_op_with_prefix(cur_op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)  )
              print("bounding: ", bound_tensor)
              rest = tf.maximum(bound_tensor, low )
              rest = tf.minimum(rest, up )
          except:
            with tf.name_scope(new_op_prefix) as scope: # the restricion ops will have the special scope prefix name
              bound_tensor = sess.graph.get_tensor_by_name( get_op_with_prefix(cur_op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)  )
              print("bounding: ", bound_tensor)
              rest = tf.maximum(bound_tensor, int(low) )
              rest = tf.minimum(rest, int(up) )     
          #print(cur_op, act_cnt)     
          #print(rest.op.node_def,' -----')
          "replace the input to the tensor, at the palce where we place Ranger, e.g., Ranger(ReLu), then we replace Relu"
          op_to_be_replaced = get_op_with_prefix(cur_op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)

          truncated_graphdef = modify_graph(sess, graph_dup_cnt, PREFIX, new_op_prefix, dummy_graph_dup_cnt, DUMMY_PREFIX)
          tf.import_graph_def(truncated_graphdef, name=PREFIX, input_map={op_to_be_replaced: rest}) 
          graph_dup_cnt += 1


          "reset the graph to contain only the duplicated path"
          truncated_graphdef = modify_graph(sess, graph_dup_cnt, PREFIX, new_op_prefix, dummy_graph_dup_cnt, DUMMY_PREFIX)
          tf.reset_default_graph()
          sess = tf.Session()
          sess.as_default()  
          tf.import_graph_def(truncated_graphdef, name=DUMMY_PREFIX)   
          dummy_graph_dup_cnt += 1  



        # check the ops, but not to bound the ops
        else:
          check_follow=False # the default setting is not to check the next op

          # the following ops of the listed operaions will be kept tracking, 
          # becuase the listed ops do not perform actual computation, so the restriction bound still applies
          oblivious_ops = ["Const", "truncated_normal", "Variable", "weights", "biases", "dropout"] 
          if( ("Reshape" in cur_op.name and "/shape" in cur_op.name) or \
              ("concat" in cur_op.name and ("axis" in cur_op.name or "values" in cur_op.name) )
             ):
            check_follow = True # we need to check the following op of Reshape/shape:0, concat/axis (these are not the actual reshape/concat ops)
          else:
            for ea in oblivious_ops: # we need to check the op follows the listed ops
              if(ea in cur_op.name): 
                check_follow = True  


      op_cnt+=1

  # we need to call modify_graph to modify the input dependency for finalization
  truncated_graphdef = modify_graph(sess, graph_dup_cnt, PREFIX, new_op_prefix, dummy_graph_dup_cnt, DUMMY_PREFIX)
  tf.import_graph_def(truncated_graphdef, name=PREFIX)    
  graph_dup_cnt += 1  
   
  "reset the graph to contain only the duplicated path"
  truncated_graphdef = modify_graph(sess, graph_dup_cnt, PREFIX, new_op_prefix, dummy_graph_dup_cnt, DUMMY_PREFIX)
  tf.reset_default_graph()
  sess = tf.Session()
  sess.as_default()
  tf.import_graph_def(truncated_graphdef, name=DUMMY_PREFIX)   
  dummy_graph_dup_cnt+=1

  "restore the vars from the old sess to the new sess"
  restore_trainable_var(sess, PREFIX, graph_dup_cnt, train_var, DUMMY_PREFIX, dummy_graph_dup_cnt, OLD_SESS)



  print("Finish graph modification!")
  print('')

  "============================================================================================================" 
  "============================================================================================================"

  global eval_prediction
  OP_FOR_EVAL = eval_prediction # op to be eval
  new_op_for_eval_name = get_op_with_prefix(OP_FOR_EVAL.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX) 
  print(new_op_for_eval_name, 'op to be eval')
  new_op_for_eval = sess.graph.get_tensor_by_name(new_op_for_eval_name + ":0") 
 
  
  eval_prediction = new_op_for_eval # need the evaluate the op from the NEW graph
  # replace the input op from the NEW graph 
  eval_data = sess.graph.get_tensor_by_name( get_op_with_prefix(eval_data.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX) + ":0")


  "Insert this AFTER inserting range check into the original graph (Tested under TF 1.14)"
  "this will get the FLOPs of the target op in the original and new graph"
  "NOTE: you need to set the shape of the input op (placeholder) to be constant value, cannot set \"None\" otherwise the FLOPs cannot be obtained due to incomplete shape "

  "Measure the FLOPs before and after inserting range restriction"
  from tensorflow.python.framework import graph_util
  "This is to get the FLOPS in the model"
  def load_pb(pb):
      with tf.gfile.GFile(pb, "rb") as f:
          graph_def = tf.GraphDef()
          graph_def.ParseFromString(f.read())
      with tf.Graph().as_default() as graph:
          tf.import_graph_def(graph_def, name='')
          return graph

  # get the FLOPs of the new graph
  g = tf.get_default_graph()  #sess.graph()
  output_graph_def = graph_util.convert_variables_to_constants(sess, g.as_graph_def(), [new_op_for_eval_name])

  with tf.gfile.GFile('graph.pb', "wb") as f:
      f.write(output_graph_def.SerializeToString())
  # *****************************


  # ***** (3) Load frozen graph *****
  g2 = load_pb('./graph.pb')
  with g2.as_default():
      flops = tf.profiler.profile(g2, options = tf.profiler.ProfileOptionBuilder.float_operation())
      print('FLOP after freezing', flops.total_float_ops)


  # get the FLOPs in the old graph
  gradef = OLD_SESS.graph.as_graph_def()  #sess.graph()
  output_graph_def = tf.graph_util.convert_variables_to_constants(OLD_SESS, gradef, [OP_FOR_EVAL.op.name])

  with tf.gfile.GFile('org-graph.pb', "wb") as f:
      f.write(output_graph_def.SerializeToString())
  # *****************************

  # ***** (3) Load frozen graph *****
  g2 = load_pb('./org-graph.pb')
  with g2.as_default():
      flops = tf.profiler.profile(g2, options = tf.profiler.ProfileOptionBuilder.float_operation())
      print('FLOP after freezing', flops.total_float_ops)


    

 


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--use_fp16',
      default=False,
      help='Use half floats instead of full floats if True.',
      action='store_true')
  parser.add_argument(
      '--self_test',
      default=False,
      action='store_true',
      help='True if running a self test.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
