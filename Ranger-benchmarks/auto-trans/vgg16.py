########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################
from numpy import *
import os
#from pylab import *

import numpy as np
from scipy.misc import imread, imresize
from caffe_classes import class_names
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import scipy 
import datetime 
import tensorflow as tf
import datetime
class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc3l)
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print i, k, np.shape(weights[k])
            sess.run(self.parameters[i].assign(weights[k]))

#if __name__ == '__main__':

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session()

#sess = tf.Session()
imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
vgg = vgg16(imgs, 'vgg16_weights.npz', sess)



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
  cur_op.append(op)
  next_op=[]
  while(not (next_op==[] and cur_op==[])):
      next_op = []
      for each in cur_op:
          printline = False
          for inp in each.inputs:
              printline = True
              print(inp)
              next_op.append(inp.op) 
          if(printline): print('')
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
"MaxPool has been renamed as pool"
op_follow_act = ["pool", "Reshape", "AvgPool"]
special_op_follow_act = "concat"
up_bound = map(float, [956, 5118, 9870, 16892, 27154, 20696, 25586, 19959, 9975, 5791, 4118, 2379, 869, 145, 42]) # upper bound for restriction
low_bound = map(float, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # low bound for restriction


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
      act_cnt= (act_cnt+1)%len(up_bound) # count the number of visited ACT (used for the case where there are two copies of ops, one for training and one testing)

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







"This is the name of the operator to be evaluated, we will find the corresponding one under the Ranger's scope"
OP_FOR_EVAL = vgg.probs
#        OP_FOR_EVAL = eval_prediction # op to be eval
new_op_for_eval_name = get_op_with_prefix(OP_FOR_EVAL.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX) 
print(new_op_for_eval_name, 'op to be eval')
new_op_for_eval = sess.graph.get_tensor_by_name(new_op_for_eval_name + ":0") 

# you can call this function to check the depenency of the final operator
# you should see the bouding ops are inserted into the dependency
# NOTE: the printing might contain duplicated output 
#get_op_dependency(new_op_for_eval.op)


path = 'fiImg'
files = []
for r, d, f in os.walk(path):
    for file in f:
        if( ".JPEG" in file ):
            files.append(os.path.join(r, file))

img = imread(files[0], mode='RGB') 
img = scipy.misc.imresize(img, [224,224])  
 


#sess.run(vgg.probs, feed_dict={vgg.imgs: [img]})[0] 
#preds = (np.argsort(prob)[::-1])[0:5]


# evaluation on the old path 
preds = OLD_SESS.run(OP_FOR_EVAL, feed_dict={vgg.imgs: [img]})[0]
print( (np.argsort(preds)[::-1])[0:10] )
print('')

# evaluation on the new path
new_x = sess.graph.get_tensor_by_name( get_op_with_prefix(vgg.imgs.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)+":0") 
preds = sess.run(new_op_for_eval, feed_dict={new_x: [img]})[0]
print( (np.argsort(preds)[::-1])[0:10] )















  





