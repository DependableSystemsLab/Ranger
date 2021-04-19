### VGG11 implementation from https://github.com/mohamedameen93/German-Traffic-Sign-Classification-Using-TensorFlow
###


from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import random
import numpy as np
import cv2 
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import pickle
import datetime
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# training hyperparameters
BATCHSIZE = 256
EPOCHS = 256

isTrain = False


SAVE_PATH  = './save/'
if(not os.path.isdir(SAVE_PATH)):
    os.mkdir(SAVE_PATH)
 

training_file = "./new-dataset/train.p" 
testing_file = "./new-dataset/test.p"


with open(training_file, mode='rb') as f:
    train = pickle.load(f) 
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

    
# Load pickled data
#train, test = load_traffic_sign_data('traffic-signs-data/train.p', 'traffic-signs-data/test.p')
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

# Number of examples
n_train, n_test = X_train.shape[0], X_test.shape[0]

# What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# How many classes?
n_classes = np.unique(y_train).shape[0]

print("Number of training examples =", n_train)
print("Number of testing examples  =", n_test)
print("Image data shape  =", image_shape)
print("Number of classes =", n_classes)

'''
# show a random sample from each class of the traffic sign dataset
rows, cols = 4, 12
fig, ax_array = plt.subplots(rows, cols)
plt.suptitle('RANDOM SAMPLES FROM TRAINING SET (one for each class)')
for class_idx, ax in enumerate(ax_array.ravel()):
    if class_idx < n_classes:
        # show a random image of the current class
        cur_X = X_train[y_train == class_idx]
        cur_img = cur_X[np.random.randint(len(cur_X))]
        ax.imshow(cur_img)
        ax.set_title('{:02d}'.format(class_idx))
    else:
        ax.axis('off')
# hide both x and y ticks
plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
plt.draw()



train_distribution, test_distribution = np.zeros(n_classes), np.zeros(n_classes)
for c in range(n_classes):
    train_distribution[c] = np.sum(y_train == c) / n_train
    test_distribution[c] = np.sum(y_test == c) / n_test
fig, ax = plt.subplots()
col_width = 0.5
bar_train = ax.bar(np.arange(n_classes), train_distribution, width=col_width, color='r')
bar_test = ax.bar(np.arange(n_classes)+col_width, test_distribution, width=col_width, color='b')
ax.set_ylabel('PERCENTAGE OF PRESENCE')
ax.set_xlabel('CLASS LABEL')
ax.set_title('Classes distribution in traffic-sign dataset')
ax.set_xticks(np.arange(0, n_classes, 5)+col_width)
ax.set_xticklabels(['{:02d}'.format(c) for c in range(0, n_classes, 5)])
ax.legend((bar_train[0], bar_test[0]), ('train set', 'test set'))
plt.show()
'''

def preprocess_features(X, equalize_hist=True):
    '''
    # convert from RGB to YUV
    X = np.array([np.expand_dims(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV)[:, :, 0], 2) for rgb_img in X])

    # adjust image contrast
    if equalize_hist:
        X = np.array([np.expand_dims(cv2.equalizeHist(np.uint8(img)), 2) for img in X])
    '''
    X = np.float32(X)

    # standardize features
    X -= np.mean(X, axis=0)
    X /= (np.std(X, axis=0) + np.finfo('float32').eps)

    return X

X_train_norm = preprocess_features(X_train)
X_test_norm = preprocess_features(X_test)


# split into train and validation
VAL_RATIO = 0.1
X_train_norm, X_val_norm, y_train, y_val = train_test_split(X_train_norm, y_train, test_size=VAL_RATIO, random_state=0)


# create the generator to perform online data augmentation
image_datagen = ImageDataGenerator(rotation_range=15.,
                                   zoom_range=0.2,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1)
'''
# take a random image from the training set
img_rgb = X_train[0]

# plot the original image
plt.figure(figsize=(1,1))
plt.imshow(img_rgb)
plt.title('Example of RGB image (class = {})'.format(y_train[0]))
plt.show()

# plot some randomly augmented images
rows, cols = 4, 10
fig, ax_array = plt.subplots(rows, cols)
for ax in ax_array.ravel():
    augmented_img, _ = image_datagen.flow(np.expand_dims(img_rgb, 0), y_train[0:1]).next()
    ax.imshow(np.uint8(np.squeeze(augmented_img)))
plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
plt.suptitle('Random examples of data augmentation (starting from the previous image)')
plt.show()
'''


def weight_variable(shape, mu=0, sigma=0.1):
    initialization = tf.truncated_normal(shape=shape, mean=mu, stddev=sigma)
    return tf.Variable(initialization)


def bias_variable(shape, start_val=0.1):
    initialization = tf.constant(start_val, shape=shape)
    return tf.Variable(initialization)


def conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'):
    return tf.nn.conv2d(input=x, filter=W, strides=strides, padding=padding)


def max_pool_2x2(x):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# network architecture definition
def my_net(x, n_classes):

    c1_out = 32
    conv1_1W = weight_variable(shape=(3, 3, 3, c1_out))
    conv1_1b = bias_variable(shape=(c1_out,))
    conv1_1 = tf.nn.relu(conv2d(x, conv1_1W) + conv1_1b)

    conv1_2W = weight_variable(shape=(3, 3, c1_out, c1_out))
    conv1_2b = bias_variable(shape=(c1_out,))
    conv1_2 = tf.nn.relu(conv2d(conv1_1, conv1_2W) + conv1_2b)

    pool1 = max_pool_2x2(conv1_2)

    drop1 = tf.nn.dropout(pool1, keep_prob=keep_prob)

    c2_out = 64
    conv2_1W = weight_variable(shape=(3, 3, c1_out, c2_out))
    conv2_1b = bias_variable(shape=(c2_out,))
    conv2_1 = tf.nn.relu(conv2d(pool1, conv2_1W) + conv2_1b)

    conv2_2W = weight_variable(shape=(3, 3, c2_out, c2_out))
    conv2_2b = bias_variable(shape=(c2_out,))
    conv2_2 = tf.nn.relu(conv2d(conv2_1, conv2_2W) + conv2_2b)

    pool2 = max_pool_2x2(conv2_2)

    drop2 = tf.nn.dropout(pool2, keep_prob=keep_prob)
 
    c3_out = 128
    conv3_1W = weight_variable(shape=(3, 3, c2_out, c3_out))
    conv3_1b = bias_variable(shape=(c3_out,))
    conv3_1 = tf.nn.relu(conv2d(pool2, conv3_1W) + conv3_1b)

    conv3_2W = weight_variable(shape=(3, 3, c3_out, c3_out))
    conv3_2b = bias_variable(shape=(c3_out,))
    conv3_2 = tf.nn.relu(conv2d(conv3_1, conv3_2W) + conv3_2b)

    pool3 = max_pool_2x2(conv3_2)


    c4_out = 128
    conv4_1W = weight_variable(shape=(3, 3, c3_out, c4_out))
    conv4_1b = bias_variable(shape=(c4_out,))
    conv4_1 = tf.nn.relu(conv2d(pool3, conv4_1W) + conv4_1b)

    conv4_2W = weight_variable(shape=(3, 3, c4_out, c4_out))
    conv4_2b = bias_variable(shape=(c4_out,))
    conv4_2 = tf.nn.relu(conv2d(conv4_1, conv4_2W) + conv4_2b)

    pool4 = max_pool_2x2(conv4_2)

    shape = int(np.prod(pool4.get_shape()[1:]))
    pool4 = tf.reshape(pool4, [-1, shape])
    fc1_out = 128
    fc1_W = weight_variable(shape=(pool4._shape[1].value, fc1_out))
    fc1_b = bias_variable(shape=(fc1_out,))
    fc1 = tf.matmul(pool4, fc1_W) + fc1_b 
    
    fc1 = tf.nn.relu(fc1)
    
    fc2_out = 128
    fc2_W = weight_variable(shape=(fc1._shape[1].value, fc2_out))
    fc2_b = bias_variable(shape=(fc2_out,))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b 

    fc2 = tf.nn.relu(fc2)
    drop_fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)

    fc3_out = n_classes
    fc3_W = weight_variable(shape=(fc2._shape[1].value, fc3_out))
    fc3_b = bias_variable(shape=(fc3_out,))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


# placeholders
x = tf.placeholder(dtype=tf.float32, shape=(1, 32, 32, 3))
y = tf.placeholder(dtype=tf.int32, shape=None)
keep_prob = tf.placeholder(tf.float32)


# training pipeline
lr = 0.001
logits = my_net(x, n_classes=n_classes)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss_function = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_step = optimizer.minimize(loss=loss_function)


# metrics and functions for model evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.cast(y, tf.int64))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# create a checkpointer to log the weights during training
checkpointer = tf.train.Saver()


global acy 
# start training

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

sess=tf.Session()
sess.as_default()

acy = 0
if(isTrain):
    try:
        checkpointer.restore(sess, SAVE_PATH)
        print("restore session successfully")
    except:
        sess.run(tf.global_variables_initializer())
        print("failed to restore session, initialize instead")

    learning = open("learning_progress.csv", "a")

    for epoch in range(EPOCHS):

        print("EPOCH {} ...".format(epoch + 1))

        batch_counter = 0
        for batch_x, batch_y in image_datagen.flow(X_train_norm, y_train, batch_size=BATCHSIZE):

            batch_counter += 1
            _,los = sess.run([train_step,loss_function], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            print(batch_counter, los)
            if batch_counter >= len(y_train) / BATCHSIZE:
                break

            # evaluate the validation accuracy per 20 steps.
            if(batch_counter % 20 == 0): 
                val_accuracy = evaluate(X_val_norm, y_val) 
                print('Val Accuracy = {:.3f}'.format(val_accuracy))

                learning.write(`epoch` + "," + `batch_counter` + "," + `val_accuracy` + "\n" )

                if(val_accuracy > acy): 
                    print("save checkpoints")
                    checkpointer.save(sess, save_path=SAVE_PATH)
                    acy = val_accuracy
else:
    try:
        # restore saved session with highest validation accuracy
        checkpointer.restore(sess, SAVE_PATH)
    except:
        sess.run(tf.global_variables_initializer())


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
    op_follow_act = ["MaxPool", "Reshape", "AvgPool"]
    special_op_follow_act = "concat"
    up_bound = [5., 10., 17., 27., 44., 108., 147., 292., 432., 200.] # upper bound for restriction
    low_bound = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.] # low bound for restriction

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
    OP_FOR_EVAL = logits 
    new_op_for_eval_name = get_op_with_prefix(OP_FOR_EVAL.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX) 
    print(new_op_for_eval_name, 'op to be eval')
    new_op_for_eval = sess.graph.get_tensor_by_name(new_op_for_eval_name + ":0") 


    # you can call this function to check the depenency of the final operator
    # you should see the bouding ops are inserted into the dependency
    # NOTE: the printing might contain duplicated output 
    #get_op_dependency(new_op_for_eval.op)

    X_data = X_test_norm[0, :, :, :]
    y_data = y_test[0]
    X_data = X_data.reshape(1,32,32,3)
    y_data = y_data.reshape(1,1)

    # evaluation on the old path 
    preds = OLD_SESS.run(OP_FOR_EVAL, feed_dict={x: X_data, y: y_data, keep_prob: 1.0})
    print(preds)
    print('')

    # evaluation on the new path
    new_x = sess.graph.get_tensor_by_name( get_op_with_prefix(x.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)+":0")
    new_y = sess.graph.get_tensor_by_name( get_op_with_prefix(y.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)+":0")
    new_prob = sess.graph.get_tensor_by_name( get_op_with_prefix(keep_prob.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)+":0")
    preds = sess.run(new_op_for_eval, feed_dict={new_x: X_data, new_y: X_data, new_prob: 1.})
    print(preds)


    
    "Insert this AFTER inserting range check into the original graph (Tested under TF 1.14)"
    "this will get the FLOPs of the target op in the original and new graph"
    "NOTE: you need to set the shape of the input op (placeholder) to be constant value (e.g., 1), it cannot be set as None otherwise the FLOPs cannot be obtained due to incomplete shape "

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






