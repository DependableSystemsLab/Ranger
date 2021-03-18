## Cifar-10 implementation
## Code based on https://github.com/exelban/tensorflow-cifar-10 
## LRN layers have been removed
##

import numpy as np
import tensorflow as tf
from time import time
import math
 

import pickle
import numpy as np
import os
from six.moves.urllib.request import urlretrieve
import tarfile
import zipfile
import sys
import datetime
#import TensorFI4 as ti

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

global isTrain 
global isTest



isTest = True
isTrain = not isTest
 
_SAVE_PATH = "./save/"
if(not os.path.isdir(_SAVE_PATH)):
    os.mkdir(_SAVE_PATH)

_BATCH_SIZE = 128
_CLASS_SIZE = 10 
_EPOCH = 128


def get_data_set(name="train"):
    x = None
    y = None

    maybe_download_and_extract()

    folder_name = "cifar_10"

    f = open('./data_set/'+folder_name+'/batches.meta', 'rb')
    f.close()

    if name is "train":
        for i in range(5):
            f = open('./data_set/'+folder_name+'/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f)
            f.close()

            _X = datadict["data"]
            _Y = datadict['labels']

            _X = np.array(_X, dtype=float) / 255.0
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 2, 3, 1])
            _X = _X.reshape(-1, 32*32*3)

            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)

    elif name is "test":
        f = open('./data_set/'+folder_name+'/test_batch', 'rb')
        datadict = pickle.load(f)
        f.close()

        x = datadict["data"]
        y = np.array(datadict['labels'])

        x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        x = x.reshape(-1, 32*32*3)

    return x, dense_to_one_hot(y)

def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot

def _print_download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()

def maybe_download_and_extract():
    main_directory = "./data_set/"
    cifar_10_directory = main_directory+"cifar_10/"
    if not os.path.exists(main_directory):
        os.makedirs(main_directory)

        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_cifar_10 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
        print("Done.")

        os.rename(main_directory+"./cifar-10-batches-py", cifar_10_directory)
        os.remove(zip_cifar_10)

def model(): 
    _IMAGE_SIZE = 32
    _IMAGE_CHANNELS = 3
    _NUM_CLASSES = 10

    with tf.name_scope('main_params'):
        x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
        x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

        prob = tf.placeholder(tf.float32, name='prob') 
        
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    with tf.variable_scope('conv1') as scope:
        conv1_1 = tf.layers.conv2d(
            inputs=x_image,
            filters=32,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )


        pool1_1 = tf.layers.max_pooling2d(conv1_1, pool_size=[3, 3], strides=2, padding='SAME') 
        conv1_2 = tf.layers.conv2d(
            inputs=pool1_1,
            filters=64,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )



        pool1_2 = tf.layers.max_pooling2d(conv1_2, pool_size=[2, 2], strides=2, padding='SAME') 


    pool1_2 = tf.nn.dropout(pool1_2, keep_prob=prob)

    with tf.variable_scope('conv2') as scope:
        conv2_1 = tf.layers.conv2d(
            inputs=pool1_2,
            filters=128,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        ) 



        conv2_2 = tf.layers.conv2d(
            inputs=conv2_1,
            filters=128,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )




        conv2_3 = tf.layers.conv2d(
            inputs=conv2_2,
            filters=128,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )



        pool = tf.layers.max_pooling2d(conv2_3, pool_size=[2,2], strides=2, padding='SAME') 



    pool = tf.nn.dropout(pool, keep_prob=prob)

    with tf.variable_scope('fully_connected') as scope:

        pool_shape = pool.get_shape().as_list()
        flat = tf.reshape( pool, [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]] )


        fc1 = tf.layers.dense(inputs=flat, units=1500, activation=tf.nn.relu) 




        fc1 = tf.nn.dropout(fc1, keep_prob=prob) 

    softmax = tf.layers.dense(inputs = fc1, units=_NUM_CLASSES)

    y_pred_cls = tf.argmax(softmax, axis=1)

    return x, y, softmax, y_pred_cls, global_step, learning_rate, prob

def lr(epoch):
    learning_rate = 1e-3
    if epoch > 80:
        learning_rate *= 0.5e-3
    elif epoch > 60:
        learning_rate *= 1e-3
    elif epoch > 40:
        learning_rate *= 1e-2
    elif epoch > 20:
        learning_rate *= 1e-1
    return learning_rate

def train(epoch):
    global epoch_start
    epoch_start = time()
    batch_size = int(math.ceil(len(train_x) / _BATCH_SIZE))
    i_global = 0


    recLearning = open("batch_loss_acc.csv", "a")

    for s in range(batch_size):
        batch_xs = train_x[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]
        batch_ys = train_y[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]
  

        start_time = time()
        i_global, _, batch_loss, batch_acc = sess.run(
            [global_step, optimizer, loss, accuracy],
            feed_dict={x: batch_xs, y: batch_ys, learning_rate: lr(epoch), prob: 0.5} )
        duration = time() - start_time

        if s % 10 == 0:
            percentage = int(round((s/batch_size)*100))

            bar_len = 29
            filled_len = int((bar_len*int(percentage))/100)
            bar = '=' * filled_len + '>' + '-' * (bar_len - filled_len)

            msg = "Global step: {:>5} - [{}] {:>3}% - acc: {:.4f} - loss: {:.4f} - {:.1f} sample/sec"
            print(msg.format(i_global, bar, percentage, batch_acc, batch_loss, _BATCH_SIZE / duration))
    
            recLearning.write(`i_global` + "," + `batch_acc` + ","+ `batch_loss` + "\n")

    test_and_save(i_global, epoch)

def test_and_save(_global_step, epoch):
    global global_accuracy
    global epoch_start

    learning = open("learning_progress.csv", "a")

    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    while i < len(test_x):
        j = min(i + _BATCH_SIZE, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j] = sess.run(
            y_pred_cls,
            feed_dict={x: batch_xs, y: batch_ys, learning_rate: lr(epoch), prob: 1.}
        )
        i = j

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean()*100
    correct_numbers = correct.sum()

    hours, rem = divmod(time() - epoch_start, 3600)
    minutes, seconds = divmod(rem, 60)
    mes = "\nEpoch {} - accuracy: {:.2f}% ({}/{}) - time: {:0>2}:{:0>2}:{:05.2f}"
    print(mes.format((epoch+1), acc, correct_numbers, len(test_x), int(hours), int(minutes), seconds))


    learning.write(`epoch` + "," + `_global_step` + "," + `acc` + "\n")

    if global_accuracy != 0 and global_accuracy < acc:

        summary = tf.Summary(value=[
            tf.Summary.Value(tag="Accuracy/test", simple_value=acc),
        ])
        train_writer.add_summary(summary, _global_step)

        saver.save(sess, save_path=_SAVE_PATH)

        mes = "This epoch receive better accuracy: {:.2f} > {:.2f}. Saving session..."
        print(mes.format(acc, global_accuracy))
        global_accuracy = acc

    elif global_accuracy == 0:
        global_accuracy = acc

    print("###########################################################################################################")



if(isTrain): 

    train_x, train_y = get_data_set("train")
    test_x, test_y = get_data_set("test")
    x, y, output, y_pred_cls, global_step, learning_rate, prob = model()


    global_accuracy = 0.
    epoch_start = 0
    
     
    # LOSS AND OPTIMIZER
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=0.9,
                                       beta2=0.999,
                                       epsilon=1e-08).minimize(loss, global_step=global_step)

    # PREDICTION AND ACCURACY CALCULATION
    correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # SAVER
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)
    init = tf.initialize_all_variables()
    sess.run(init)


    try:
        print("\nTrying to restore last checkpoint ...")
#        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
        saver.restore(sess, save_path=_SAVE_PATH)
    except ValueError:
        print("\nFailed to restore checkpoint. Initializing variables instead.")
        sess.run(tf.global_variables_initializer())



elif(isTest):
    test_x, test_y = get_data_set("test")
    x, y, output, y_pred_cls, global_step, learning_rate, prob = model()


    saver = tf.train.Saver()
    sess = tf.Session()  
    sess.as_default()  

    try:
        print("\nTrying to restore last checkpoint ...")
#        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
        saver.restore(sess, save_path= _SAVE_PATH)
    except ValueError:
        print("\nFailed to restore checkpoint. Initializing variables instead.")
        sess.run(tf.global_variables_initializer())
 






global pred 
pred = output



 
global isTrain 
global isTest
global pred
if(isTrain):
    train_start = time()

    for i in range(_EPOCH):
        print("\nEpoch: {}/{}\n".format((i+1), _EPOCH))
        train(i)

    hours, rem = divmod(time() - train_start, 3600)
    minutes, seconds = divmod(rem, 60)
    mes = "Best accuracy pre session: {:.2f}, time: {:0>2}:{:0>2}:{:05.2f}"
    print(mes.format(global_accuracy, int(hours), int(minutes), seconds))


if(isTest):






  
    "============================================================================================================"
    "=================  Begin to insert restriction on selective layers  ========================================"
    "============================================================================================================"
    "NOTE: You should not use nested graph as the default graph will be reset with the new duplicated graph"
    "INPUT: 1) the restriction bounds; 2) name of the ACT operator and their corresponding follow ops (default name is provided)"
    "============================================================================================================"


    # get all the operators in the graph
    ops = [tensor for op in sess.graph.get_operations() for tensor in op.values()]
    # get the graphdef to manipulate the nodes in the graph
    graph_def = sess.graph.as_graph_def()

    def get_op_dependency(op):
      "get all the node that precedes the target op"
      "you can use this function to check if the restriction ops have been inserted"
      cur_op = []
      a = open('op-dep.txt', "w")
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
              target_graph_prefix = scope_name  + "/" + target_graph_prefix  # e.g., ranger/dummy/ranger/relu 
              if(dummy_graph_dup_cnt>0):
                target_graph_prefix = dummy_scope_name + "/" + target_graph_prefix # e.g., dummy/ranger/dummy/ranger/relu:0
                dummy_graph_dup_cnt-=1

      return target_graph_prefix

    def restore_trainable_var(sess, scope_name, dup_cnt, train_var, dummy_scope_name, dummy_graph_dup_cnt, OLD_SESS):
      "map back the variable values from the original graph to the new graph" 
      target_graph_prefix = get_target_scope_prefix(scope_name,dup_cnt,dummy_scope_name,dummy_graph_dup_cnt)
      tmp = []
      for each in train_var: 
        tmp.append( tf.assign(sess.graph.get_tensor_by_name(target_graph_prefix + each.name), OLD_SESS.run( OLD_SESS.graph.get_tensor_by_name( each.name ) )) )
      sess.run(tmp) 

    def get_op_with_prefix(op_name, dup_cnt, scope_name, dummy_graph_dup_cnt, dummy_scope_name):
      "Need to call this function to return the name of the ops under the NEW graph (with scope prefix)" 
      op_name = get_target_scope_prefix(scope_name,dup_cnt,dummy_scope_name,dummy_graph_dup_cnt) + op_name
      return op_name

    import re
    def modify_graph(sess, dup_cnt, scope_name, prefix_of_bound_op, dummy_graph_dup_cnt, dummy_scope_name): 
      "Modify the graph def to: 1) remove the nodes from older paths (we only need to keep the latest path)"
      " and 2) modify the input dependency to only associate with the latest path" 

      graph_def = sess.graph.as_graph_def() 
      target_graph_prefix = get_target_scope_prefix(scope_name,dup_cnt,dummy_scope_name,dummy_graph_dup_cnt)
      #print('target prefix ==> ', target_graph_prefix, dup_cnt)

      # Delete nodes from the redundant paths, we only retain the ones from the most recent path, otherwise the size of graph will explode
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

      # append the remaining nodes into the graphdef
      mod_graph_def = tf.GraphDef()
      mod_graph_def.node.extend(nodes) 
       

      #For the newly created op, we need to modify the input dependency so that it only relies on the latest graph
      #because we've only kpet the latest graph in the modified graphdef
      # Delete references to deleted nodes
      for node in mod_graph_def.node: 
          inp_names = []
          if(prefix_of_bound_op in node.name): # only for the restriction op
              for inp in node.input:
                  if prefix_of_bound_op in inp or target_graph_prefix in inp:
                      inp_names.append(inp)
                  else:  
                      # here because we copy the graghdef from the previous graph, it has dependency to the PREVIOUS graph"
                      # so we need to remove this redepency by using input from only the latest path,
                      # we do so by changing the scope prefix to be target_graph_prefix
                      bfname = inp
                      if(scope_name in inp):
                        # regular expression to match any pattern like "ranger_1, ranger, dummy_1, dummy"
                        regexp = re.escape(scope_name) + "_\d+/|" + re.escape(scope_name) + "/|" + \
                                  re.escape(dummy_scope_name) + "_\d+/|" + re.escape(dummy_scope_name) + "/" # pattern for "ranger_1/" or "ranger"
                        # we first remove all those prefix, and then append the target_graph_prefix to it
                        inp_names.append( target_graph_prefix + re.sub(regexp, "", inp) )
                        afname = target_graph_prefix + re.sub(regexp, "", inp)
                      else:
                        inp_names.append(target_graph_prefix + inp)
                        afname = target_graph_prefix + inp
     
              del node.input[:] # delete all the inputs (in the original dependency)
              node.input.extend(inp_names) # keep the modified input dependency 

      return mod_graph_def

    def printgraphdef(graphdef):
      "print each node name from the graphdef"
      for each in graphdef.node: 
          print(each.name) 

    def printgraph(sess):
        "print each node name from the sess graph"
        ops = [tensor for op in sess.graph.get_operations() for tensor in op.values()]
        #a = open("op.txt", "a")
        for n in ops:   
          #a.write(n.name + "\n")
          print(n.name)


    "Below is the input needed"
    #NOTE: if you rename the name of the opeartor, you'll need to sepcify it in the following"
    # below gives default op name from TensorFlow
    act = "Relu" 
    op_follow_act = ["MaxPool", "Reshape", "AvgPool"]
    special_op_follow_act = "concat"
    up_bound = [1, 2., 2., 3., 4., 9.] # upper bound for restriction
    low_bound = [0.,0.,0.,0.,0.,0.] # low bound for restriction
    "End of input needed"


    PREFIX = 'ranger' # scope name for the duplication in which we insert the restricion op
    DUMMY_PREFIX = 'dummy' # scope name for the dummy duplication, which is for resetting the default graph (no new op is inserted)
    graph_dup_cnt = 0 # count the number of duplication, used to track the scope prefix of the new op
    dummy_graph_dup_cnt = 0 # count the num of dummy graph duplication, used to track the scope prefix of the op

    op_cnt = 0 # num of op
    act_cnt = 0 # num of act
    check_follow = False # flag for checking the following op (e.g., when the current op is ACT, we perform checking on the following op)  
    new_op_prefix = "bound_op_prefix" # prefix of the newly created restricion ops (tf.maximum and tf.minimum)
    OLD_SESS = sess # keep the old session, because the default graph will be reset with the duplicated one
    train_var = tf.trainable_variables() # all vars before duplication

    # get all the operators in the graph
    ops = [tensor for op in sess.graph.get_operations() for tensor in op.values()]
    graph_def = sess.graph.as_graph_def() 

     
    "iterate each op in the graph and insert bounding ops"
    for cur_op in ops:   

        "This is a ACT op"
        if( act in cur_op.name and ("gradients" not in cur_op.name)  ): 
          # bounding
          with tf.name_scope(new_op_prefix) as scope: # the restricion ops will have the special scope prefix name
            bound_tensor = sess.graph.get_tensor_by_name( get_op_with_prefix(cur_op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)  )
            print("bounding: ", bound_tensor, up_bound[act_cnt])
            rest = tf.maximum(bound_tensor, low_bound[act_cnt] ) # low bound restriction
            rest = tf.minimum(rest, up_bound[act_cnt] ) # up bound restriction
           
          op_to_be_replaced = get_op_with_prefix(cur_op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX) 

          "insert the restriction ops into the graph"
          # delete redundant paths in graphdef and modify the input dependency to be depending on the latest path only
          truncated_graphdef = modify_graph(sess, graph_dup_cnt, PREFIX, new_op_prefix, dummy_graph_dup_cnt, DUMMY_PREFIX)
          # import the modified graghdef (inserted with bouding ops) into the current graph
          tf.import_graph_def(truncated_graphdef, name=PREFIX, input_map={op_to_be_replaced: rest}) 
          graph_dup_cnt += 1


          "reset the graph to contain only the duplicated path (otherwise the graph size will explode (TensorFlow has 2GB limit for graphdef)"
          truncated_graphdef = modify_graph(sess, graph_dup_cnt, PREFIX, new_op_prefix, dummy_graph_dup_cnt, DUMMY_PREFIX)
          tf.reset_default_graph()
          sess = tf.Session()
          sess.as_default()  
          tf.import_graph_def(truncated_graphdef, name=DUMMY_PREFIX)   
          dummy_graph_dup_cnt += 1  

        
          check_follow = True # this is a ACT, so we need to check the following op
          act_cnt= (act_cnt+1)%len(up_bound) # count the number of visited ACT (used for the case where there are two copies of ops, one for training and one testing)

        
        elif(check_follow):
          "Check the op that follows ACT"
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
            # for the case such as: Reshape/shape:0 precedes Reshape/, the latter is the actual op to be bounded
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







    "pred is the name of the operator to be evaluated, we find its replica under the new graph"
    OP_FOR_EVAL = pred 
    new_op_for_eval_name = get_op_with_prefix(OP_FOR_EVAL.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX) 
    print(new_op_for_eval_name, 'op to be eval')
    new_op_for_eval = sess.graph.get_tensor_by_name(new_op_for_eval_name + ":0") 
   

    # you can call this function to check the depenency of the final operator
    # you should see the bouding ops are inserted into the dependency
    # NOTE: the printing might contain duplicated output
    #get_op_dependency(new_op_for_eval.op)

    "you can specify input here"
    tx = test_x[:3,:]
    ty = test_y[:3,:]

    # evaluation on the old path
    preds = OLD_SESS.run(OP_FOR_EVAL, feed_dict={x: tx, y: ty, prob: 1.})
    print(preds)
    print('')

    # evaluation on the new path
    new_x = sess.graph.get_tensor_by_name( get_op_with_prefix(x.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)+":0")
    new_y = sess.graph.get_tensor_by_name( get_op_with_prefix(y.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)+":0")
    new_prob = sess.graph.get_tensor_by_name( get_op_with_prefix(prob.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)+":0")
    preds = sess.run(new_op_for_eval, feed_dict={new_x: tx, new_y: ty, new_prob: 1.})
    print(preds)




 
         

 


sess.close()

