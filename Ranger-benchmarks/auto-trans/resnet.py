#!/usr/bin/env python

import os
from datetime import datetime
import time
import tensorflow as tf
import numpy as np
import sys
import select
import cPickle as pickle
from IPython import embed
from sklearn.utils.extmath import softmax
import imagenet_input as data_input
import resnet
import os 


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Dataset Configuration
tf.app.flags.DEFINE_string('test_dataset', 'scripts/val.txt', """Path to the test dataset list file""")
tf.app.flags.DEFINE_string('test_image_root', '/data1/common_datasets/imagenet_resized/ILSVRC2012_val/', """Path to the root of ILSVRC2012 test images""")
tf.app.flags.DEFINE_string('mean_path', './ResNet_mean_rgb.pkl', """Path to the imagenet mean""")
tf.app.flags.DEFINE_integer('num_classes', 1000, """Number of classes in the dataset.""")
tf.app.flags.DEFINE_integer('num_test_instance', 50000, """Number of test images.""")

# Network Configuration
tf.app.flags.DEFINE_integer('batch_size', 100, """Number of images to process in a batch.""")

# Optimization Configuration
tf.app.flags.DEFINE_float('l2_weight', 0.0001, """L2 loss weight applied all the weights""")
tf.app.flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")
tf.app.flags.DEFINE_float('initial_lr', 0.1, """Initial learning rate""")
tf.app.flags.DEFINE_string('lr_step_epoch', "80.0,120.0,160.0", """Epochs after which learing rate decays""")
tf.app.flags.DEFINE_float('lr_decay', 0.1, """Learning rate decay factor""")
tf.app.flags.DEFINE_boolean('finetune', False, """Whether to finetune.""")

# Training Configuration
tf.app.flags.DEFINE_string('checkpoint', './alexnet_baseline_2/model.ckpt-399999', """Path to the model checkpoint file""")
tf.app.flags.DEFINE_string('output_file', './alexnet_baseline_2/eval.pkl', """Path to the result pkl file""")
tf.app.flags.DEFINE_integer('test_iter', 100, """Number of test batches during the evaluation""")
tf.app.flags.DEFINE_integer('display', 100, """Number of iterations to display training info.""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.95, """The fraction of GPU memory to be allocated""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")

FLAGS = tf.app.flags.FLAGS


def get_lr(initial_lr, lr_decay, lr_decay_steps, global_step):
    lr = initial_lr
    for s in lr_decay_steps:
        if global_step >= s:
            lr *= lr_decay
    return lr


def train():
    print('[Dataset Configuration]')
    print('\tImageNet test root: %s' % FLAGS.test_image_root)
    print('\tImageNet test list: %s' % FLAGS.test_dataset)
    print('\tNumber of classes: %d' % FLAGS.num_classes)
    print('\tNumber of test images: %d' % FLAGS.num_test_instance)

    print('[Network Configuration]')
    print('\tBatch size: %d' % FLAGS.batch_size)
    print('\tCheckpoint file: %s' % FLAGS.checkpoint)

    print('[Optimization Configuration]')
    print('\tL2 loss weight: %f' % FLAGS.l2_weight)
    print('\tThe momentum optimizer: %f' % FLAGS.momentum)
    print('\tInitial learning rate: %f' % FLAGS.initial_lr)
    print('\tEpochs per lr step: %s' % FLAGS.lr_step_epoch)
    print('\tLearning rate decay: %f' % FLAGS.lr_decay)

    print('[Evaluation Configuration]')
    print('\tOutput file path: %s' % FLAGS.output_file)
    print('\tTest iterations: %d' % FLAGS.test_iter)
    print('\tSteps per displaying info: %d' % FLAGS.display)
    print('\tGPU memory fraction: %f' % FLAGS.gpu_fraction)
    print('\tLog device placement: %d' % FLAGS.log_device_placement)

    sess = tf.Session()

    global_step = tf.Variable(0, trainable=False, name='global_step')

    FLAGS.test_dataset = "./val.txt"
    # Get images and labels of ImageNet
    print('Load ImageNet dataset')
    with tf.device('/cpu:0'):
        print('\tLoading test data from %s' % FLAGS.test_dataset)
        with tf.variable_scope('test_image'):
            test_images, test_labels = data_input.inputs(FLAGS.test_image_root, FLAGS.test_dataset, FLAGS.batch_size, False, num_threads=1, center_crop=True)



    # Build a Graph that computes the predictions from the inference model.
    images = tf.placeholder(tf.float32, [FLAGS.batch_size, data_input.IMAGE_HEIGHT, data_input.IMAGE_WIDTH, 3])
    labels = tf.placeholder(tf.int32, [FLAGS.batch_size])

#        images = tf.placeholder(tf.float32, [1, data_input.IMAGE_HEIGHT, data_input.IMAGE_WIDTH, 3])
#        labels = tf.placeholder(tf.int32, [1])



    # Build model
    with tf.device('/cpu:0'):
        hp = resnet.HParams(batch_size=FLAGS.batch_size,
                            num_gpus=1,
                            num_classes=FLAGS.num_classes,
                            weight_decay=FLAGS.l2_weight,
                            momentum=FLAGS.momentum,
                            finetune=FLAGS.finetune)
    network = resnet.ResNet(hp, [images], [labels], global_step)
    network.build_model()
    print('\tNumber of Weights: %d' % network._weights)
    print('\tFLOPs: %d' % network._flops)

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))


    sess.run(init)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=10000)
    if FLAGS.checkpoint is not None:
       saver.restore(sess, FLAGS.checkpoint)
       print('Load checkpoint %s' % FLAGS.checkpoint)
    else:
        print('No checkpoint file of basemodel found. Start from the scratch.')

    # Start queue runners & summary_writer
    tf.train.start_queue_runners(sess=sess)



    "============================================================================================================"
    "=================  Begin to insert restriction on selective layers  ========================================"
    "============================================================================================================"
    "NOTE: this model requires GPU (otherwise it'll report error while restoring the variables from the old graph to the new graph)"


    # get all the operators in the graph
    ops = [tensor for op in sess.graph.get_operations() for tensor in op.values()]
    graph_def = sess.graph.as_graph_def()

    def get_op_dependency(op):
      "get all the node that precedes the target op"
      cur_op = []
      #op = sess.graph.get_tensor_by_name("ranger_11/ranger_10/ranger_9/ranger_8/ranger_7/ranger_6/ranger_5/ranger_4/ranger_3/ranger_2/ranger_1/ranger/Relu_5:0").op
      cur_op.append(op)
      next_op=[]

      a = open("resnet-op.txt", "a") # save all the ops depend on the output op into file
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

    def restore_all_var(sess, scope_name, dup_cnt, all_var, dummy_scope_name, dummy_graph_dup_cnt, OLD_SESS):
      "need to map back the variable values to the ones under the new scope" 

      target_graph_prefix = get_target_scope_prefix(scope_name,dup_cnt,dummy_scope_name,dummy_graph_dup_cnt)

      tmp = [] 
      for each in all_var:
        #print( target_graph_prefix ,  each.name )
        sess.run( tf.assign(sess.graph.get_tensor_by_name(target_graph_prefix + each.name), OLD_SESS.run( OLD_SESS.graph.get_tensor_by_name( each.name ) )) )
      

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
        a = open("op.txt", "w")
        for n in ops:   
          a.write(n.name + "\n")
          

    # in resenet-18, Relu is renamed as relu
    act = "relu"
    op_follow_act = ["MaxPool", "Reshape", "AvgPool"]
    special_op_follow_act = "concat"
    up_bound = map(float, [7, 8, 7, 5, 11, 5, 12, 6, 11, 5, 12, 5, 14, 5, 12, 5, 66]) # upper bound for restriction
    low_bound = map(float, [0, 0, 0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0]) # low bound for restriction


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
    all_var = tf.global_variables() # all vars before duplication 

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
    # restore the variables to the latest path
     
    truncated_graphdef = modify_graph(sess, graph_dup_cnt, PREFIX, new_op_prefix, dummy_graph_dup_cnt, DUMMY_PREFIX)
    tf.reset_default_graph()
    sess = tf.Session()
    sess.as_default()
    #printgraphdef(truncated_graphdef)
    tf.import_graph_def(truncated_graphdef, name=DUMMY_PREFIX)   
    dummy_graph_dup_cnt+=1

    "restore all the variables from the orignial garph to the new graph"
    restore_all_var(sess, PREFIX, graph_dup_cnt, all_var, DUMMY_PREFIX, dummy_graph_dup_cnt, OLD_SESS)
#    printgraph(sess) 


    print("Finish graph modification!")
    print('')
    "============================================================================================================" 
    "============================================================================================================"


    "This is the name of the operator to be evaluated, we will find the corresponding one under the Ranger's scope"
    OP_FOR_EVAL = network.probs
    new_op_for_eval_name = get_op_with_prefix(OP_FOR_EVAL.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX) 
    print(new_op_for_eval_name, 'op to be eval')
    new_op_for_eval = sess.graph.get_tensor_by_name( new_op_for_eval_name + ":0") 

    # you can call this function to check the depenency of the final operator
    # you should see the bouding ops are inserted into the dependency
    # NOTE: the printing might contain duplicated output 
    #get_op_dependency(new_op_for_eval.op)

    # input to eval the results
    for i in range(2):
        test_images_val, test_labels_val = OLD_SESS.run([test_images[0], test_labels[0]]) 

    # evaluation on the old path
    preds = OLD_SESS.run(OP_FOR_EVAL, 
                feed_dict={network.is_train:False, images:test_images_val, labels:test_labels_val})
    print((np.argsort(np.asarray(preds)[0] )[::-1])[0:10])
    print('')

    # evaluation on the new path
    new_x = sess.graph.get_tensor_by_name( get_op_with_prefix(images.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)+":0")
    new_y = sess.graph.get_tensor_by_name( get_op_with_prefix(labels.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)+":0")
    new_is_train = sess.graph.get_tensor_by_name(  get_op_with_prefix(network.is_train.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)+":0")
    #new_prob2 = sess.graph.get_tensor_by_name(  get_op_with_prefix(model.prob2.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)+":0")

    preds = sess.run(new_op_for_eval, 
                feed_dict={new_is_train:False, new_x: test_images_val, new_y: test_labels_val})
    print((np.argsort(np.asarray(preds)[0] )[::-1])[0:10])





 


def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()
