# Copyright (c) 2017 Andrey Voroshilov

import os
import tensorflow as tf
import numpy as np
import scipy.io
import time

from PIL import Image

from argparse import ArgumentParser
 


os.environ["CUDA_VISIBLE_DEVICES"] = "-1    "

def imread_resize(path):
    img_orig = scipy.misc.imread(path)
    img = scipy.misc.imresize(img_orig, (227, 227)).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    return img, img_orig.shape

def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)
    
def get_dtype_np():
    return np.float32

def get_dtype_tf():
    return tf.float32
    
# SqueezeNet v1.1 (signature pool 1/3/5)
########################################

def load_net(data_path):
    if not os.path.isfile(data_path):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % data_path)

    weights_raw = scipy.io.loadmat(data_path)
    
    # Converting to needed type
    conv_time = time.time()
    weights = {}
    for name in weights_raw:
        weights[name] = []
        # skipping '__version__', '__header__', '__globals__'
        if name[0:2] != '__':
            kernels, bias = weights_raw[name][0]
            weights[name].append( kernels.astype(get_dtype_np()) )
            weights[name].append( bias.astype(get_dtype_np()) )
    print("Converted network data(%s): %fs" % (get_dtype_np(), time.time() - conv_time))
    
    mean_pixel = np.array([104.006, 116.669, 122.679], dtype=get_dtype_np())
    return weights, mean_pixel

def preprocess(image, mean_pixel):
    swap_img = np.array(image)
    img_out = np.array(swap_img)
    img_out[:, :, 0] = swap_img[:, :, 2]
    img_out[:, :, 2] = swap_img[:, :, 0]
    return img_out - mean_pixel

def unprocess(image, mean_pixel):
    swap_img = np.array(image + mean_pixel)
    img_out = np.array(swap_img)
    img_out[:, :, 0] = swap_img[:, :, 2]
    img_out[:, :, 2] = swap_img[:, :, 0]
    return img_out

def get_weights_biases(preloaded, layer_name):
    weights, biases = preloaded[layer_name]
    biases = biases.reshape(-1)
    return (weights, biases)

def fire_cluster(net, x, preloaded, cluster_name):
    # central - squeeze
    layer_name = cluster_name + '/squeeze1x1'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='VALID')
    x = _act_layer(net, layer_name + '_actv', x)
    
    # left - expand 1x1
    layer_name = cluster_name + '/expand1x1'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x_l = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='VALID')
    x_l = _act_layer(net, layer_name + '_actv', x_l)

    # right - expand 3x3
    layer_name = cluster_name + '/expand3x3'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x_r = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='SAME')
    x_r = _act_layer(net, layer_name + '_actv', x_r)
    
    # concatenate expand 1x1 (left) and expand 3x3 (right)
    x = tf.concat([x_l, x_r], 3)
    net[cluster_name + '/concat_conc'] = x
    
    return x

def net_preloaded(preloaded, input_image, pooling, needs_classifier=False, keep_prob=None):
    net = {}
    cr_time = time.time()

    x = tf.cast(input_image, get_dtype_tf())

    # Feature extractor
    #####################
    
    # conv1 cluster
    layer_name = 'conv1'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='VALID', stride=(2, 2))
    x = _act_layer(net, layer_name + '_actv', x)
    x = _pool_layer(net, 'pool1_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='VALID')

    # fire2 + fire3 clusters
    x = fire_cluster(net, x, preloaded, cluster_name='fire2')
    fire2_bypass = x
    x = fire_cluster(net, x, preloaded, cluster_name='fire3')
    x = _pool_layer(net, 'pool3_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='VALID')

    # fire4 + fire5 clusters
    x = fire_cluster(net, x, preloaded, cluster_name='fire4')
    fire4_bypass = x
    x = fire_cluster(net, x, preloaded, cluster_name='fire5')
    x = _pool_layer(net, 'pool5_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='VALID')

    # remainder (no pooling)
    x = fire_cluster(net, x, preloaded, cluster_name='fire6')
    fire6_bypass = x
    x = fire_cluster(net, x, preloaded, cluster_name='fire7')
    x = fire_cluster(net, x, preloaded, cluster_name='fire8')
    x = fire_cluster(net, x, preloaded, cluster_name='fire9')
    
    # Classifier
    #####################
    if needs_classifier == True:
        # Dropout [use value of 50% when training]
        x = tf.nn.dropout(x, keep_prob)
    
        # Fixed global avg pool/softmax classifier:
        # [227, 227, 3] -> 1000 classes
        layer_name = 'conv10'
        weights, biases = get_weights_biases(preloaded, layer_name)
        x = _conv_layer(net, layer_name + '_conv', x, weights, biases)
        x = _act_layer(net, layer_name + '_actv', x)
        
        # Global Average Pooling
        x = tf.nn.avg_pool(x, ksize=(1, 13, 13, 1), strides=(1, 1, 1, 1), padding='VALID')
        net['classifier_pool'] = x
        
        x = tf.nn.softmax(x)
        net['classifier_actv'] = x
    
    print("Network instance created: %fs" % (time.time() - cr_time))
   
    return net
    
def _conv_layer(net, name, input, weights, bias, padding='SAME', stride=(1, 1)):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, stride[0], stride[1], 1),
            padding=padding)
    x = tf.nn.bias_add(conv, bias)
    net[name] = x
    return x

def _act_layer(net, name, input):
    x = tf.nn.relu(input)
    net[name] = x
    return x
    
def _pool_layer(net, name, input, pooling, size=(2, 2), stride=(3, 3), padding='SAME'):
    if pooling == 'avg':
        x = tf.nn.avg_pool(input, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                padding=padding)
    else:
        x = tf.nn.max_pool(input, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                padding=padding)
    net[name] = x
    return x

def build_parser():
    ps = ArgumentParser()
    ps.add_argument('--in',             dest='input', help='input file', metavar='INPUT', required=False)
    ps.add_argument('--fool',           dest='fool', type=int, help='if image needs to be altered to fool the network classification (argument - class number)', metavar='FOOL')
    return ps

def main():
    import time
 

    # Loading ImageNet classes info
    classes = []
    with open('synset_words.txt', 'r') as classes_file:
        classes = classes_file.read().splitlines()

    # Loading network
    data, sqz_mean = load_net('sqz_full.mat')

    config = tf.ConfigProto(log_device_placement = False)
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'

    g = tf.Graph()
    

    sess = tf.Session()

    # Building network
    image = tf.placeholder(dtype=get_dtype_tf(), shape=[None,224,224,3], name="image_placeholder")
    keep_prob = tf.placeholder(get_dtype_tf())
    sqznet = net_preloaded(data, image, 'max', True, keep_prob)


    "============================================================================================================"
    "=================  Begin to insert restriction on selective layers  ========================================"
    "============================================================================================================"

    # get all the operators in the graph
    ops = [tensor for op in sess.graph.get_operations() for tensor in op.values()]
    graph_def = sess.graph.as_graph_def()

    def get_op_dependency(op):
      "get all the node that precedes the target op"
      cur_op = []
      a = open("dep.txt", "w")
      #op = sess.graph.get_tensor_by_name("ranger_11/ranger_10/ranger_9/ranger_8/ranger_7/ranger_6/ranger_5/ranger_4/ranger_3/ranger_2/ranger_1/ranger/Relu_5:0").op
      cur_op.append(op)
      next_op=[]
      while(not (next_op==[] and cur_op==[])):
          next_op = []
          for each in cur_op:
              printline = False
              for inp in each.inputs:
                  printline = True
                  a.write(str(inp) + "\n")
                  next_op.append(inp.op) 
              if(printline): 
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
    up_bound = map(float, [958., 2018, 1587, 2470, 1892, 1228, 2003, 3024, 1736, 3031, 2687, 1408, 1903, 3329, 1661, 2285, 2749, 1041, 1884, 2126, 823, 1772, 2306, 748, 1086, 0]) # upper bound for restriction
    low_bound = map(float, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # low bound for restriction


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
    OP_FOR_EVAL = sqznet['classifier_actv'] 
    new_op_for_eval_name = get_op_with_prefix(OP_FOR_EVAL.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX) 
    print(new_op_for_eval_name, 'op to be eval')
    new_op_for_eval = sess.graph.get_tensor_by_name(new_op_for_eval_name + ":0") 

    # you can call this function to check the depenency of the final operator
    # you should see the bouding ops are inserted into the dependency
    # NOTE: the printing might contain duplicated output 
    #get_op_dependency(new_op_for_eval.op)

    path = "./fiImg/"

    #path = "./outOfBoundData/"
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f: 
            files.append(os.path.join(r, file)) 

    # Loading image
    img_content, orig_shape = imread_resize(files[0])
    img_content = scipy.misc.imresize(img_content, [224,224,3]) 



    sqznet_results = OLD_SESS.run(OP_FOR_EVAL, feed_dict={image: [preprocess(img_content, sqz_mean)], keep_prob: 1.})[0][0][0]
    pred = (np.argsort(sqznet_results)[::-1])[0:10]
    print(pred)

    # evaluation on the new path
    new_x = sess.graph.get_tensor_by_name( get_op_with_prefix(image.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)+":0") 
    new_prob = sess.graph.get_tensor_by_name( get_op_with_prefix(keep_prob.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)+":0") 
    sqznet_results = sess.run(new_op_for_eval, feed_dict={new_x: [preprocess(img_content, sqz_mean)], new_prob: 1.})[0][0][0]
    pred = (np.argsort(sqznet_results)[::-1])[0:10]
    print(pred)    


if __name__ == '__main__':
    main()