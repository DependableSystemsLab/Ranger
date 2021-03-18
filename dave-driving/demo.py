import tensorflow as tf
import scipy.misc
import model
import cv2
from subprocess import call
#import driving_data # to import this lib, you'll need to specify the path to the dataset
import time
import TensorFI as ti
import datetime
import os
from argparse import ArgumentParser

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

parser = ArgumentParser()

# set default: false
# show tensorboard (graph)
# no fault injection first, just show the graph


# show what the wront steering angle look like
# show example 1000 first, then another one

parser.add_argument('--isInsertRanger', type=lambda x: (str(x).lower() == 'true'), default=False,
                    help="flag to decide whether inserting Ranger into the model") 
parser.add_argument('--input', type=str,
                    help="path to input image") 

parser.add_argument('--isFI', type=lambda x: (str(x).lower() == 'true'), default=False,
                    help="flag to decide whether to inject fault or not") 

parser.add_argument('--saveTFboard', type=lambda x: (str(x).lower() == 'true'), default=False,
                    help="save summary into tensorboard logs") 

parser.add_argument('--showNode', type=lambda x: (str(x).lower() == 'true'), default=False,
                    help="show num of nodes in the graph (from the target op)") 




args = parser.parse_args() 



"You need to provide the pre-trained weights"
"or you can train the model by yourself"
sess = tf.InteractiveSession(  )
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

#tx, ty = driving_data.LoadWholeSet() # you can try to load the whole/test/training set

# file for saving results
resFile = open("fi-result.csv", "a")

# ops that will be evaluated 
global OP_FOR_EVAL
global INPUT_x
global INPUT_keep_prob

OP_FOR_EVAL = model.y
INPUT_x = model.x
INPUT_keep_prob = model.keep_prob

def insertRanger(sess, ifInsert):

    num_of_nodes_bounded = 0 # count the num of nodes being bounded
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


    def get_op_dependency(op):
      "get all the node that precedes the target op" 
      print("=======================================================")
      print("======== Printing the dependency to op: %s ======="%op.name.replace(PREFIX+"/" , "").replace(DUMMY_PREFIX+"/" , ""))
      print("=======================================================")
      cur_op = []
      total_num_of_ops = 0
      #op = sess.graph.get_tensor_by_name("ranger_11/ranger_10/ranger_9/ranger_8/ranger_7/ranger_6/ranger_5/ranger_4/ranger_3/ranger_2/ranger_1/ranger/Relu_5:0").op
      a = open('alex-dep.txt', "w")
      cur_op.append(op)
      next_op=[]
      while(not (next_op==[] and cur_op==[])):
          next_op = []
          for each in cur_op:
              printline = False
              for inp in each.inputs:
                  printline = False
                  if("Variable" not in inp.name and "Minimum/y" not in inp.name and "Maximum/y" not in inp.name and "/shape" not in inp.name):
                      if("Minimum" in inp.name or "Maximum" in inp.name):
                        print("%s <======"%inp.op.name.replace(PREFIX+"/" , "").replace(DUMMY_PREFIX+"/" , ""))
                      else:
                        print(inp.op.name.replace(PREFIX+"/" , "").replace(DUMMY_PREFIX+"/" , ""))
                      total_num_of_ops +=1
                  #a.write(str(inp) + "\n")
                  next_op.append(inp.op) 
              if(printline): 
                print('')
                #a.write("\n\n")
          cur_op = next_op 
      print("Total num of nodes: %d"%total_num_of_ops)
      print("=====================")

    global OP_FOR_EVAL
    global INPUT_x
    global INPUT_keep_prob

    if not ifInsert:
        get_op_dependency(OP_FOR_EVAL.op)

        return sess, sess

    "============================================================================================================"
    "=================  Begin to insert restriction on selective layers  ========================================"
    "============================================================================================================"

    print("=====================================================================")
    print("======== Begin to insert restriction ops on selective layers ========")
    print("=====================================================================")
    # get all the operators in the graph
    ops = [tensor for op in sess.graph.get_operations() for tensor in op.values()]
    graph_def = sess.graph.as_graph_def()



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
    up_bound = [2.8, 26., 39., 94., 164., 562., 2538., 2368., 2431.] # upper bound for restriction
    low_bound = [0.,0.,0.,0.,0.,0.,0.,0.,0.] # low bound for restriction


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
            print("bounding: %s with (%f, %f)"%(cur_op.name, low_bound[act_cnt], up_bound[act_cnt]) )
            num_of_nodes_bounded += 1
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
                print("bounding: %s with (%f, %f)"%(cur_op.name, low, up) )
                num_of_nodes_bounded += 1
                rest = tf.maximum(bound_tensor, low )
                rest = tf.minimum(rest, up )
            except:
              with tf.name_scope(new_op_prefix) as scope: # the restricion ops will have the special scope prefix name
                bound_tensor = sess.graph.get_tensor_by_name( get_op_with_prefix(cur_op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)  )
                print("bounding: %s with (%f, %f)"%(cur_op.name, low, up) )
                num_of_nodes_bounded += 1
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


    print("======== Finish graph modification! ========")
    print("Total num of nodes that are bounded: %d"%num_of_nodes_bounded)
    print("")

    "============================================================================================================" 
    "============================================================================================================"



    "This is the name of the operator to be evaluated, we will find the corresponding one under the Ranger's scope"
    OP_FOR_EVAL = model.y
    new_op_for_eval_name = get_op_with_prefix(OP_FOR_EVAL.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX) 
    #print("Evaluating %s under the new graph"%(OP_FOR_EVAL.op.name))
    #print("")
    new_op_for_eval = sess.graph.get_tensor_by_name(new_op_for_eval_name + ":0") 
    new_x = sess.graph.get_tensor_by_name( get_op_with_prefix(model.x.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)+":0") 
    new_prob = sess.graph.get_tensor_by_name( get_op_with_prefix(model.keep_prob.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)+":0")

    OP_FOR_EVAL = new_op_for_eval
    INPUT_x = new_x
    INPUT_keep_prob = new_prob

    get_op_dependency(OP_FOR_EVAL.op)

    return sess, OLD_SESS


 
sess, OLD_SESS = insertRanger(sess, args.isInsertRanger) # return the new sess and the old sess 

if args.isFI:
  fi = ti.TensorFI(sess, logLevel = 50, name = "convolutional", disableInjections=True)

  

print("")
print("======== Printing the results ========")
 
full_image = scipy.misc.imread( args.input, mode="RGB")
image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0

 
golden = sess.run(OP_FOR_EVAL, feed_dict={INPUT_x: [image], INPUT_keep_prob: 1.0})[0][0]   

if args.isFI:
  fi.turnOnInjections()

totalFI = 0.
sdcCount = 0  
fiTime = 1
for j in range(fiTime):
    degrees = sess.run(OP_FOR_EVAL, feed_dict={INPUT_x: [image], INPUT_keep_prob: 1.0})[0][0]  

    #resFile.write(`degrees` + ",")
print("prediction angle without fault: %f"%(golden))

if args.isFI:
  print("prediction angle with fault: %f"%(degrees))

#print(i, golden, ' --> ', degrees,  j)
#resFile.write("\n")


if args.saveTFboard:
  # Make the log files in TensorBoard 
  logs_path = "./logs"
  logWriter = tf.summary.FileWriter( logs_path, sess.graph )












