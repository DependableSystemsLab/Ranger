# Insert this script into the TF program after you've loaded the model and ready to run on it
# this script will modify the default tf session to insert the restriction ops
# 
# NOTE: 
#   1) You should NOT use nested graph as the default graph will be reset with the new duplicated graph 
#     WHY need to reset the default graph:
#     TensorFlow has a 2GB limit on the graphdef, so if we don't reset the graph to contain only the new graph, 
#     The graph will contain lots of old operators in prior duplication and the graph size will quickly explode
#     This becomes a problem for large model or model that requires lots of restriction ops (e.g., VGG16, ResNet, SqueezeNet)
#   2) This code assume the session variable is called "sess" (change it if needed)
#   3) All the operators in the original graph are duplicated in the new graph under a new scope name. 
#      If you want to find the counterpart of target_op from the original graph, call get_target_scope_prefix()
#      to get the scope name, the name of the new op is: target_graph_prefix + target_op.name (see line 339)


# INPUT:
#   1) restriction bounds 
#   2) name of the ops to be bounded (default name is provided, but the op name might be renamed in your program)
# OUTPUT:
#   A new graph (loaded into current default graph) in which the restricion ops are inserted
# HOW TO evaluate:
#   You can call the original op to evaluate the model (e.g., model.y) and find its replica in the new graph and evaluate it



"================================================================================================================"
"=================  Begin to insert restriction ops on selective layers  ========================================"
"================================================================================================================" 


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



PREFIX = 'ranger' # scope name for the duplication in which we insert the restricion op
DUMMY_PREFIX = 'dummy' # scope name for the dummy duplication, which is for resetting the default graph (no new op is inserted)
graph_dup_cnt = 0 # count the number of duplication, used to track the scope prefix of the new op
dummy_graph_dup_cnt = 0 # count the num of dummy graph duplication, used to track the scope prefix of the op



op_cnt = 0 # num of op
act_cnt = 0 # num of act
check_follow = False # flag for checking the following op (e.g., when the current op is ACT, we perform checking on the following op)  
new_op_prefix = "bound_op_prefix" # prefix of the newly created restricion ops (tf.maximum and tf.minimum)
OLD_SESS = sess # keep the old session, because the default graph will be reset with the duplicated one
train_var = tf.global_variables() # all vars before duplication



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
        print("bounding: %s with (%f, %f)"%(cur_op.name, low_bound[act_cnt], up_bound[act_cnt]) )
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
            print("bounding: %s with (%f, %f)"%(cur_op.name, low, up) )
            rest = tf.maximum(bound_tensor, low )
            rest = tf.minimum(rest, up )
        except:
          with tf.name_scope(new_op_prefix) as scope: # the restricion ops will have the special scope prefix name
            bound_tensor = sess.graph.get_tensor_by_name( get_op_with_prefix(cur_op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)  )
            print("bounding: %s with (%f, %f)"%(cur_op.name, low, up) )
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

"================================================================================================================"
"=================  Finish inserting restriction ops on selective layers  ========================================"
"================================================================================================================" 






"Put the op of interest to be evaluated on the right hand of below statement"
OP_FOR_EVAL =  "Name of the op you want to evaluate. You'll also need to do the same for the input ops - line 364" 
new_op_for_eval_name = get_op_with_prefix(OP_FOR_EVAL.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX) 
print(new_op_for_eval_name, 'op to be eval')
# the counterpart of OP_FOR_EVAL in the new graph
new_op_for_eval = sess.graph.get_tensor_by_name(new_op_for_eval_name + ":0") 


# you can call this function to check the depenency of the target operator
# you should see the bouding ops are inserted into the dependency
# NOTE: the printing might contain duplicated output
#get_op_dependency(new_op_for_eval.op)

# you can specify input here
tx = test_x[:3,:]
ty = test_y[:3,:]

# evaluation on the old graph
# change the below to match semantic of your TF program
# NOTE: we use OLD_SESS because OP_FOR_EVAL is in the original graph
preds = OLD_SESS.run(OP_FOR_EVAL, feed_dict={x: tx, y: ty, prob: 1.})
print(preds)
print('')

# for all the inputs in feed_dict, we need to find its counterpart in the new graph
new_x = sess.graph.get_tensor_by_name( get_op_with_prefix(x.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)+":0")
new_y = sess.graph.get_tensor_by_name( get_op_with_prefix(y.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)+":0")
new_prob = sess.graph.get_tensor_by_name( get_op_with_prefix(prob.op.name, graph_dup_cnt, PREFIX, dummy_graph_dup_cnt, DUMMY_PREFIX)+":0")
# evaluation on the new path
preds = sess.run(new_op_for_eval, feed_dict={new_x: tx, new_y: ty, new_prob: 1.})
print(preds)


