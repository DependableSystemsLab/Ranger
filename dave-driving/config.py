# The following is the snippet to be inserted to the source program to enable selecitve range restriction
#
# Please ensure consistent indent with the source program
#


# param for automated transformaiton 
# NOTE: the following params are for the default setting. 
#	 	Change it if needed (e.g., if you use a different name for the activation function)
#	
act = "Relu" 
op_follow_act = ["MaxPool", "Reshape", "AvgPool"]
special_op_follow_act = "concat"

# below are the bound values for the Nvidia-Dave driving model
up_bound = [2.8, 26., 39., 94., 164., 562., 2538., 2368., 2431.] # upper bound for restriction
low_bound = [0.,0.,0.,0.,0.,0.,0.,0.,0.] # low bound for restriction


### put it in YAML file format


# inserting ranger
from ranger import *
# instrument the original graph to create a new graph with selective range restriction
# OLD_SESS is the original graph
# sess is the new graph
OLD_SESS, sess, dup_cnt, dummy_graph_dup_cnt = insertRanger(sess, act, op_follow_act, special_op_follow_act, up_bound, low_bound ) 

# To evaluate an operator in the new graph, you'll need to use its counterpart in the NEW graph 
# I.e., you can't use the old operator in the new graph
# For example: output = sess.run(model.y, feed_dict{input_x: XX, .. } )
# 				You'll need to find the new operator of model.y and all the operators in ``feed_dict'' in the new graph
# 				This can be done by calling the ``get_op_from_new_graph'' function
# 				E.g., To find model.y ==> model.y = get_op_from_new_graph( ...., op = model.y )
# 				then you can use model.y in the new graph
#
OP_FOR_EVAL = get_op_from_new_graph(sess =sess, op = model.y, graph_dup_cnt= dup_cnt, dummy_graph_dup_cnt = dummy_graph_dup_cnt)
INPUT_x = get_op_from_new_graph(sess =sess, op = model.x, graph_dup_cnt= dup_cnt, dummy_graph_dup_cnt = dummy_graph_dup_cnt)
INPUT_keep_prob = get_op_from_new_graph(sess =sess, op = model.keep_prob, graph_dup_cnt= dup_cnt, dummy_graph_dup_cnt = dummy_graph_dup_cnt)





