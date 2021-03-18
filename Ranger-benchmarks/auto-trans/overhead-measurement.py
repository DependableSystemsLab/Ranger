"Insert this AFTER inserting range check into the original graph (Tested under TF 1.14)"
"this will get the FLOPs of the target op in the original and new graph"
"NOTE: you need to set the shape of the input op (placeholder) to be constant value, cannot set "None" otherwise the FLOPs cannot be obtained due to incomplete shape "

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

