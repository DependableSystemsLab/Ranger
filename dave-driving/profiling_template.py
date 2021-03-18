# this is a template for generating code to derive the restriction bounds

import math
"need filling"

num_images = int(len(x_data) * percentage_of_img)

outputFile=

val_dist_folder = 'ACT-dist/'
if(not os.path.isdir(val_dist_folder)):
    os.mkdir(val_dist_folder)

def storeAct(maxFileName, minFileName, data, save_raw_data):
    data = data.flatten()
    maxFileName = val_dist_folder + maxFileName
    wri1 = open(maxFileName, 'a')

    minFileName = val_dist_folder + minFileName
    wri2 = open(minFileName, 'a')

    minVal = np.amin(data)
    maxVal = np.amax(data)

    if(not save_raw_data):
        # only save the maximal value per op's output, otherwise the size of the file will grow very large  
        wri1.write(`maxVal` + ",")
        wri2.write(`minVal` + ",")
    else:
        for each in data:
            wri1.write(`each` + ",")
            wri2.write(`each` + ",")
        

import csv
import numpy as np


def get_rest_val(total_act, restriction_bound_pert, act):
    up_bounds = []
    low_bounds = []
    for i in range(total_act):
        act_val = []

        maxFileName = val_dist_folder + "act-max-{}.csv".format(i+1)
        minFileName = val_dist_folder + "act-min-{}.csv".format(i+1)
        

        res = open(maxFileName, "r")
        data = csv.reader(res)
        for line in data:
            line = line[:-1]
            act_val += (line)
        act_val = np.asarray(act_val, float)
        max_sort_act = np.sort(act_val)


        if( 'relu' in act.lower() ):
            low_bound = 0
        elif( 'elu' in act.lower() ):
            low_bound = -1
        else:
            res = open(maxFileName, "r")
            data = csv.reader(res)
            for line in data:
                line = line[:-1]
                act_val += (line)
            act_val = np.asarray(act_val, float)
            min_sort_act = np.sort(act_val)
            low_bound = min_sort_act[ (len(max_sort_act)-1) -  int(len(max_sort_act) * restriction_bound_pert - 1 ) ]

        up_bound = math.ceil( max_sort_act[ int(len(max_sort_act) * restriction_bound_pert - 1 ) ])
        up_bounds.append(up_bound)
        low_bounds.append(low_bound)

        print("Restriction value for the < {} > ACT layer: < {} , {} >".format(i, up_bound, low_bound  ))

 
    import yaml 
    with open(outputFile) as f:
         list_doc = yaml.load(f)
 
    list_doc['up_bound'] = up_bounds
    list_doc['low_bound'] = low_bounds 
 

    with open('new.yaml', "w") as f:
        yaml.dump(list_doc, f,default_flow_style=False) 

ops = [tensor for op in sess.graph.get_operations() for tensor in op.values()]

    
"find the ACT ops"
ACT_op = []
for cur_op in ops:   
    if( act in cur_op.name and ("gradients" not in cur_op.name)  ): 
        ACT_op.append( sess.graph.get_tensor_by_name(cur_op.name) )
total_act = len(ACT_op)


for i in range(num_images): 

    img = [ x_data[i] ] # [] for appending the dim of the tensor


    "need filling"
    value = sess.run()
    
    for j in range(total_act):
    	storeAct("act-max-{}.csv".format(j+1),"act-min-{}.csv".format(j+1), value[j], save_raw_data)



get_rest_val(total_act, restriction_bound_pert, act)