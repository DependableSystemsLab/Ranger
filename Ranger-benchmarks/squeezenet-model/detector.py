
import os
import csv
import numpy as np
a = open("res.csv", "a")


bound = [958, 2018, 1587, 2470, 1892, 1228, 2003, 3024, 1736, 3031, 2687, 1408, 1903, 3329, 1661, 2285, 2749, 1041, 1884, 2126, 823, 1772, 2306, 748, 1086, 700]

bound = np.asarray(bound)
bound = bound.astype(float)
bound *= 1.05
print bound


detRes = np.zeros((10,4000))

for i in range(1):
 
    csvr = open( `i` + "-fi.csv", "r"  )
    act = csv.reader(csvr)

    symp = bound[i]

    print '----------'
    # index of input
    ind = 0
    for each in act:
	print ind, len(each) - 1
	'''
        each = each[:-1]
        each = np.asarray(each)
        each = each.astype(float)

        fiTime = 0
        for ea in each:
            if(ea >= symp):
                detRes[ind][fiTime] = 1 

            fiTime+=1
	'''
        ind+=1
	
    
