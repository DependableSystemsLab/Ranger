# Library of fault injection functions called at runtime for common operations in TensorFlow
# NOTE: These are called by the corresponding functions inserted in the TensorFlow graph at RUNTIME

import tensorflow as tf
import numpy as np
import logging
import conv2d
from fiConfig import * 
from fiLog import *
from threading import current_thread

# FIXME: Add this to the list of dependencies for this module
from sklearn.neighbors import KNeighborsClassifier	
from sklearn.utils.extmath import softmax

# global variable to determine fine grained levels of logging
# WARNING: Setting these to True may generate a lot of log data

logReturn = False	# log return values of functions	
logArgs = False	# log arguments of operators
logInjection = False	# log fault injection and checking

# This is the initialization function for the config file 
# and is called from TensorFI.py's constructor

np.random.seed(25) # set random seed for reproducibility purpose


# NOTE: This has to be in this module or else fiConf won't be accessible
def initFIConfig(fiParams):
	"Initialize the global variable fiConf with the params"
	global fiConf
	global count
	# instance of the current op (e.g., 3 ADD op means 3 instances of ADD op)
	global visitedOp
	# random instance of the op to be injected 
	global randInstanceMap
	# order of the current op (e.g., the sequence of the current op in all of the op in the dataflow graph)
	global totalVistedOp
	# which op to be injected in the whole run
	global injectedOp

	# count the number of ReLu inst
	global reluNum
	global matNum
	global addNum


	fiConf = FIConfig(fiParams)
	logging.debug("Initialized config file : " + str(fiConf))
	
	# Setup the random seed for the fault injector if one is specified
	if fiConf.faultSeed: np.random.seed( fiConf.faultSeed )	 


	# Initialize the count of the selected operations to 0 (for skipCount)
	count = 0
	visitedOp = {}
	randInstanceMap = {}
	totalVistedOp = 0
	injectedOp = 0
	

	reluNum = 0


	matNum = -1
	addNum = -1

	return fiConf

# End of fiConfing

def getFIConfig():
	"Return the fiConfig that was initialized"
	global fiConf
	return fiConf
# End of getFIConfig

# These functions have to do with the faultLog and are called from TensorFI.py

faultLogs = { }		# Global map of Threads to their fault logs

def initFILog(name):
	"Initialize the fault injection log - optionally specify a thread number"
	
	global faultLogs
	global logName

	logName = name
	faultLog = FILog(logName)

	# Add the fault log to the log for the current thread
	current = current_thread()
	faultLogs[ current ] = faultLog

	# logging.debug("Initialized faultLog for thread " + str(current) + " as " + logName)

# End of initFILog

def getCurrentFaultLog():
	"Return the fault log for the current thread (if it exists), add it otherwise"
	# Precondition: faultLogs != None

	global faultLogs
	global logName
	
	current = current_thread()
	faultLog = None
	
	# If we cannot find the faultLog for the current thread, add it to the faultLogs
	# FIXME: This doesn't work because TensorFlow uses its own threading infrastructure
	# and ThreadIDs are not the same during log creation time and log access time
	# So we always end up writing to the first entry of the faultLogs dictionary
	if not faultLogs.has_key(current):
		# logging.debug("Cannot find fault log for " + str(current) )
		faultLog = FILog(logName + "-" + current.name)
		faultLogs[ current ] = faultLog
		# faultLog = faultLogs.values()[0]
	else:
		# Otherwise, return the fault log for the current thread
		faultLog = faultLogs[current]

	# logging.debug("Returning fault log " + str(faultLog) + " for thread " + str(current) )
	return faultLog

# End of getCurrentFaultLog

def logRun(runCount):
	"Update the run count in the log file"
	
	global count

	# Reset the count on a new run
	count = 0
	faultLog = getCurrentFaultLog()	# Get the fault log for the current thread

	# Log the runCount and start a new section of the logFile
	faultLog.updateRunCount( runCount ) 
	faultLog.dashedLine()

# End of logRun

# These are the basic fault injection functions that're called at runtime
# NOTE: We need to first call initFIConfig before these are called 

def perturb(val):
	"Inject a single fault in res - fault type depends on config param"
	# Precoditions: injectScalar != None && injectTensor != None
	
	faultLog = getCurrentFaultLog()	# Get the fault log for the current thread

	isScalar = np.isscalar(val)
	vType = val.dtype

	if logInjection:
		logging.debug("\tPerturbing " + str(val)  + " of type: " + str(vType) + " isScalar: " + str(isScalar) )

	
	# Check if the object is a scalar or a tensor, and call the corresponding injection function
	if isScalar: 
		res = fiConf.injectScalar( vType, val.copy()) 
	else:  
		res = fiConf.injectTensor( vType, val.copy())  

	# Enter an entry in the fault log that we injected a fault here
	faultLog.updateOriginal( val )
	faultLog.updateInjected( res )

	return res

# End of perturb

def condPerturb(op, res):
	"Calls the perturb function if and only if the op Operation is included for injection"
	
	# Pre-condition: injectMap != None && skipCount != None 
	global count	# Keeps track of how many times the selected operation(s) are executed
	global visitedOp

	faultLog = getCurrentFaultLog()	# Get the fault log for the current thread
	
	if logInjection: 
		logging.debug("\tChecking if operation " + str(op) + " is chosen for injection")
	
	# Check if the operation is chosen for injection and if so, inject a fault	
 
	if fiConf.isSelected(op): 
		count = count + 1	# If it's selected, then update the execution count

		if logInjection: logging.debug("\tOperation " + str(op) + " is chosen for injection")
		
		# Enter the op and count in the faultLog - as we won't have access to it later
		# NOTE: This is not actually written to the logFIle till faultLog.commit is called
		#		so we won't write to the log if a fault is not injected into it
		faultLog.updateOp( op )
		faultLog.updateCount( count )
		
		# If the operation exceeds the number of times it is to be skipped (default=0)
		if (count > fiConf.skipCount):	

			"inject faults based on the error rate"
			if(fiConf.injectMode == "errorRate" ):
				# Retreive the probability of perturbing this instruction
				# and generate a random number in the interval [0, 1]
				# and only perturb it only if the random no. <= the probability 

				if(str(op) != "Ops.IDENTITY"):
					print(str(op), res.shape) 
				
				prob = fiConf.getProbability(op)
				rn = np.random.random()		# random.random returns a number in [0, 1] 
				if (rn <= prob):     
					res = perturb(res) # Perturb is called to inject the fault  
#					faultLog.commit()  # Write the log entry to the fault log 	 
			
			"inject faults based on the dynamic instance of op, i.e., inject one instance for each op"
			if(fiConf.injectMode == "dynamicInstance"):
				# Retreive the total instances of this instruction
				# each operation will be injected once only
				# and generate a random number to select a random instance of the operation
				# and only perturb it only if the current instance has been selected 
				instance = fiConf.getInstance(op)  
				#randInstanceMap[op] = 1

				
				if (not randInstanceMap.has_key(op)): 
					# random instance of the selected op to be injected
					randInstanceMap[op] = np.random.randint(low=1, high=instance+1)	
					faultLog.updateInjectedInstance(randInstanceMap[op], instance)
				

				# first instance of the op
				if(not visitedOp.has_key(op)):	visitedOp[op] = 1	
				# not the first instance of op
				else:							visitedOp[op] += 1	

 


				# determine if the current instance is selected for injection 
				if(visitedOp[op] == randInstanceMap[op]):    
					res = perturb(res)

					faultLog.commit()
				# current run has finished, re-initialize the visit table for the next run (optional)
				if(visitedOp[op] == instance):
					visitedOp[op] = 0  

			"inject one fault per run"
			if(fiConf.injectMode == "oneFaultPerRun"):
				# refer the global variable for remembering the order of the current op
				global totalVistedOp
				global injectedOp
				# get the amount of total op
				totalInstance = fiConf.totalInstance
				totalVistedOp += 1


				# one op to be injected in the whole run
				if(injectedOp == 0):
					
					#############
					injectedOp = np.random.randint(low= 1    , high=totalInstance) 
					########

					faultLog.updateInjectedInstance(injectedOp, totalInstance)

				
				if(totalVistedOp == injectedOp):
					res = perturb(res)
					
					faultLog.commit()
				# current run has finished, re-initialize the visit table for the next run (optional)
				if(totalVistedOp == totalInstance):
					totalVistedOp = 0
					injectedOp = 0

		# Done with if count

	# Done with if isSelected
	return res

# End of condPerturb

# This is a specialized function to cast into values of different types	
def castType(type):
	"Returns the appropriate injection function based on the type"
	
	# Create specialized functions for each type
	# FIXME: Only 4 types are supported now. Support more types later.
	def castFloat32(value):
		logging.debug("Casting to " + str(type))
		return np.float32(value) 
	def castInt32(value):
		logging.debug("Casting to " + str(type))
		return  np.int32(value) 
	def castInt64(value):
		logging.debug("Casting to " + str(type))
		return np.int64(value)
	def castFloat64(value):
		logging.debug("Casting to " + str(type))
		return np.float64(value)
	
	# Check the type parameter and return the appropriate function
	if (type==np.float32):
		return castFloat32
	elif (type==np.int32):
		return castInt32
	elif (type==np.int64):
		return castInt64
	elif (type==np.float64):
		return castFloat64
	else:
		raise TypeError("Unknown type " + type)
	return None
# End of castType

# Debugging function to log the values of the arguments
# if and only if logArgs is set to True
def getArgs(*args):
	"Return a string of the args if logArgs is True; Empty String otherwise"
	res = " "
	if logArgs:
		res +="( "
		for arg in args:
			res = res + " , " + str(arg)
		res += " )"
	return res

# Start the implementation of the injectFault functions for each op type

# This is a special case for the Cast function which needs to remember the type
# We use closures to remember the type and cast it appropriately at "runtime"
def createInjectFaultCast(type):
	"Returns a Function to call injectFault on cast nodes"
	
	castInto = castType(type) 	# get the appropriate casting function for the type

	def injectFaultCast(a, b = None):
		"Inject a fault into a Cast instruction"
		logging.debug("Calling Operator Cast " + getArgs(a, b))
		# If we're given 2 parameters, treat it as the default case
		if b != None:
			res = np.cast(a, b)
		else:
			# Call the function for this type with 'a'
			res = castInto(a)
		res = condPerturb(Ops.CAST, res)

		if logReturn: logging.debug("\tReturning " + str(res) )
		return res

	# Return the injectFaultCast function
	return injectFaultCast


def injectFaultNoop():
	"Inject a fault in the Noop operaton - does nothing"
	logging.debug("Calling Operator Noop") 
	# No need to call Perturb as there's nothing to return
	return

def injectFaultAssign(a, b):
	"Inject a fault in the assignement operation"
	logging.debug("Calling Operator Assigment " + getArgs(a, b))
	res = b		# FIXME: Check semantics of assignment operator

	sess = tf.Session()
	op = tf.assign(a,b)
	res = sess.run(op)

	res = condPerturb(Ops.ASSIGN, res)
	if logReturn: logging.debug("\tReturning from Assignment " + str(res)  )
	return res	

def injectFaultIdentity(a):
	"Inject a fault in the identitiy operation"	
	logging.debug("Calling Operator Identity " + getArgs(a))
	res = a
	res = condPerturb(Ops.IDENTITY, res)
	if logReturn: logging.debug("\tReturning from Identity " + str(res) )
	return res	

def injectFaultAdd(a, b):
	"Function to call injectFault on Add nodes"
	logging.debug("Calling Operator Add " + getArgs(a, b))
	resOp = tf.add(a, b)
	with tf.Session() as sess:
		res = resOp.eval()

	res = condPerturb(Ops.ADD, res)


	if logReturn: logging.debug("\tReturning from Add " + str(res) )
	return res	

def injectFaultSub(a, b):
	"Function to call injectFault on Sub nodes"
	logging.debug("Calling Operator Sub " + getArgs(a, b))
	res = a - b
	res = condPerturb(Ops.SUB, res)
	if logReturn: logging.debug("\tReturning from Sub " + str(res) )
	return res	

def injectFaultMul(a, b):
	"Function to call injectFault on Mul nodes"
	logging.debug("Calling Operator Mul " + getArgs(a, b))
	res = a * b
	res = condPerturb(Ops.MUL,res)
	if logReturn: logging.debug("\tReturning from Mul " + str(res) )
	return res

def injectFaultSquare(a):
	"Function to call injectFault on Square nodes"
	logging.debug("Calling Operator Square " + getArgs(a))
	res = a * a
	res = condPerturb(Ops.SQUARE,res)
	if logReturn: logging.debug("\tReturning from Square " + str(res) )
	return res

def injectFaultShape(a):
	"Function to call injectFault on Shape nodes"
	logging.debug("Calling Operator Shape " + getArgs(a))
	# If it's a tensor, call shape on it directly
	# Otherwise, use numpy to get its shape
	if isinstance(a, tf.Tensor):
		res = a.shape()
	else:
		# res = tf.convert_to_tensor( np.shape(a) , dtype = np.int32 )
		res = np.int32( np.shape(a) )
	# res should be either a scalar or tensor here
	res = condPerturb(Ops.SHAPE,res)
	if logReturn: logging.debug("\tReturning from Shape " + str(res) )
	return res

def injectFaultSize(a):
	"Function to call injectFault on Size nodes"
	logging.debug("Calling Operator Size " + getArgs(a))
	res = a.size()
	res = condPerturb(Ops.SIZE, res)
	if logReturn: logging.debug("\tReturning from Size " + str(res) )
	return res

def injectFaultFill(a, b):
	"Function to call injectFault on Shape nodes"
	logging.debug("Calling Operator Fill " + getArgs(a, b))
	res = np.full(a, b)
	res = condPerturb(Ops.FILL, res)
	if logReturn: logging.debug("\tReturning from Fill" + str(res) )
	return res

def injectFaultFloorMod(a, b):
	"Function to call injectFault on FloorMod nodes"
	logging.debug("Calling Operator FloorMod " + getArgs(a, b)) 
	# FIXME: Need to check if mod is the equivalent of floorMod in NumPy
	res = np.mod(a, b)
	res = condPerturb(Ops.FLOORMOD, res)
	if logReturn: logging.debug("\tReturning from FloorMod " + str(res) )
	return res

def injectFaultRange(start, stop, step, dtype = None):
	"Function to call injectFault on Range nodes"
	logging.debug("Calling Operator Range " + getArgs(start, stop, step))
	res = np.int32(np.arange(start, stop, step, dtype))
	res = condPerturb(Ops.RANGE, res)
	if logReturn: logging.debug("\tReturning from Range " + str(res) )
	return res	

def injectFaultRank(a):
	"Function to call injectFault on Rank nodes"
	logging.debug("Calling Operator Rank " + getArgs(a))
	res = np.int32( np.ndim(a) )
	res = condPerturb(Ops.RANK, res)
	if logReturn: logging.debug("\tReturning from Rank " + str(res) )
	return res	

def injectFaultSum(a, b):
	"Function to call injectFault on Sum nodes"
	logging.debug("Calling Operator Sum " + getArgs(a, b))
	# Check if b is an integer scalar array
	# and if so, pass it to np.sum
	# Otherwise, ignore it (FIXME: is this the correct behavior ?)
	if np.isscalar(b):
		res = np.sum(a, b)
	else:
		res = np.sum(a)
	res = condPerturb(Ops.SUM, res)
	if logReturn: logging.debug("\tReturning from Sum " + str(res) )
	return res

def injectFaultReshape(a, b):
	"Function to call injectFault on Reshape"
	logging.debug("Calling Operator Reshape " + getArgs(a, b))

	try:
		res = np.reshape(a, b)
	except:
		res = a
		print(" value dim was changed, this would lead to error. So we keep the same value dim")
	res = condPerturb(Ops.RESHAPE, res)
 
		
 
	if logReturn: logging.debug("\tReturning from Reshape " + str(res) )
	return res

def injectFaultMatMul(a, b):
	"Function to call injectFault on matrix multiplication"
	logging.debug("Calling Operator MatMul " + getArgs(a, b))


	matmul = tf.matmul(a,b)
	with tf.Session() as sess:
		res = matmul.eval()
#	res = np.matmul(a, b)
	res = condPerturb(Ops.MATMUL, res)



	if logReturn: logging.debug("\tReturning from MatMul " + str(res) )
	return res

def injectFaultArgMax(a, b):
	"Function to call injectFault on ArgMax"
	logging.debug("Calling Operator ArgMax " + getArgs(a, b)) 	
	res = np.argmax(a, b)
	res = condPerturb(Ops.ARGMAX, res)
	if logReturn: logging.debug("\tReturning from ArgMax " + str(res) )
	return res

def injectFaultArgMin(a, b):
	"Function to call injectFault on ArgMin"
	logging.debug("Calling Operator ArgMin " + getArgs(a, b))
	res = np.argmin(a, b)
	res = condPerturb(Ops.ARGMIN, res)
	if logReturn: logging.debug("\tReturning from ArgMin " + str(res) )
	return res

def injectFaultEqual(a, b):
	"Function to call injectFault on equal"
	logging.debug("Calling Operator Equal " + getArgs(a, b)) 
	res = np.equal(a, b)
	res = condPerturb(Ops.EQUAL, res)
	if logReturn: logging.debug("\tReturning from Equal " + str(res) )
	return res

def injectFaultNotEqual(a, b):
	"Function to call injectFault on not equal"
	logging.debug("Calling Operator Not Equal " + getArgs(a, b))
	res = np.not_equal(a, b)
	res = condPerturb(Ops.NOT_EQUAL, res)
	if logReturn: logging.debug("\tReturning from Not Equal " + str(res) )
	return res

def injectFaultLessEqual(a, b):
	"Function to call injectFault on less equal"
	logging.debug("Calling Operator Less Equal " + getArgs(a, b))
	res = np.less_equal(a, b)
	res = condPerturb(Ops.LESS_EQUAL, res)
	if logReturn: logging.debug("\tReturning from Less Equal " + str(res) )
	return res

def injectFaultMean(a, b):
	"Function to call injectFault on mean"
	logging.debug("Calling Operator mean " + getArgs(a, b))
	# FIXME: This only works if we call np.mean on b[0]. Need to figure out why.



#	res = np.mean(a, b[0])
	sess = tf.Session()
	op = tf.reduce_mean(a, b)
	res = sess.run(op)


	res = condPerturb(Ops.MEAN, res)
	if logReturn: logging.debug("\tReturning from Mean " + str(res) )
	return res

def injectFaultCountNonZero(a):
	"Function to call injectFault on countNonZero"
	logging.debug("Calling Operator CountNonZero " + getArgs(a)) 
	res = np.count_nonzero(a)
	res = condPerturb(Ops.COUNT_NONZERO, res)
	if logReturn: logging.debug("\tReturning on CountNonZero " + str(res) )
	return res

def injectFaultConv2D(a, b, strides, padding):
	"Function to call injectFault on Conv2D"
	logging.debug("Calling Operator conv2D " + getArgs(a, b))
	# FIXME: Make this work with any type, not just float32 
#	res = np.float32( conv2d.conv2d(a, b) ) 
	conv = tf.nn.conv2d(a , b, strides=strides.tolist(), padding=padding)
 
	with tf.Session() as sess:
		res = conv.eval()
	
	res = condPerturb(Ops.CONV2D, res)
	if logReturn: logging.debug("\tReturning from Conv2D " + str(res) )
	return res

def injectFaultRelu(a):
	"Function to call injectFault on RelU"
	logging.debug("Calling Operator RelU " + getArgs(a))

	
	relu = tf.nn.relu(a)
	with tf.Session() as sess:
		res = relu.eval()
	
	res = condPerturb(Ops.RELU, res)


	"=============================================="
	"perform value truncation for value restriction"
	"comment this block if you want to remove Ranger"
	global reluNum
	upperBound = [958, 2018, 1587, 2470, 1892, 1228, 2003, 3024, 1736, 3031, 2687, 1408, 1903, 3329, 1661, 2285, 2749, 1041, 1884, 2126,  
			823, 1772, 2306, 748, 1086, 700]
	### Change ReLu ACT 
	res = np.minimum(res, upperBound[reluNum], res)
	res = np.maximum(res, 0, res)

	reluNum = ( reluNum+1  )%len(upperBound)
	"=============================================="


	if logReturn: logging.debug("\tReturning from RelU " + str(res) )
	return res


def injectFaultELU(a):
	"Function to call injectFault on RelU"
	logging.debug("Calling Operator RelU " + getArgs(a))

	relu = tf.nn.elu(a)
	with tf.Session() as sess:
		res = relu.eval()
 
#	res = np.maximum(a, 0, a)
	res = condPerturb(Ops.ELU, res)
	if logReturn: logging.debug("\tReturning from RelU " + str(res) )
	return res








def injectFaultMaxPool(a, ksize, strides, padding):
#def injectFaultMaxPool(a, b, c, d):
	"Function to call injectFault on MaxPool"
#	logging.debug("Calling Operator MaxPool " + getArgs(a, b))
	# FIXME: This seems to work, but not sure if it always does 
	#	Default parameter is (1,2) for the axis value 

#	res = np.amax(a)
	maxpool = tf.nn.max_pool(a, ksize=ksize.tolist(), strides=strides.tolist(), padding=padding)
	with tf.Session() as sess:
		res = maxpool.eval()	

	res = condPerturb(Ops.MAXPOOL, res)

	 

	"=============================================="
	"perform value truncation for value restriction"
	"comment this block if you want to remove Ranger"
	global reluNum

	upperBound = [958, 2018, 1587, 2470, 1892, 1228, 2003, 3024, 1736, 3031, 2687, 1408, 1903, 3329, 1661, 2285, 2749, 1041, 1884, 2126,
						823, 1772, 2306, 748, 1086, 700]

	# bound of first MPool follows the first Relu
	# bound of following MPool follow ConCatV2, which takes the largest from the prior 2 ReLu
	bound = 0.
	if(reluNum <= 2):
		bound = upperBound[ reluNum -1 ]
	else:
		bound = max( upperBound[ reluNum -1 ], upperBound[ reluNum - 2 ]) 

	### Change ReLu ACT
	res = np.minimum(res, bound, res)     
	res = np.maximum(res, 0, res)
	"=============================================="



	if logReturn: logging.debug("\tReturningfrom MaxPool  " + str(res) )
	return res


def injectFaultAvgPool(a, ksize, strides, padding): 
	"Function to call injectFault on AvgPool" 

	logging.debug("Calling Operator AvgPool " + getArgs(a))
#	res = np.amax(a)
	avgpool = tf.nn.avg_pool(a, ksize=ksize.tolist(), strides=strides.tolist(), padding=padding)
	with tf.Session() as sess:
		res = avgpool.eval()	

	res = condPerturb(Ops.AVGPOOL, res)
 

	if logReturn: logging.debug("\tReturningfrom AvgPool  " + str(res) )
	return res






def injectFaultUnpack(a , b):
	"Function to call injectFault on unpack"
	logging.debug("Calling Operator Unpack " + getArgs(a))
	# This operation is deprecated in TF 1.0 and above


#	if logReturn: logging.debug("\tReturning from Unpack " + str(res) )
#	raise NotImplementedError("AssignAdd")
 
	pack = tf.unstack(a ,b)	# pack is changed into tf.stack
	with tf.Session() as sess:
		res = pack.eval()

#	res = np.array_split(a, a.shape[1]) 
	# FIXME: Can't inject faults into unpack as it's not a tensor or scalar
	# res = condPerturb(Ops.UNPACK, res)
	if logReturn: logging.debug("\tReturning from Unpack " + str(res) )
#	return res

def injectFaultUnstack(a):
	"Function to call injectFault on unstack"
	# This is the same as Unpack in newer versions of TF
	return injectFaultUnpack(a)

def injectFaultStridedSlice(a, b, c, d):
	"Function to call injectFault on StridedSlice"
	logging.debug("Calling Operator StridedSlice " + getArgs(a))
	# FIXME: Implement this functionality
	res = a
	res = condPerturb(Ops.STRIDEDSLICE, res)
	if logReturn: logging.debug("\tReturning from StridedSlice " + str(res) )
	return res
		
def injectFaultExpandDims(a, b):
	"Function to call injectFault on ExpandDims"
	logging.debug("Calling Operator ExpandDims " + getArgs(a, b))
	res = np.expand_dims(a, b)
	res = condPerturb(Ops.EXPANDDIMS, res)
	if logReturn: logging.debug("\tReturning from ExpandDims " + str(res) )
	return res

def injectFaultPack(a, b = None):
	"Function to call injectFault on Pack"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator Pack" + getArgs(a))
	# res = np.stack(a, b)
	# FIXME: This throws an exception, so we dummied it out
 
	a = np.atleast_1d(a)
	if(b!=None):
		a = np.append(a,b)
 

	pack = tf.stack(a)	# pack is changed into tf.stack
	with tf.Session() as sess:
		res = pack.eval()

#	res = a
	res = condPerturb(Ops.PACK, res)
	if logReturn: logging.debug("\tReturning " + str(res) )
	return res

def injectFaultConcatV2(a, b, c, d=None):
	"Function to call injectFault on ConcatV2"
	logging.debug("Calling Operator ConcatV2" + getArgs(a.shape, b.shape, c.shape))
	# FIXME: According to tf doc, this should only take 2 arguments
	# 	but somehow it is complaining that it requires 3 arguments
 

	sess = tf.Session()

	op = tf.concat([a,b] , c)
		
	res = sess.run(op) 

	res = condPerturb(Ops.CONCATV2, res)



	"=============================================="
	"perform value truncation for value restriction"
	"comment this block if you want to remove Ranger"

	global reluNum
	upperBound = [958, 2018, 1587, 2470, 1892, 1228, 2003, 3024, 1736, 3031, 2687, 1408, 1903, 3329, 1661, 2285, 2749, 1041, 1884, 2126,
					823, 1772, 2306, 748, 1086, 700]


	# bound ConCatV2, which takes the largest from the prior 2 ReLu, as it concats from two ReLu Ops
	bound = max( upperBound[ reluNum -1 ], upperBound[ reluNum - 2 ])

	### Change ReLu ACT
	res = np.minimum(res, bound, res)
	res = np.maximum(res, 0, res)

	"=============================================="

	if logReturn: logging.debug("\tReturning from CONCATV2 " + str(res) )


	
	return res  


def injectFaultSoftmax(a):
	"Function to call injectFault on Softmax"
	logging.debug("Calling Operator Softmax " + getArgs(a))
	resOp = tf.nn.softmax(a)
	with tf.Session() as sess:
		res = resOp.eval()
#	res1 = softmax(a)		# Use the implementation from the sklearn library

	res = condPerturb(Ops.SOFTMAX, res)
	if logReturn: logging.debug("\tReturning from Softmax " + str(res) )
	return res

def injectFaultMaximum(a, b):
	"Function to call injectFault on Maximum"
	logging.debug("Calling Operator Maximum " + getArgs(a, b)) 
	res = np.maximum(a, b)
	res = condPerturb(Ops.MAXIMUM, res)
	if logReturn: logging.debug("\tReturning from Maximum " + str(res) )
	return res

def injectFaultMinimum(a, b):
	"Function to call injectFault on Maximum"
	logging.debug("Calling Operator Minimum " + getArgs(a, b)) 
	res = np.minimum(a, b)
	res = condPerturb(Ops.MINIMUM, res)
	if logReturn: logging.debug("\tReturning from Minimum " + str(res) )
	return res


from tensorflow.python.ops import control_flow_ops 
def injectFaultSwitch(a, cond):
	"Function to call injectFault on Switch"
	logging.debug("Calling Operator Switch " + getArgs(a, cond))
	# FIXME: Actually implement the Switch operation
	# 	Only there's no TensorFlow documentation for it !!!
	# res = np.select(a, b)
 
	sess = tf.Session()
	op = control_flow_ops.switch(data=a, pred=cond)

	try:
		res = sess.run(op)	
	except:
		return a, a 

#	res = a, a
	# res = condPerturb(Ops.SWITCH, res)
	if logReturn: logging.debug("\tReturning from Switch " + str(res) )
	return res

def injectFaultGreater(a, b):
	"Function to call injectFault on Greater"
	logging.debug("Calling Operator Greater " + getArgs(a, b))
	res = np.greater(a, b)
	res = condPerturb(Ops.GREATER, res)
	if logReturn: logging.debug("\tReturning from Greater " + str(res) )
	return res

def injectFaultNeg(a):
	"Function to call injectFault on negative"
	logging.debug("Calling Operator Neg " + getArgs(a))
	res = np.negative(a)
	res = condPerturb(Ops.NEGATIVE, res)
	if logReturn: logging.debug("\tReturning from Neg " + str(res) )
	return res

def injectFaultPow(a, b):
	"Function to call injectFault on pow"
	logging.debug("Calling Operator Pow " + getArgs(a, b))
	res = np.power(a, b)
	res = condPerturb(Ops.POWER, res)
	if logReturn: logging.debug("\tReturning from Pow " + str(res) )
	return res

def injectFaultAbs(a):
	"Function to call injectFault on absolute"
	logging.debug("Calling Operator Abs " + getArgs(a))
	res = np.absolute(a)
	res = condPerturb(Ops.ABSOLUTE, res)
	if logReturn: logging.debug("\tReturning from Abs " + str(res) )
	return res

def injectFaultRsqrt(a):
	"Function to call injectFault on Rsqrt"
	logging.debug("Calling Operator Rsqrt " + getArgs(a))
	res = np.reciprocal( np.sqrt(a) )
	res = condPerturb(Ops.RSQRT, res)
	if logReturn: logging.debug("\tReturning from Rsqrt " + str(res) )
	return res

def injectFaultNN(a, b, c):
	"Function to call injectFault on Nearest Neighbors"
	# FIXME: According to the TF docs, this operation doesn't exist !
	#	Not sure what the third parameter is supposed to be.
	logging.debug("Calling Operator Nearest Neighbors " + getArgs(a, b, c))
	res = KNeighborsClassifier(a)
	if logReturn: logging.debug("\tReturning from Nearest Neighbors " + str(res) )
	return res

def injectFaultLog(a):
	"Function to call injectFault on Log"
	logging.debug("Calling Operator Log " + getArgs(a))

	sess = tf.Session()
	op = tf.math.log(a)
	res=sess.run(op)
# 	res = np.reciprocal( np.log(a) )

	res = condPerturb(Ops.LOG, res)
	if logReturn: logging.debug("\tReturning from Log " + str(res) )
	return res

def injectFaultRealDiv(a, b):
	"Function to call injectFault on RealDiv"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator Log " + getArgs(a, b))
	res = np.divide( a, b )
	res = condPerturb(Ops.REALDIV, res)
	if logReturn: logging.debug("\tReturning from RealDiv " + str(res) )
	return res

def injectFaultBiasAdd(a, b):
	"Function to call injectFault on BiasAdd"
	logging.debug("Calling Operator BiasAdd " + getArgs(a, b))
	res = a + b 
	res = condPerturb(Ops.BIASADD, res)
	if logReturn: logging.debug("\tReturning from BiasAdd " + str(res) )
	return res

def injectFaultSigmoid(a):
	"Function to call injectFault on Sigmoid"
	logging.debug("Calling Operator Sigmoid " + getArgs(a))
	res = np.reciprocal( 1 + np.exp(-a) ) 
	res = condPerturb(Ops.SIGMOID, res)
	if logReturn: logging.debug("\tReturning from Sigmoid " + str(res) )
	return res

def injectFaultTanh(a):
	"Function to call injectFault on Tanh"
	logging.debug("Calling Operator Tanh " + getArgs(a))
	res = np.tanh( a ) 
	res = condPerturb(Ops.TANH, res)
 

	if logReturn: logging.debug("\tReturning from Tanh " + str(res) )
	return res

# End of implemented operators
def injectFaultRandomUniform(a):
	"Function to call injectFault on Random Uniform"
	# FIXME: Implement this functionality
	resOp = tf.random_uniform(a)
	with tf.Session() as sess:
		res = resOp.eval()

	res = condPerturb(Ops.RANDOMUNIFORM, res) 

	if logReturn: logging.debug("\tReturning from RANDOMUNIFORM " + str(res) )


	return res
#	logging.debug("Calling Operator RandomUniform")
#	raise NotImplementedError("RandomUniform")

##### None of the functions below have been implemented yet as they're not used #####
#### If you implement any of them, please move them above the line              ####
#####          Otherwise, they will all raise NotImplementedError(OpName)       ####3
 
def injectFaultDynamicStitch(inputs):
	"Function to call injectFault on Dynamic stitch"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator Dynamic stitch ") 
	raise NotImplementedError("DynamicStitch")	

def injectFaultFloorDiv(inputs):
	"Function to call injectFault on FloorDiv"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator FloorDiv ") 
	raise NotImplementedError("FloorDiv")	

def injectFaultTile(inputs):
	"Function to call injectFault on Tile"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator Tile")
	raise NotImplementedError("Tile")	

def injectFaultConcatOffset(inputs):
	"Function to call injectFault on ConcatOffset"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator ConcatOffset")
	raise NotImplementedError("ConcatOffset")	

def injectFaultSplit(num_split, inputs):
	"Function to call injectFault on Split"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator Split")
#	raise NotImplementedError("Split")	
	
	sess = tf.Session() 
  
	op = tf.split(inputs, num_or_size_splits=num_split, axis=-1)
	res = sess.run(op)

	return res


def injectFaultSoftmaxCEWL(inputs):
	"Function to call injectFault on Softmax CEWL"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator SoftmaxCEWL")
	raise NotImplementedError("SoftmaCEWL")	


def injectFaultSlice(inputs, b, c):
	"Function to call injectFault on Slice"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator Slice")
 

	op = tf.slice(inputs, b,c)
	sess= tf.Session()
	res = sess.run(op)
	return res


	#raise NotImplementedError("Slice")	

def injectFaultBroadcastGA(inputs):
	"Function to call injectFault on Broadcast gradient args"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator BroadcastGA")
	raise NotImplementedError("BroadcastGA")	

def injectFaultTruncatedNormal(a):
	"Function to call injectFault on TruncatedNormal"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator TruncatedNormal") # + str(a))

	op = tf.truncated_normal(a)
	sess= tf.Session()
	res = sess.run(op)
	return res	


#	raise NotImplementedError("TruncatedNormal")

def injectFaultRandomUniformInt(a):
	"Function to call injectFault on Random Uniform Int"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator RandomUniformInt")
	raise NotImplementedError("RandomUniformInt")

def injectFaultRandomStandardNormal(a):
	"Function to call injectFault on Random Standard Normal"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator RandomStandardNormal")
	raise NotImplementedError("RandomStandardNormal")

def injectFaultFloor(a):
	"Function to call injectFault on Floor"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator Floor")

	resOp = tf.floor(a)
	sess = tf.Session()
	res = sess.run(resOp)
	res = condPerturb(Ops.FLOOR, res)
	return res
#	raise NotImplementedError("Floor")

def injectFaultRefSwitch(a):
	"Function to call injectFault on RefSwitch"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator RefSwitch")
	raise NotImplementedError("RefSwitch")

def injectFaultProd(a,b):
	"Function to call injectFault on Prod"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator Prod")
	 

	op = tf.keras.backend.prod(a,b[0])
	sess= tf.Session()
	res = sess.run(op)
	return res
#	raise NotImplementedError("Prod")

def injectFaultSqueeze(a):
	"Function to call injectFault on Squeee"
	# FIXME: Implement this functionality
	resOp = tf.squeeze(a)
	sess = tf.Session()
	res = sess.run(resOp)
	return res

#	logging.debug("Calling Operator Squeeze")
#	raise NotImplementedError("Squeeze")

def injectFaultUnique(a):
	"Function to call injectFault on Unique"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator Unique")
	raise NotImplementedError("Unique")

def injectFaultReciprocal(a):
	"Function to call injectFault on Reciprocal"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator Reciprocal")
	raise NotImplementedError("Reciprocal")

def injectFaultScatterAdd(a):
	"Function to call injectFault on ScatterAdd"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator ScatterAdd")
	raise NotImplementedError("ScatterAdd")

def injectFaultReluGrad(a):
	"Function to call injectFault on ReluGrad"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator ReluGrad")
	raise NotImplementedError("ReluGrad")

def injectFaultMaxPoolGrad(a):
	"Function to call injectFault on MaxPoolGrad"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator MaxPoolGrad")
	raise NotImplementedError("MaxPoolGrad")

def injectFaultTanhGrad(a):
	"Function to call injectFault on TanhGrad"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator TanhGrad")
	raise NotImplementedError("TanhGrad")

def injectFaultSigmoidGrad(a):
	"Function to call injectFault on SigmoidGrad"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator SigmoidGrad")
	raise NotImplementedError("SigmoidGrad")

def injectFaultBiasAddGrad(a):
	"Function to call injectFault on BiasAddGrad"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator BiasAddGrad")
	raise NotImplementedError("BiasAddGrad")

def injectFaultShapeN(inputs):
	"Function to call injectFault on ShapeN"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator ShapeN")
	raise NotImplementedError("ShapeN")

def injectFaultAddN(inputs):
	"Function to call injectFault on AddN"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator AddN")
	raise NotImplementedError("AddN")

def injectFaultConv2DBackprop(inputs):
	"Function to call injectFault on Conv2DBackprop"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator Conv2DBackProp")
	raise NotImplementedError("Conv2DBackProp")

def injectFaultApplyAdam(inputs):
	"Function to call injectFault on ApplyAdam"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator ApplyAdam")
	raise NotImplementedError("ApplyAdam")
	
def injectFaultSelect(inputs, b, c):
	"Function to call injectFault on Select"
	# FIXME: Implement this functionality
 

	# select is deprecated, use tf.where 
	op = tf.where(inputs, b, c)
	sess= tf.Session()
	res = sess.run(op)
	return res

	logging.debug("Calling Operator Select")
	#raise NotImplementedError("Select")


from tensorflow.python.ops import control_flow_ops 
def injectFaultMerge(inputs, b):
	"Function to call injectFault on Merge"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator Merge")

	

	op = control_flow_ops.merge(inputs=[inputs,b])
	sess= tf.Session()
	res = sess.run(op)
  

	return res

#	raise NotImplementedError("Merge")

def injectFaultTranspose(inputs):
	"Function to call injectFault on Transpose"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator Transpose")
	raise NotImplementedError("Transpose")

def injectFaultTranspose(inputs):
	"Function to call injectFault on Transpose"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator Transpose")
	raise NotImplementedError("Transpose")

def injectFaultGather(inputs):
	"Function to call injectFault on Gather"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator Gather")
	raise NotImplementedError("Gather")

def injectFaultUnsortedSegmentSum(inputs):
	"Function to call injectFault on UnsortedSegmentSum"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator UnsortedSegmentSum")
	raise NotImplementedError("UnsortedSegmentSum")

def injectFaultInvertPermutation(inputs):
	"Function to call injectFault on InvertPermutation"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator InvertPermuation")
	raise NotImplementedError("InvertPermutation")
	
def injectFaultApplyGradientDescent(inputs):
	"Function to call injectFault on applying gradient descent"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator ApplyGradientDescent")
	raise NotImplementedError("ApplyGradientDescent")

def injectFaultZerosLike(inputs):
	"Function to call injectFault on ZerosLike"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator ZerosLike")


	op = tf.zeros_like(inputs)
	sess= tf.Session()
	res = sess.run(op)
	return res


	#raise NotImplementedError("ZerosLike")
	
def injectFaultPreventGradient(inputs):
	"Function to call injectFault on PreventGradient"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator PreventGradient")
	raise NotImplementedError("PreventGradient")
	
def injectFaultSSSmcEWL(inputs):
	"Function to call injectFault on SoftSparseMax.."
	# FIXME: Implement this functionality
	logging.debug("Calling Operator SoftSparseMax")
	raise NotImplementedError("SoftSparseMax")
	
def injectFaultAll(a):
	"Function to call injectFault on All operation"
	# FIXME: Implement this functionality
	# Not clear what this does - TF doc is silent about this
	logging.debug("Calling Operator All")
	raise NotImplementedError("All")
	
def injectFaultAssert(a):
	"Function to call injectFault on Assert operation"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator Assert")
	raise NotImplementedError("Assert")
	
def injectFaultLess(a):
	"Function to call injectFault on Less operation"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator Less")
	raise NotImplementedError("Less")

def injectFaultFSRHOP(a):
	"Function to call Inject fault on FertileResource Op"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator FSRHOP")
	raise NotImplementedError("FSRHOP")

def injectFaultL2Loss(a):
	"Function to call Inject fault on L2Loss operation"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator L2Loss")
	raise NotImplementedError("L2Loss")

def injectFaultApplyMomentum(a):
	"Function to call Inject fault on ApplyMomentum operation"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator ApplyMomentum")
	raise NotImplementedError("ApplyMomentum")

def injectFaultAssignAdd(a):
	"Function to call Inject fault on AssignAdd operation"
	# FIXME: Implement this functionality
	logging.debug("Calling Operator AssignAdd")
	raise NotImplementedError("AssignAdd")
	
##### End of unimplemented functions ###################
	
# This is the generic "Catch-all" function - it should be last
# It takes a variable number of arguments in the inputs array
def injectFaultGeneric(*inputs):
	"Generic Function to call fault injection on each input and zero it out"
	outputs = []
	logging.debug("Calling generic fiFunc on " + str(inputs))
	# Perturb the input and add it to the outpus
	# FIXME: Should we NOT actually do the operation as well ??

 
	for each in inputs:
		outputs.append( (each) )
	if logReturn: logging.debug("\tReturning " + str(outputs))
	return outputs



def injectLRN(a, bias, alpha, beta):

	resOp = tf.nn.lrn( a , 4, bias=bias, alpha=alpha, beta=beta)
	with tf.Session() as sess:
		res = resOp.eval() 
	res = condPerturb(Ops.LRN, res)
	return res

#	raise NotImplementedError("LRN")


def injectFusedBatchNorm(a, gamma, beta, moving_mean, moving_variance):
  
	res = a
	res = condPerturb(Ops.FusedBatchNorm, res)
	return res

def injectFaultConcat(a, b, c, d): 
 
 

	sess =tf.Session()
	op = tf.concat([d,c,b], a)
	res = sess.run(op)

	return res

def injectFaultSparseToDense(a, b, c, d):
	sess =tf.Session()
 
	op = tf.sparse_to_dense(a, b, bool(c), bool(d))
	res = sess.run(op)

	return res

def injectFaultDiv(a, b):
 

	op = tf.div(a,b)
	sess= tf.Session()
	res = sess.run(op)


	return res


"Changing the res doesn't affect the final output???"
def injectFaultSquaredDifference(a,b):
	op = tf.squared_difference(a,b)
	sess= tf.Session()
	res = sess.run(op)

	return res




# End of injectFault operations

# The functions in this table are the ones defined above
# FIXME: These are fairly repetitive, so perhaps generate them automatically
#	Also, maybe these should be sorted alphabetically - this is getting quite big
opTable = { 
			"NoOp" : injectFaultNoop,	# First operation 
			"Add": injectFaultAdd,
			"Sub": injectFaultSub,
			"Mul": injectFaultMul,
			"Square" : injectFaultSquare,
			"Assign" : injectFaultAssign,	
			"Identity": injectFaultIdentity,
			"Range": injectFaultRange,
			"Rank": injectFaultRank,
			"Sum" : injectFaultSum,
			"Shape": injectFaultShape,
			"Fill": injectFaultFill,
			"Size": injectFaultSize,
			"FloorMod" : injectFaultFloorMod,
			"DynamicStitch" : injectFaultDynamicStitch,
			"Maximum" : injectFaultMaximum,
			"Max" : injectFaultMaximum,	# FIXME: Not sure if Max is a synonymn of Maximum or a new operation
			"Minimum" : injectFaultMinimum,
			"Min" : injectFaultMinimum,	# FIXME: Not sure if Min is a synonymn of Minimum or a new operation
			"FloorDiv" : injectFaultFloorDiv,
			"Reshape" : injectFaultReshape,
			"Tile" : injectFaultTile,
			"ConcatV2" : injectFaultConcatV2,
			"ConcatOffset" : injectFaultConcatOffset,
			"BiasAdd" : injectFaultBiasAdd,



			"Split" : injectFaultSplit, 



			"Sigmoid" : injectFaultSigmoid,
			"Tanh" : injectFaultTanh,
			"Softmax" : injectFaultSoftmax,
			"SoftmaxCrossEntropyWithLogits" : injectFaultSoftmaxCEWL,
			"Pack" : injectFaultPack,
			"Slice" : injectFaultSlice,
			"StridedSlice" : injectFaultStridedSlice,
			"BroadcastGradientArgs" : injectFaultBroadcastGA,
			"Neg" : injectFaultNeg,
			"Pow" : injectFaultPow,
			"Abs" : injectFaultAbs,
			"Unpack": injectFaultUnpack,
			"Unstack": injectFaultUnstack,
			"MatMul" : injectFaultMatMul,
			"ArgMax" : injectFaultArgMax,
			"ArgMin" : injectFaultArgMin,
			"Equal" : injectFaultEqual,
			"NotEqual" : injectFaultNotEqual,
			"LessEqual" : injectFaultLessEqual,
			"TruncatedNormal" : injectFaultTruncatedNormal,
			"Conv2D" : injectFaultConv2D,
			"Relu" : injectFaultRelu, 
			"MaxPool" : injectFaultMaxPool, 
			"RandomUniform" : injectFaultRandomUniform,
			"RandomUniformInt" : injectFaultRandomUniformInt,
			"RandomStandardNormal" : injectFaultRandomStandardNormal,
			"Floor" : injectFaultFloor,
			"Rsqrt" : injectFaultRsqrt,
			"Log" : injectFaultLog,
			"RefSwitch" : injectFaultRefSwitch,
			"NearestNeighbors" : injectFaultNN, 
			"Prod" : injectFaultProd,
			"Squeeze" : injectFaultSqueeze,
			"Unique" : injectFaultUnique,
			"Reciprocal" : injectFaultReciprocal,
			"ScatterAdd" : injectFaultScatterAdd,
			"ReluGrad" : injectFaultReluGrad,
			"MaxPoolGrad" : injectFaultMaxPoolGrad,
			"TanhGrad" : injectFaultTanhGrad,
			"SigmoidGrad" : injectFaultSigmoidGrad,
			"BiasAddGrad" : injectFaultBiasAddGrad,
			"ShapeN" : injectFaultShapeN,
			"AddN" : injectFaultAddN,
			"Conv2DBackpropInput" : injectFaultConv2DBackprop,
			"Conv2DBackpropFilter" : injectFaultConv2DBackprop,
			"ApplyAdam" : injectFaultApplyAdam,
			"Select" : injectFaultSelect,
			"Switch" : injectFaultSwitch,
			"Merge" : injectFaultMerge,
			"Transpose" : injectFaultTranspose,
			"Gather" : injectFaultGather,
			"UnsortedSegmentSum" : injectFaultUnsortedSegmentSum,
			"InvertPermutation" : injectFaultInvertPermutation,
			# Casts are treated differently, so don't add them to this table ! See createInjectFaultCast
			# "Cast" : injectFaultCast,		
			"Mean" : injectFaultMean,
			"Count_nonzero" : injectFaultCountNonZero,
			"RealDiv" : injectFaultRealDiv,
			"Greater" : injectFaultGreater,
			"ApplyGradientDescent" : injectFaultApplyGradientDescent,
			"ZerosLike" : injectFaultZerosLike,
			"PreventGradient" : injectFaultPreventGradient,
			"ExpandDims" : injectFaultExpandDims,
			"SparseSoftmaxCrossEntropyWithLogits" : injectFaultSSSmcEWL,
			"All" : injectFaultAll,
			"Assert" : injectFaultAssert,
			"Less" : injectFaultLess,
			"FertileStatsResourceHandleOp" : injectFaultFSRHOP,
			"L2Loss" : injectFaultL2Loss,
			"ApplyMomentum" : injectFaultApplyMomentum,
			"AssignAdd" : injectFaultAssignAdd,
			"Unknown": injectFaultGeneric,		# Last operation
			# "Unknown": None			# For debugging purposes




			"LRN" : injectLRN , 
			"FusedBatchNorm" : injectFusedBatchNorm,
			"Elu" : injectFaultELU,

			"Concat" : injectFaultConcat,
			"SparseToDense" : injectFaultSparseToDense,
			"Div" : injectFaultDiv,
			"SquaredDifference" : injectFaultSquaredDifference,
			"AvgPool" : injectFaultAvgPool

		}	

