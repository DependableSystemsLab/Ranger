# These are the list of fault injection functions for different types of faults
# NOTE: There are separate versions of the scalar and tensor values for portability
# If you add a new fault type, please create both the scalar and tensor functions 

import numpy as np

# Currently, we support 8 types of faults {None, Rand, Zero, Rand-element, bitFlip-element, bitFlip-tensor, binaryBitFlip, sequentialBitFlip} - See fiConfig.py

def randomScalar( dtype, max = 1.0 ):
    "Return a random value of type dtype from [0, max]"
    return dtype.type( np.random.random() * max )

def randomTensor( dtype, tensor):
    "Random replacement of a tensor value with another one"
    # The tensor.shape is a tuple, while rand needs linear arguments
    # So we need to unpack the tensor.shape tuples as arguments using *  
    res = np.random.rand( *tensor.shape ) 
    return dtype.type( res )

def zeroScalar(dtype, val):
    "Return a scalar 0 of type dtype"
    # val is a dummy parameter for compatibility with randomScalar
    return dtype.type( 0.0 )

def zeroTensor(dtype, tensor):
    "Take a tensor and zero it"
    res = np.zeros( tensor.shape ) 
    return dtype.type( res )

def noScalar(dtype, val):
    "Dummy injection function that does nothing"
    return val

def noTensor(dtype, tensor):
    "Dummy injection function that does nothing"
    return tensor

def randomElementScalar( dtype, max = 1.0):
    "Return a random value of type dtype from [0, max]"
    return dtype.type( np.random.random() * max )

def randomElementTensor ( dtype, val):
    "Random replacement of an element in a tensor with another one"
    "Only one element in a tensor will be changed while the other remains unchanged" 
    dim = val.ndim 
    
    if(dim==1):
        index = np.random.randint(low=0 , high=(val.shape[0]))
        val[index] = np.random.random() 
    elif(dim==2):
        index = [np.random.randint(low=0 , high=(val.shape[0])) , np.random.randint(low=0 , high=(val.shape[1]))]
        val[ index[0] ][ index[1] ] = np.random.random()

    return dtype.type( val )



def float2bin(number, decLength = 2, bitWidth = 16): 
    "convert float data into binary expression"
    # we consider fixed-point data type, 32 bit: 1 sign bit, 21 integer and 10 mantissa

    # split integer and decimal part into seperate variables  
    integer, decimal = str(number).split(".") 
    # convert integer and decimal part into integer  
    integer = int(integer)  

    # truncate the data and return maximal range representation
    if( integer >= pow(2, bitWidth - decLength)- 0.25 ): 
        integer = pow(2, bitWidth - decLength)-0.25

    # Convert the integer part into binary form. 
    res = bin(integer)[2:] + "."        # strip the first binary label "0b"
 

    res = res.zfill(bitWidth - decLength + 1)
    
    def decimalConverter(decimal): 
        "E.g., it will return `x' as `0.x', for binary conversion"
        decimal = '0' + '.' + decimal 
        return float(decimal)

    # iterate times = length of binary decimal part
    for x in range(decLength): 
        # Multiply the decimal value by 2 and seperate the integer and decimal parts 
        # formating the digits so that it would not be expressed by scientific notation
        integer, decimal = format( (decimalConverter(decimal)) * 2, '.10f' ).split(".")    
        res += integer 
 

    return res 


def randomBitFlip(val):
    "Flip a random bit in the data to be injected" 

    # Split the integer part and decimal part in binary expression
    def getBinary(number, decWidth = 2, bitWidth = 16):
        # integer data type
        if(isinstance(number, int)):

            # truncate data 
            if(number >= pow(2, (bitWidth- decWidth) )-0.25  ):
                number = pow(2, (bitWidth- decWidth) ) - 0.25

            integer = bin(int(number)).lstrip("0b") 
            # 14 digits for integer
            integer = integer.zfill( (bitWidth- decWidth) )
            # integer has no mantissa
            dec = ''    
        # float point datatype                      
        else:
            binVal = float2bin(number)              
            # split data into integer and decimal part  
            integer, dec = binVal.split(".")    
        return integer, dec

    # we use a tag for the sign of negative val, and then consider all values as positive values
    # the sign bit will be tagged back when finishing bit flip
    negTag = 1
    if(str(val)[0]=="-"):
        negTag=-1

    if(isinstance(val, np.bool_)):  
        # boolean value
        return bool( (val+1)%2 )
    else:   
        # turn the val into positive val
        val = abs(float(val))
        integer, dec = getBinary(val)

    intLength = len(integer)
    decLength = len(dec)


    if( val >= pow(2, intLength)- 0.25 ):
            val = pow(2, intLength)-0.25


    # random index of the bit to flip  
    index = np.random.randint(low=0 , high = intLength + decLength)
 
    # flip the sign bit (optional)
#   if(index==-1):
#       return val*negTag*(-1)

    # bit to flip at the integer part
    if(index < intLength):      
        # bit flipped from 1 to 0, thus minusing the corresponding value
        if(integer[index] == '1'):  val -= pow(2 , (intLength - index - 1))  
        # bit flipped from 0 to 1, thus adding the corresponding value
        else:                       val += pow(2 , (intLength - index - 1))
    # bit to flip at the decimal part  
    else:                       
        index = index - intLength     
        # bit flipped from 1 to 0, thus minusing the corresponding value
        if(dec[index] == '1'):  val -= 2 ** (-index-1)
        # bit flipped from 0 to 1, thus adding the corresponding value
        else:                   val += 2 ** (-index-1) 

    return val*negTag

def bitElementScalar( dtype, val ):
    "Flip one bit of the scalar value"   
    return dtype.type( randomBitFlip(val) )

def bitElementTensor( dtype, val):
    "Flip ont bit of a random element in a tensor"
    # flatten the tensor into a vector and then restore the original shape in the end
    valShape = val.shape
    val = val.flatten()
    # select a random data item in the data space for injection
    index = np.random.randint(low=0, high=len(val))
    val[index] = randomBitFlip(val[index])  
    val = val.reshape(valShape)

    return dtype.type( val )

def bitScalar( dtype, val):
    "Flip one bit of the scalar value"
    return dtype.type( randomBitFlip(val) )

def bitTensor ( dtype, val):
    "Flip one bit of all elements in a tensor"
    # dimension of tensor value 
    dim = val.ndim      

    # the value is 1-dimension (i.e., vector)
    if(dim==1):         
        col = val.shape[0]
        for i in range(col):
            val[i] = randomBitFlip(val[i])

    # the value is 2-dimension (i.e., matrix)
    elif(dim==2):
        row = val.shape[0]
        col = val.shape[1]
        # flip one bit of each element in the tensor
        for i in range(row):
            for j in range(col): 
                val[i][j] = randomBitFlip(val[i][j]) 

    return dtype.type( val )




##########################################
def initBinaryInjection(isFirstTime=True):
    "init func is firstly intialized in the injectFault module"
    # flag for whether the last 2 FI cause SDC or not
    global sdcFromLastFI 
    # index of the data to be injected, monotonically starts from the first data 
    global indexOfInjectedData
    # we use binary search to decide to bit to be injected, requiring upper and lower bound for the binary search
    # upper bound for the binary search
    global upperIndexOfInjectedBit
    # lower bound for the binary search
    global lowerIndexOfInjectedBit
    # list of the index of the "1" of the current injected data
    global indexOfBit_1
    # list of the index of the "0" of the current injected data
    global indexOfBit_0
    # flag for whether still looking for the SDC at the 0 bits
    global isLookFor0
    # flag for whether still looking for the SDC at the 1 bits
    global isLookFor1
    # index of the previous 2 injected bits
    global indexOfLastInjectedBit 
    # global counter of sdc rate at the current op, the sdc rate now is based on the binary FI
    global sdcRate
    # count the SDC-causing bits at the current datapoint
    global sdcBitCount
    global indexOfLatesetSDC_nonSdcBit
    global isKeepDoingFI
    global fiTime 
    global fiDonePerData
    global isTransit


    # The first to do FI for the current Op, we inject the first data
    if(isFirstTime): 
        indexOfInjectedData = 0
        sdcRate = 0.
        isKeepDoingFI = True
        fiTime = 0 
        sdcBitCount = 0
        fiDonePerData = False
    # this is for initliazation purpose, the bound will be updated when doing FI
    upperIndexOfInjectedBit = -1
    lowerIndexOfInjectedBit = -1
    indexOfBit_0 = []
    indexOfBit_1 = []
    isLookFor0 = False
    isLookFor1 = False
    indexOfLatesetSDC_nonSdcBit = [-1, -1]
    sdcFromLastFI = -1
    indexOfLastInjectedBit = 0
    isTransit = False
    

def fiOnList(indexList, is0, injectedData, intLen ):
    "Perform binary FI on the list of '0' or '1' bits"
    # flag for whether the last FI cause SDC or not, used for indicating the next bit to be injected
    global sdcFromLastFI
    # index of the last two injected bits
    global indexOfLastInjectedBit 
    # index of the latest bits that will cause and not cause SDC 
    global indexOfLatesetSDC_nonSdcBit
    # index of the data to be injected, monotonically starts from the first data 
    global indexOfInjectedData
    # we use binary search to decide to bit to be injected, requiring upper and lower bound for the binary search
    # upper bound for the binary search 
    global upperIndexOfInjectedBit
    # lower bound for the binary search 
    global lowerIndexOfInjectedBit
    # list of the index of the "1" of the current injected data
    global indexOfBit_1
    # list of the index of the "0" of the current injected data
    global indexOfBit_0
    # flag for keep doing FI for "0"
    global isLookFor0
    # flag for keep doing FI for "1"
    global isLookFor1 
    # index of the injected bit 
    global indexOfInjectedBit
    # count of the times of FI so far
    global fiTime 

    # this is the first FI entry
    if(upperIndexOfInjectedBit== -1 and lowerIndexOfInjectedBit== -1):
        upperIndexOfInjectedBit = 0
        lowerIndexOfInjectedBit = len(indexList)-1
    # FI entry for the following bits

    # update the bound for the binary search based on SDC caused by last FI
    elif(sdcFromLastFI==True):
        # last FI causes SDC, we update the upper bound, narrow the bound to the right
        upperIndexOfInjectedBit =  indexList.index(indexOfLastInjectedBit) + 1
    elif(sdcFromLastFI==False):
        # last FI does not cause SDC, we update the lower bound, narrow the bound to the left
        lowerIndexOfInjectedBit = indexList.index(indexOfLastInjectedBit) - 1

    "cut-off for binary search: "
    " (1) two neigboring bits were visited, fault at higher bit causes SDC and not for the lower one"
    " (2) binary search ends at the two end (i.e., the highest or lowest bit) "
    # indexOfLastInjectedBit could be not in indexList when indexOfLastInjectedBit came from list of "0" and now the list 
    # is the list of "1" (occurs when in the transition for the list of 0 to list of 1) 
    if( ( (indexOfLastInjectedBit in indexList) and (upperIndexOfInjectedBit == lowerIndexOfInjectedBit == indexList.index(indexOfLastInjectedBit)))
        or (upperIndexOfInjectedBit>lowerIndexOfInjectedBit) ): 
 
        # binary FI at the current list of "0" is done, 
        # initialize the value for the FI on the bits of "1"
        upperIndexOfInjectedBit = -1
        lowerIndexOfInjectedBit = -1
        # return False means there is no FI at current bit, look for the bits of "1"
        return False, 0
    else:
        # get the index of the bit to be injected
        indexOfInjectedBit = indexList[ (upperIndexOfInjectedBit+lowerIndexOfInjectedBit)/2 ]  

        print(indexOfInjectedBit)
        fiTime += 1
        if(is0):
            # the data is negative and injected is '0', so the delta by bit flip is negative
            if(str(injectedData)[0] == "-"):
                afterBitFlip = injectedData - pow(2, intLen-(indexOfInjectedBit))
            # the data is positive and injected is '0', so the delta by bit flip is positive
            else:
                afterBitFlip = injectedData + pow(2, intLen-(indexOfInjectedBit))
        else:
            # the data is negative and injected is '1', so the delta by bit flip is positive
            if(str(injectedData)[0] == "-"):
                afterBitFlip = injectedData + pow(2, intLen-(indexOfInjectedBit))
            # the data is positive and injected is '1', so the delta by bit flip is negative
            else:
                afterBitFlip = injectedData - pow(2, intLen-(indexOfInjectedBit))  

    indexOfLastInjectedBit = indexOfInjectedBit 

    print afterBitFlip
    # return True means it'll keep doing FI for the current type of bits (2 types of bits: 0 / 1)
    # return the updated value after bit flip as well
    return True, afterBitFlip

def binaryBitFlip(dtype, val):      
    "binary FI on both tensor and scalar values"
    # index of the data to be injected, monotonically starts from the first data 
    global indexOfInjectedData
    # we use binary search to decide to bit to be injected, requiring upper and lower bound for the binary search
    # upper bound for the binary search 
    global upperIndexOfInjectedBit
    # lower bound for the binary search 
    global lowerIndexOfInjectedBit
    # list of the index of the "1" of the current injected data
    global indexOfBit_1
    # list of the index of the "0" of the current injected data
    global indexOfBit_0
    # flag for keep doing FI for "0"
    global isLookFor0
    # flag for keep doing FI for "1"
    global isLookFor1  
    # count the number of sdc-caused bits
    global sdcBitCount
    # cumulative SDC rate for current data
    global sdcRate
    global indexOfInjectedBit
    # index of the latest bits that will cause and will not cause SDC 
    global indexOfLatesetSDC_nonSdcBit
    # index of the last injected bit
    global indexOfLastInjectedBit
    # flagging whether keep doing FI, False when the last data item on current Op was injected
    global isKeepDoingFI
    # lenth of the integer
    global intLen
    # indicate the sdc result from the last FI
    global sdcFromLastFI  
    # sign for whether the FI on the current data is done
    global fiDonePerData
    # sign for the transition from 0-bit to 1-bit 
    global isTransit


    def updateSDC(indexList): 
        global indexOfLatesetSDC_nonSdcBit  

        # has been updating the index of the non-sdc bits, which means none of the bits cause SDC
        if(indexOfLatesetSDC_nonSdcBit[0] == -1):
            sdcBitCount = 0
        # has been udpating hte index of the sdc bits, which means all of the bits cause SDC
        elif(indexOfLatesetSDC_nonSdcBit[1] == -1):
            sdcBitCount = len(indexList)
        # found SDC-causing boundary
        else: 
            sdcBitCount = indexList.index(indexOfLatesetSDC_nonSdcBit[0]) + 1
            
        return sdcBitCount


    if(sdcFromLastFI == True and (not isTransit)):
        # update the index of the latest bit that cause SDC
        indexOfLatesetSDC_nonSdcBit[0] = indexOfLastInjectedBit 
    elif(sdcFromLastFI == False and (not isTransit)):
        # update the index of the latest bit that does not cause SDC
        indexOfLatesetSDC_nonSdcBit[1] = indexOfLastInjectedBit

    # treat a scalar value as an array with one element, and thus the FI process is the same for scalar and tensor
    isScalar = np.isscalar(val)
    if(isScalar):
        val = np.atleast_1d(val)
    else:
        val = np.asarray(val, type(val))
        valShape = val.shape
        val = val.flatten() 

    injectedData = val[ indexOfInjectedData ] 

    # create the index of the '1' and '0' bit of the current data
    if(indexOfBit_0==[] and indexOfBit_1==[]):
        # we remove the mantissa point since we're using fixed-width datatype (21 for integer, 10 for mantissa)

        if(isinstance(injectedData, int)):
            binVal = bin(abs(injectedData)).lstrip("0b") 
            binVal = binVal.zfill(21)
            intLen = 20
        elif(isinstance(injectedData, float)):
            binVal = float2bin(abs(injectedData)).replace('.', '') 
            intLen = 20

        for index in range(len(binVal)):
            if(binVal[index] == '1'):
                indexOfBit_1.append(index)
                isLookFor1 = True
            elif(binVal[index] == '0'):
                indexOfBit_0.append(index) 
                isLookFor0 = True

    "If a fault at higher bit does not cause SDC, faults at lower bits will not cause SDC"
    "Note that here the bit should be the same bit, i.e., either 0 or 1"
    "Therefore, we need to individually consider the SDC at the 0 bits and 1 bits"
    if(isLookFor0):
        isLookFor0, updatedVal = fiOnList(indexList=indexOfBit_0, is0=True, injectedData=injectedData, intLen= intLen) 
        # update the bit-flipped value
        if(isLookFor0):
            val[indexOfInjectedData] = updatedVal
        else:
            sdcBitCount = updateSDC(indexOfBit_0) 
            # intialize the for the FI on the bits of '1'
            indexOfLatesetSDC_nonSdcBit = [-1, -1]
            # transition from 0-bit to 1-bit
            isTransit = True 
    elif(isLookFor1):
        isTransit = False
        isLookFor1, updatedVal = fiOnList(indexList=indexOfBit_1, is0=False, injectedData=injectedData, intLen= intLen) 
        if(isLookFor1):
            val[indexOfInjectedData] = updatedVal
        else:
            sdcBitCount += updateSDC(indexOfBit_1)       

    # FI for current data is done.
    # will do FI for the next data.
    if(isLookFor0==False and isLookFor1==False):
        # sign for the finish of FI on current data
        fiDonePerData = True
        # cumulative SDC rate for the data at current Op
        sdcRate += float(sdcBitCount) / ( (len(indexOfBit_1)+len(indexOfBit_0))*len(val) ) 
        # initialize the value for performing FI on the next datapoint
        initBinaryInjection(isFirstTime=False) 
        indexOfInjectedData += 1  


    if(indexOfInjectedData < len(val)):
#   if(indexOfInjectedData < 100):
        isKeepDoingFI = True
    else:
        isKeepDoingFI = False
 
    if(not isScalar):
        return dtype.type( val.reshape(valShape) )
    else:
        return dtype.type(val[0])


##################################
def sequentialFIinit():
    # index of the data to be injected, monotonically starts from the first data 
    global indexOfInjectedData
    # index of the last injected bit
    global indexOfInjectedBit
    # sign for whether keep doing FI
    global isKeepDoingFI
 
    isKeepDoingFI = True
    indexOfInjectedData = 0
    indexOfInjectedBit = 0
 

def sequentialBitFlip(dtype, val):
    # index of the data to be injected, monotonically starts from the first data 
    global indexOfInjectedData
    # index of the last injected bit
    global indexOfInjectedBit
    # sign for whether keep doing FI
    global isKeepDoingFI
 

    isScalar = np.isscalar(val)
    if(isScalar):
        val = np.atleast_1d(val)
    else:
        val = np.asarray(val, type(val))
        valShape = val.shape
        val = val.flatten() 

    injectedData = val[ indexOfInjectedData ] 

#   if(isinstance(injectedData, int)):
#       binVal = bin(abs(injectedData)).lstrip("0b") 
#       binVal = binVal.zfill(21)
#       maxIndex = 20
#   elif(isinstance(injectedData, float)):
    binVal = float2bin(abs(injectedData)).replace('.', '')  
    maxIndex = 30

    if(str(injectedData)[0] == "-"):
        if(binVal[indexOfInjectedBit] == '0'):
            # delta is negative
            injectedData -= pow(2, (20- indexOfInjectedBit))
        else:
            # delta is positive
            injectedData += pow(2, (20- indexOfInjectedBit))
    else:
        if(binVal[indexOfInjectedBit] == '0'):
            # delta is positive
            injectedData += pow(2, (20- indexOfInjectedBit))
        else:
            # delta is negative
            injectedData -= pow(2, (20- indexOfInjectedBit))

    val[indexOfInjectedData] = injectedData
    # index of the next bit to be injected
    indexOfInjectedBit += 1
    # inject the next datapoint
    if(indexOfInjectedBit > maxIndex):
        indexOfInjectedData += 1
        indexOfInjectedBit = 0

    if(indexOfInjectedData < len(val)):
#   if(indexOfInjectedData < 100):
        isKeepDoingFI = True
    else:
        isKeepDoingFI = False
 
    if(not isScalar):
        return dtype.type( val.reshape(valShape) )
    else:
        return dtype.type(val[0])






