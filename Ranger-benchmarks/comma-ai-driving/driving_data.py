import scipy.misc
import random

xs = []
ys = []

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0
test_batch_pointer = 0

#read data.txt
with open("driving_dataset/data.txt") as f:
    for line in f:
        xs.append("driving_dataset/" + line.split()[0])
        #the paper by Nvidia uses the inverse of the turning radius,
        #but steering wheel angle is proportional to the inverse of turning radius
        #so the steering wheel angle in radians is used as the output


	#### NOTE: we use the steering angel as the output directly
        ys.append(float(line.split()[1]))  #* scipy.pi / 180)
 
#get number of images
num_images = len(xs)


org_xs = xs
org_ys = ys

random.seed(1)
#shuffle list of images
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

# 70% training; 20% val; 10% testing
train_xs = xs[:int(len(xs) * 0.7)]
train_ys = ys[:int(len(xs) * 0.7)]

val_xs = xs[int(len(xs) * 0.7): int(len(xs) * 0.9)]
val_ys = ys[int(len(xs) * 0.7): int(len(xs) * 0.9)]

test_xs = xs[-int(len(xs) * 0.1):]
test_ys = ys[-int(len(xs) * 0.1):]

'''
a = open("res.csv", "a")
for i in range( len(ys) ):
    a.write( `ys[i]` + "\n")
'''


num_train_images = len(train_xs)
num_val_images = len(val_xs)
num_test_images = len(test_xs)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images])[-150:], [66, 200]) / 255.0)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images])[-150:], [66, 200]) / 255.0)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out

def LoadTestBatch(batch_size):
    global test_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(test_xs[(test_batch_pointer + i) % num_test_images])[-150:], [66, 200]) / 255.0)
        y_out.append([test_ys[(test_batch_pointer + i) % num_test_images]])
    test_batch_pointer += batch_size
    return x_out, y_out


def LoadValSet(): 
    x_out = []
    y_out = []
    for i in range(num_val_images):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(val_xs[ i ] )[-150:], [66, 200]) / 255.0)
        y_out.append([val_ys[ i ]])
    return x_out, y_out

def LoadTestSet():
    x_out = []
    y_out = []
    for i in range(num_test_images):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(test_xs[ i ] )[-150:], [66, 200]) / 255.0)
        y_out.append([test_ys[ i ]])
    return x_out, y_out

def LoadWholeSet():
    x_out = []
    y_out = []
    for i in range( len(org_ys) ):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(org_xs[ i ] )[-150:], [66, 200]) / 255.0)
        y_out.append([org_ys[ i ]])
    return x_out, y_out


