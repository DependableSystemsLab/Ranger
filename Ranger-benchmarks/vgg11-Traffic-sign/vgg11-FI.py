"""
A VGG11 to work on German traffic sign dataset
Implementation from https://github.com/mohamedameen93/German-Traffic-Sign-Classification-Using-TensorFlow/blob/master/Traffic_Sign_Classifier.ipynb
Dataset is also available from the same site above

Important: The dataset in the above site is stored in python pickle, using python 3
Currently TensorFI only supports python 2, so you need to use python 3 to load the .p file
And then using pickle in python 2 version to store the dataset again. 
"""


import os 
import numpy as np
import cv2 
#import matplotlib.pyplot as plt 
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.contrib.layers import flatten
import pickle 
import TensorFI as ti
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# training hyperparameters
BATCHSIZE = 64
EPOCHS = 300
BATCHES_PER_EPOCH = 300

## Important, make sure the .p file can be loaded using pickle in python 2
#training_file = "Path_to_train_data" 
#testing_file = "Path_to_test_data"
training_file = "./new-dataset/train.p" 
testing_file = "./new-dataset/test.p" 



with open(training_file, mode='rb') as f:
    train = pickle.load(f) 
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

# Number of examples
n_train, n_test = X_train.shape[0], X_test.shape[0]

# What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# How many classes?
n_classes = np.unique(y_train).shape[0]

print("Number of training examples =", n_train)
print("Number of testing examples  =", n_test)
print("Image data shape  =", image_shape)
print("Number of classes =", n_classes)

'''
# show a random sample from each class of the traffic sign dataset
rows, cols = 4, 12
fig, ax_array = plt.subplots(rows, cols)
plt.suptitle('RANDOM SAMPLES FROM TRAINING SET (one for each class)')
for class_idx, ax in enumerate(ax_array.ravel()):
    if class_idx < n_classes:
        # show a random image of the current class
        cur_X = X_train[y_train == class_idx]
        cur_img = cur_X[np.random.randint(len(cur_X))]
        ax.imshow(cur_img)
        ax.set_title('{:02d}'.format(class_idx))
    else:
        ax.axis('off')
# hide both x and y ticks
plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
plt.draw()



train_distribution, test_distribution = np.zeros(n_classes), np.zeros(n_classes)
for c in range(n_classes):
    train_distribution[c] = np.sum(y_train == c) / n_train
    test_distribution[c] = np.sum(y_test == c) / n_test
fig, ax = plt.subplots()
col_width = 0.5
bar_train = ax.bar(np.arange(n_classes), train_distribution, width=col_width, color='r')
bar_test = ax.bar(np.arange(n_classes)+col_width, test_distribution, width=col_width, color='b')
ax.set_ylabel('PERCENTAGE OF PRESENCE')
ax.set_xlabel('CLASS LABEL')
ax.set_title('Classes distribution in traffic-sign dataset')
ax.set_xticks(np.arange(0, n_classes, 5)+col_width)
ax.set_xticklabels(['{:02d}'.format(c) for c in range(0, n_classes, 5)])
ax.legend((bar_train[0], bar_test[0]), ('train set', 'test set'))
plt.show()
'''

def preprocess_features(X, equalize_hist=True):
    '''
    # convert from RGB to YUV
    X = np.array([np.expand_dims(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV)[:, :, 0], 2) for rgb_img in X])

    # adjust image contrast
    if equalize_hist:
        X = np.array([np.expand_dims(cv2.equalizeHist(np.uint8(img)), 2) for img in X])
    '''
    X = np.float32(X)

    # standardize features
    X -= np.mean(X, axis=0)
    X /= (np.std(X, axis=0) + np.finfo('float32').eps)

    return X

X_train_norm = preprocess_features(X_train)
X_test_norm = preprocess_features(X_test)


# split into train and validation
VAL_RATIO = 0.2
X_train_norm, X_val_norm, y_train, y_val = train_test_split(X_train_norm, y_train, test_size=VAL_RATIO, random_state=0)


# create the generator to perform online data augmentation
image_datagen = ImageDataGenerator(rotation_range=15.,
                                   zoom_range=0.2,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1)
'''
# take a random image from the training set
img_rgb = X_train[0]

# plot the original image
plt.figure(figsize=(1,1))
plt.imshow(img_rgb)
plt.title('Example of RGB image (class = {})'.format(y_train[0]))
plt.show()

# plot some randomly augmented images
rows, cols = 4, 10
fig, ax_array = plt.subplots(rows, cols)
for ax in ax_array.ravel():
    augmented_img, _ = image_datagen.flow(np.expand_dims(img_rgb, 0), y_train[0:1]).next()
    ax.imshow(np.uint8(np.squeeze(augmented_img)))
plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
plt.suptitle('Random examples of data augmentation (starting from the previous image)')
plt.show()
'''


def weight_variable(shape, mu=0, sigma=0.1):
    initialization = tf.truncated_normal(shape=shape, mean=mu, stddev=sigma)
    return tf.Variable(initialization)


def bias_variable(shape, start_val=0.1):
    initialization = tf.constant(start_val, shape=shape)
    return tf.Variable(initialization)


def conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'):
    return tf.nn.conv2d(input=x, filter=W, strides=strides, padding=padding)


def max_pool_2x2(x):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


 

# network architecture definition
def my_net(x, n_classes):

    c1_out = 32
    conv1_1W = weight_variable(shape=(3, 3, 3, c1_out))
    conv1_1b = bias_variable(shape=(c1_out,))
    conv1_1 = tf.nn.relu(conv2d(x, conv1_1W) + conv1_1b)

    conv1_2W = weight_variable(shape=(3, 3, c1_out, c1_out))
    conv1_2b = bias_variable(shape=(c1_out,))
    conv1_2 = tf.nn.relu(conv2d(conv1_1, conv1_2W) + conv1_2b)

    pool1 = max_pool_2x2(conv1_2)

    drop1 = tf.nn.dropout(pool1, keep_prob=keep_prob)

    c2_out = 64
    conv2_1W = weight_variable(shape=(3, 3, c1_out, c2_out))
    conv2_1b = bias_variable(shape=(c2_out,))
    conv2_1 = tf.nn.relu(conv2d(pool1, conv2_1W) + conv2_1b)

    conv2_2W = weight_variable(shape=(3, 3, c2_out, c2_out))
    conv2_2b = bias_variable(shape=(c2_out,))
    conv2_2 = tf.nn.relu(conv2d(conv2_1, conv2_2W) + conv2_2b)

    pool2 = max_pool_2x2(conv2_2)

    drop2 = tf.nn.dropout(pool2, keep_prob=keep_prob)
 
    c3_out = 128
    conv3_1W = weight_variable(shape=(3, 3, c2_out, c3_out))
    conv3_1b = bias_variable(shape=(c3_out,))
    conv3_1 = tf.nn.relu(conv2d(pool2, conv3_1W) + conv3_1b)

    conv3_2W = weight_variable(shape=(3, 3, c3_out, c3_out))
    conv3_2b = bias_variable(shape=(c3_out,))
    conv3_2 = tf.nn.relu(conv2d(conv3_1, conv3_2W) + conv3_2b)

    pool3 = max_pool_2x2(conv3_2)


    c4_out = 128
    conv4_1W = weight_variable(shape=(3, 3, c3_out, c4_out))
    conv4_1b = bias_variable(shape=(c4_out,))
    conv4_1 = tf.nn.relu(conv2d(pool3, conv4_1W) + conv4_1b)

    conv4_2W = weight_variable(shape=(3, 3, c4_out, c4_out))
    conv4_2b = bias_variable(shape=(c4_out,))
    conv4_2 = tf.nn.relu(conv2d(conv4_1, conv4_2W) + conv4_2b)

    pool4 = max_pool_2x2(conv4_2)

    shape = int(np.prod(pool4.get_shape()[1:]))
    pool4 = tf.reshape(pool4, [-1, shape])
    fc1_out = 128
    fc1_W = weight_variable(shape=(pool4._shape[1].value, fc1_out))
    fc1_b = bias_variable(shape=(fc1_out,))
    fc1 = tf.matmul(pool4, fc1_W) + fc1_b 
    
    fc1 = tf.nn.relu(fc1)
    
    fc2_out = 128
    fc2_W = weight_variable(shape=(fc1._shape[1].value, fc2_out))
    fc2_b = bias_variable(shape=(fc2_out,))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b 

    fc2 = tf.nn.relu(fc2)
    drop_fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)

    fc3_out = n_classes
    fc3_W = weight_variable(shape=(fc2._shape[1].value, fc3_out))
    fc3_b = bias_variable(shape=(fc3_out,))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

# placeholders
x = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3))
y = tf.placeholder(dtype=tf.int32, shape=None)
keep_prob = tf.placeholder(tf.float32)


# training pipeline
lr = 0.001
logits = my_net(x, n_classes=n_classes)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss_function = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_step = optimizer.minimize(loss=loss_function)


# metrics and functions for model evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.cast(y, tf.int64))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# evalute data in a batch, used for typical fault-free run
def evaluate(X_data, y_data):
    
    num_examples = X_data.shape[0]
    total_accuracy = 0
    
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCHSIZE):
        batch_x, batch_y = X_data[offset:offset+BATCHSIZE], y_data[offset:offset+BATCHSIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += accuracy * len(batch_x)
        
    return total_accuracy / num_examples


# evalute data one by one, used for typical FI, e.g., binary FI
def evalEach(X_data, y_data):

#    num_examples = X_data.shape[0]
#    total_accuracy = 0
#    sess = tf.get_default_session()
#    for offset in range(0, num_examples, BATCHSIZE):
#        batch_x, batch_y = X_data[offset:offset+BATCHSIZE], y_data[offset:offset+BATCHSIZE]
    accuracy = sess.run(accuracy_operation, feed_dict={x: X_data, y: y_data, keep_prob: 1.0})
    return accuracy


# create a checkpointer to log the weights during training
checkpointer = tf.train.Saver()


global acy 
# start training
with tf.Session() as sess:

    "You need to first train the network" 
    train = False
    acy = 0

    SAVE_PATH = './save/'   # directory for saving your trained model
    # train the network
    if(train):
        sess.run(tf.global_variables_initializer())
#        checkpointer.restore(sess, 'checkpoints/traffic_sign_model.ckpt')
        for epoch in range(EPOCHS):

            print("EPOCH {} ...".format(epoch + 1))
    
            batch_counter = 0
            for batch_x, batch_y in image_datagen.flow(X_train_norm, y_train, batch_size=BATCHSIZE):

                batch_counter += 1
                _,los = sess.run([train_step,loss_function], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
                print(epoch, batch_counter, los)
                if batch_counter == BATCHES_PER_EPOCH:
                    break

            # at epoch end, evaluate accuracy on both training and validation set
            if((epoch+1) % 100 ==0):
                train_accuracy = evaluate(X_train_norm, y_train)
                val_accuracy = evaluate(X_test_norm, y_test)
                print('Train Accuracy = {:.3f} - Validation Accuracy: {:.3f}'.format(train_accuracy, val_accuracy))
 
            if(train_accuracy > acy):   # save the model only if it achives better performance 
                # log current weights
                print("save checkpoints")
                checkpointer.save(sess, save_path=SAVE_PATH)
                acy = train_accuracy
    # you can test the model after training
    else: 
        checkpointer.restore(sess, SAVE_PATH)   # restore the trained model
        
        # save FI results into file, "eachRes" saves each FI result
        fiRes = open("relu-vgg11-statistic.csv", "a")
        eachRes = open("relu-vgg11-eachFI.csv", "a")

        tX = X_test_norm[:1000, :, :, :]
        tY = y_test[:1000]

        fi = ti.TensorFI(sess, logLevel = 50, name = "convolutional", disableInjections=False)
        fi.turnOnInjections()

        "Here we choose the inputs with different labels"
        "You can change to any other inputs for injection"
        index = [48, 35, 29, 64, 33, 24, 31, 209, 201, 177]

        for each in index:
            X = X_test_norm[each, :, :, :]
            Y = y_test[each]


            X = X.reshape(1,32,32,3)
            Y = Y.reshape(1,1)


            sdcCount = 0.

            totalFI = 1
            for j in range(totalFI):

                acy = evalEach(X, Y)

                if(acy == 1.):
                    eachRes.write(`1` + ",")
                else:
                    eachRes.write(`0` + ",")
                    sdcCount += 1

                print(each, j, acy)
            fiRes.write( `sdcCount/totalFI` + "\n")
            eachRes.write("\n")
            



