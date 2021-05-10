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




sess = tf.InteractiveSession(  )
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")


OP_FOR_EVAL = model.y
INPUT_x = model.x
INPUT_keep_prob = model.keep_prob



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












