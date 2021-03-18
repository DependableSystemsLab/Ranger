import tensorflow as tf
import scipy.misc
import model
import cv2
from subprocess import call
import driving_data
import time

import datetime
import os
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))



saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")


xs, ys = driving_data.LoadTestSet()
