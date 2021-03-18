#!/usr/bin/env python

import os
from datetime import datetime
import time
import tensorflow as tf
import numpy as np
import sys
import select
import cPickle as pickle
from IPython import embed
from sklearn.utils.extmath import softmax
import imagenet_input as data_input
import resnet
import os
import TensorFI as ti



os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Dataset Configuration
tf.app.flags.DEFINE_string('test_dataset', 'scripts/val.txt', """Path to the test dataset list file""")
tf.app.flags.DEFINE_string('test_image_root', '/data1/common_datasets/imagenet_resized/ILSVRC2012_val/', """Path to the root of ILSVRC2012 test images""")
tf.app.flags.DEFINE_string('mean_path', './ResNet_mean_rgb.pkl', """Path to the imagenet mean""")
tf.app.flags.DEFINE_integer('num_classes', 1000, """Number of classes in the dataset.""")
tf.app.flags.DEFINE_integer('num_test_instance', 50000, """Number of test images.""")

# Network Configuration
tf.app.flags.DEFINE_integer('batch_size', 100, """Number of images to process in a batch.""")

# Optimization Configuration
tf.app.flags.DEFINE_float('l2_weight', 0.0001, """L2 loss weight applied all the weights""")
tf.app.flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")
tf.app.flags.DEFINE_float('initial_lr', 0.1, """Initial learning rate""")
tf.app.flags.DEFINE_string('lr_step_epoch', "80.0,120.0,160.0", """Epochs after which learing rate decays""")
tf.app.flags.DEFINE_float('lr_decay', 0.1, """Learning rate decay factor""")
tf.app.flags.DEFINE_boolean('finetune', False, """Whether to finetune.""")

# Training Configuration
tf.app.flags.DEFINE_string('checkpoint', './alexnet_baseline_2/model.ckpt-399999', """Path to the model checkpoint file""")
tf.app.flags.DEFINE_string('output_file', './alexnet_baseline_2/eval.pkl', """Path to the result pkl file""")
tf.app.flags.DEFINE_integer('test_iter', 100, """Number of test batches during the evaluation""")
tf.app.flags.DEFINE_integer('display', 100, """Number of iterations to display training info.""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.95, """The fraction of GPU memory to be allocated""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")

FLAGS = tf.app.flags.FLAGS


def get_lr(initial_lr, lr_decay, lr_decay_steps, global_step):
    lr = initial_lr
    for s in lr_decay_steps:
        if global_step >= s:
            lr *= lr_decay
    return lr


def train():
    print('[Dataset Configuration]')
    print('\tImageNet test root: %s' % FLAGS.test_image_root)
    print('\tImageNet test list: %s' % FLAGS.test_dataset)
    print('\tNumber of classes: %d' % FLAGS.num_classes)
    print('\tNumber of test images: %d' % FLAGS.num_test_instance)

    print('[Network Configuration]')
    print('\tBatch size: %d' % FLAGS.batch_size)
    print('\tCheckpoint file: %s' % FLAGS.checkpoint)

    print('[Optimization Configuration]')
    print('\tL2 loss weight: %f' % FLAGS.l2_weight)
    print('\tThe momentum optimizer: %f' % FLAGS.momentum)
    print('\tInitial learning rate: %f' % FLAGS.initial_lr)
    print('\tEpochs per lr step: %s' % FLAGS.lr_step_epoch)
    print('\tLearning rate decay: %f' % FLAGS.lr_decay)

    print('[Evaluation Configuration]')
    print('\tOutput file path: %s' % FLAGS.output_file)
    print('\tTest iterations: %d' % FLAGS.test_iter)
    print('\tSteps per displaying info: %d' % FLAGS.display)
    print('\tGPU memory fraction: %f' % FLAGS.gpu_fraction)
    print('\tLog device placement: %d' % FLAGS.log_device_placement)


    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Get images and labels of ImageNet
        print('Load ImageNet dataset')
        with tf.device('/cpu:0'):
            print('\tLoading test data from %s' % FLAGS.test_dataset)
            with tf.variable_scope('test_image'):
                test_images, test_labels = data_input.inputs(FLAGS.test_image_root, FLAGS.test_dataset, FLAGS.batch_size, False, num_threads=1, center_crop=True)



        # Build a Graph that computes the predictions from the inference model.
        images = tf.placeholder(tf.float32, [FLAGS.batch_size, data_input.IMAGE_HEIGHT, data_input.IMAGE_WIDTH, 3])
        labels = tf.placeholder(tf.int32, [FLAGS.batch_size])
 



        # Build model
        with tf.device('/GPU:0'):
            hp = resnet.HParams(batch_size=FLAGS.batch_size,
                                num_gpus=1,
                                num_classes=FLAGS.num_classes,
                                weight_decay=FLAGS.l2_weight,
                                momentum=FLAGS.momentum,
                                finetune=FLAGS.finetune)
        network = resnet.ResNet(hp, [images], [labels], global_step)
        network.build_model()
        print('\tNumber of Weights: %d' % network._weights)
        print('\tFLOPs: %d' % network._flops)

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))

        '''debugging attempt
        from tensorflow.python import debug as tf_debug
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        def _get_data(datum, tensor):
            return tensor == train_images
        sess.add_tensor_filter("get_data", _get_data)
        '''

        sess.run(init)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=10000)
        if FLAGS.checkpoint is not None:
           saver.restore(sess, FLAGS.checkpoint)
           print('Load checkpoint %s' % FLAGS.checkpoint)
        else:
            print('No checkpoint file of basemodel found. Start from the scratch.')

 
        # Start queue runners & summary_writer
        tf.train.start_queue_runners(sess=sess)



        

        fi = ti.TensorFI(sess, logLevel = 50, name = "convolutional", disableInjections=False)

        # save the results
        t1 = open("fi-org-resnet-top1.csv", "a")
        t5 = open("fi-org-resnet-top5.csv", "a")

        fiTime = 1

        for i in range(FLAGS.test_iter):
            fi.turnOffInjections()

            test_images_val, test_labels_val = sess.run([test_images[0], test_labels[0]])


            fi.turnOnInjections()
            for j in range(fiTime):
                probs = sess.run([ network.probs ],
                            feed_dict={network.is_train:False, images:test_images_val, labels:test_labels_val})
      
                probs = np.asarray(probs)
                probs = probs[0]


                counter = 0
                for each_prob in probs:
                    pred = (np.argsort(each_prob)[::-1])[0:5]
                    label = test_labels_val[counter]
                    counter+=1
        
                    print(pred,  'label:', label)

                    if(label == pred[0]):
                        t1.write(`1` + ",")
                        t5.write(`1` + ",")
                    elif(label in pred[1:]):
                        t1.write(`0` + ",")
                        t5.write(`1` + ",")
                    else:
                        t1.write(`0` + ",")
                        t5.write(`0` + ",")        

                    print('--------fi on resnet, %d img, %d FI run' % (i+1, j+1) )            
            
            t1.write("\n")
            t5.write("\n")

 
        


def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()
