# Copyright (c) 2017 Andrey Voroshilov

import os
import tensorflow as tf
import numpy as np
import scipy.io
import time

from PIL import Image

from argparse import ArgumentParser

import TensorFI_ranger as ti



os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def imread_resize(path):
    img_orig = scipy.misc.imread(path)
    img = scipy.misc.imresize(img_orig, (227, 227)).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    return img, img_orig.shape

def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)
    
def get_dtype_np():
    return np.float32

def get_dtype_tf():
    return tf.float32
    
# SqueezeNet v1.1 (signature pool 1/3/5)
########################################

def load_net(data_path):
    if not os.path.isfile(data_path):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % data_path)

    weights_raw = scipy.io.loadmat(data_path)
    
    # Converting to needed type
    conv_time = time.time()
    weights = {}
    for name in weights_raw:
        weights[name] = []
        # skipping '__version__', '__header__', '__globals__'
        if name[0:2] != '__':
            kernels, bias = weights_raw[name][0]
            weights[name].append( kernels.astype(get_dtype_np()) )
            weights[name].append( bias.astype(get_dtype_np()) )
    print("Converted network data(%s): %fs" % (get_dtype_np(), time.time() - conv_time))
    
    mean_pixel = np.array([104.006, 116.669, 122.679], dtype=get_dtype_np())
    return weights, mean_pixel

def preprocess(image, mean_pixel):
    swap_img = np.array(image)
    img_out = np.array(swap_img)
    img_out[:, :, 0] = swap_img[:, :, 2]
    img_out[:, :, 2] = swap_img[:, :, 0]
    return img_out - mean_pixel

def unprocess(image, mean_pixel):
    swap_img = np.array(image + mean_pixel)
    img_out = np.array(swap_img)
    img_out[:, :, 0] = swap_img[:, :, 2]
    img_out[:, :, 2] = swap_img[:, :, 0]
    return img_out

def get_weights_biases(preloaded, layer_name):
    weights, biases = preloaded[layer_name]
    biases = biases.reshape(-1)
    return (weights, biases)

def fire_cluster(net, x, preloaded, cluster_name):
    # central - squeeze
    layer_name = cluster_name + '/squeeze1x1'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='VALID')
    x = _act_layer(net, layer_name + '_actv', x)
    
    # left - expand 1x1
    layer_name = cluster_name + '/expand1x1'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x_l = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='VALID')
    x_l = _act_layer(net, layer_name + '_actv', x_l)

    # right - expand 3x3
    layer_name = cluster_name + '/expand3x3'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x_r = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='SAME')
    x_r = _act_layer(net, layer_name + '_actv', x_r)
    
    # concatenate expand 1x1 (left) and expand 3x3 (right)
    x = tf.concat([x_l, x_r], 3)
    net[cluster_name + '/concat_conc'] = x
    
    return x

def net_preloaded(preloaded, input_image, pooling, needs_classifier=False, keep_prob=None):
    net = {}
    cr_time = time.time()

    x = tf.cast(input_image, get_dtype_tf())

    # Feature extractor
    #####################
    
    # conv1 cluster
    layer_name = 'conv1'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='VALID', stride=(2, 2))
    x = _act_layer(net, layer_name + '_actv', x)
    x = _pool_layer(net, 'pool1_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='VALID')

    # fire2 + fire3 clusters
    x = fire_cluster(net, x, preloaded, cluster_name='fire2')
    fire2_bypass = x
    x = fire_cluster(net, x, preloaded, cluster_name='fire3')
    x = _pool_layer(net, 'pool3_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='VALID')

    # fire4 + fire5 clusters
    x = fire_cluster(net, x, preloaded, cluster_name='fire4')
    fire4_bypass = x
    x = fire_cluster(net, x, preloaded, cluster_name='fire5')
    x = _pool_layer(net, 'pool5_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='VALID')

    # remainder (no pooling)
    x = fire_cluster(net, x, preloaded, cluster_name='fire6')
    fire6_bypass = x
    x = fire_cluster(net, x, preloaded, cluster_name='fire7')
    x = fire_cluster(net, x, preloaded, cluster_name='fire8')
    x = fire_cluster(net, x, preloaded, cluster_name='fire9')
    
    # Classifier
    #####################
    if needs_classifier == True:
        # Dropout [use value of 50% when training]
        x = tf.nn.dropout(x, keep_prob)
    
        # Fixed global avg pool/softmax classifier:
        # [227, 227, 3] -> 1000 classes
        layer_name = 'conv10'
        weights, biases = get_weights_biases(preloaded, layer_name)
        x = _conv_layer(net, layer_name + '_conv', x, weights, biases)
        x = _act_layer(net, layer_name + '_actv', x)
        
        # Global Average Pooling
        x = tf.nn.avg_pool(x, ksize=(1, 13, 13, 1), strides=(1, 1, 1, 1), padding='VALID')
        net['classifier_pool'] = x
        
        x = tf.nn.softmax(x)
        net['classifier_actv'] = x
    
    print("Network instance created: %fs" % (time.time() - cr_time))
   
    return net
    
def _conv_layer(net, name, input, weights, bias, padding='SAME', stride=(1, 1)):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, stride[0], stride[1], 1),
            padding=padding)
    x = tf.nn.bias_add(conv, bias)
    net[name] = x
    return x

def _act_layer(net, name, input):
    x = tf.nn.relu(input)
    net[name] = x
    return x
    
def _pool_layer(net, name, input, pooling, size=(2, 2), stride=(3, 3), padding='SAME'):
    if pooling == 'avg':
        x = tf.nn.avg_pool(input, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                padding=padding)
    else:
        x = tf.nn.max_pool(input, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                padding=padding)
    net[name] = x
    return x

def build_parser():
    ps = ArgumentParser()
    ps.add_argument('--in',             dest='input', help='input file', metavar='INPUT', required=False)
    ps.add_argument('--fool',           dest='fool', type=int, help='if image needs to be altered to fool the network classification (argument - class number)', metavar='FOOL')
    return ps

def main():
    import time

    parser = build_parser()
    options = parser.parse_args()

    # Loading image
#    img_content, orig_shape = imread_resize(options.input)
#    img_content_shape = (1,) + img_content.shape

    # Loading ImageNet classes info
    classes = []
    with open('synset_words.txt', 'r') as classes_file:
        classes = classes_file.read().splitlines()

    # Loading network
    data, sqz_mean = load_net('sqz_full.mat')

    config = tf.ConfigProto(log_device_placement = False)
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'

    g = tf.Graph()
    

    sess = tf.Session(config=config)

    # Building network
    image = tf.placeholder(dtype=get_dtype_tf(), shape=[None,224,224,3], name="image_placeholder")
    keep_prob = tf.placeholder(get_dtype_tf())
    sqznet = net_preloaded(data, image, 'max', True, keep_prob)


    fi = ti.TensorFI(sess, logLevel = 50, name = "convolutional", disableInjections=False)



    top1 = open("top1.csv", "a")
    top5 = open("top5.csv", "a")

    "specify the path to the images that you want to inject faults"
    path = "./fiImg/"
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f: 
            files.append(os.path.join(r, file))
    ##########################################


    "provide the labels"
    label = {
    './fiImg/ILSVRC2012_val_00000014.JPEG' : 757,
    './fiImg/ILSVRC2012_val_00006922.JPEG' : 569, 
    './fiImg/ILSVRC2012_val_00000198.JPEG' : 16,
    './fiImg/ILSVRC2012_val_00001892.JPEG' : 670,
    './fiImg/ILSVRC2012_val_00000793.JPEG' : 654,
    './fiImg/ILSVRC2012_val_00001807.JPEG' : 894,
    './fiImg/ILSVRC2012_val_00000434.JPEG' : 675,
    './fiImg/ILSVRC2012_val_00000573.JPEG' : 874,
    './fiImg/ILSVRC2012_val_00000604.JPEG' : 273,
    './fiImg/ILSVRC2012_val_00000903.JPEG' : 919,
    }



    #boundRes = open("outOfBound.csv", "a")
    #coverageRes = open('coverage.csv', 'a')
    fiRun = 1000

    for img in files:
 
        # Loading image
        img_content, orig_shape = imread_resize(img)

        img_content = scipy.misc.imresize(img_content, [224,224,3]) 

        "generate label"
        lab = label[ img ] 


        t1 = 0.
        t5 = 0.
        outBound = 0.
        coverage = 0.
        for j in range(fiRun):

            ti.injectFault.outOfBound = False
            sqznet_results = sess.run(sqznet['classifier_actv'], feed_dict={image: [preprocess(img_content, sqz_mean)], keep_prob: 1.})[0][0][0]
     

            pred = (np.argsort(sqznet_results)[::-1])[0:5]
        
            if( pred[0] == lab):
                top1.write(`1` + ",")
                top5.write(`1` + ",")
                t1 += 1
                t5 += 1
            elif(lab in pred[1:]):
                top5.write(`1` + ",")
                top1.write(`0` + ",")
                t5 += 1
            else:
                top1.write(`0` + ",")
                top5.write(`0` + ",")

            '''
            if( ti.injectFault.outOfBound ):
                outBound += 1
                boundRes.write( ti.injectFault.outOfBound + "," )
                if( lab != pred[0] ):
                    coverage += 1
            '''
            print 'input ', img, 'fi time: ', j, 'top1: ',  t1/(j+1), 'top5: ', t5/(j+1)
 
            #coverageRes.write(`(fiRun-t1)/fiRun` + "," + `(fiRun-t5)/fiRun`  + "," + `coverage/fiRun` + "\n")
            #boundRes.write("\n") 
            #print 'input ', img, 'fi time: ', j, 'top1: ',  t1/(j+1), 'top5: ', t5/(j+1)


            top1.write("\n")
            top5.write("\n")
 


        # Outputting result
#        sqz_class = np.argmax(sqznet_results)
#        print("\nclass: [%d] '%s' with %5.2f%% confidence" % (sqz_class, classes[sqz_class], sqznet_results[sqz_class] * 100))

        

if __name__ == '__main__':
    main()
