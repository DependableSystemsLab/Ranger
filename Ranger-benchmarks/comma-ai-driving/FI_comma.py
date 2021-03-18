import tensorflow as tf
import scipy.misc
import model
import cv2
from subprocess import call
import driving_data
import time
import TensorFI as ti
import datetime
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 


#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)

"You need to provide the pre-trained weights"
"or you can train the model by yourself"
sess = tf.InteractiveSession( config=tf.ConfigProto(gpu_options=gpu_options) )
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0

# file for saving results
resFile = open("fi-result.csv", "a")

fi = ti.TensorFI(sess, logLevel = 50, name = "convolutional", disableInjections=True)


"specify the input for injection. You can change to any other inputs"
index = [481, 1387, 1611, 4320, 5925, 12241, 15941, 20446, 24898, 25692]

for i in index:
    full_image = scipy.misc.imread("driving_dataset/" + str(i) + ".jpg", mode="RGB")
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0

    '''    
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi 
#    call("clear")
    print(i , ".png", " Predicted steering angle: " + str(degrees) + " degrees", driving_data.ys[i])
    resFile.write(`i` + "," + `degrees` + "," + `driving_data.ys[i]` + "\n")
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    i += 1 

    '''

    fi.turnOnInjections()
    degrees = model.y.eval(feed_dict={model.x: [image], model.prob1: 1.0, model.prob2: 1.})[0][0] * 180.0 / scipy.pi 
    golden = degrees 
    print(degrees)
    fi.turnOnInjections()
 
    totalFI = 0.
    sdcCount = 0  
    fiTime = 1
    for j in range(fiTime):
        degrees = model.y.eval(feed_dict={model.x: [image], model.prob1: 1.0, model.prob2: 1.})[0][0] * 180.0 / scipy.pi 

        resFile.write(`degrees` + ",")
    print(i, j, '------------', 'comma, org')
    resFile.write("\n")

    
