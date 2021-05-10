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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

sess = tf.InteractiveSession( config=tf.ConfigProto(gpu_options=gpu_options)  )


saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0

resFile = open("wholeSetEval.csv", "a")

#fi = ti.TensorFI(sess, logLevel = 50, name = "convolutional", disableInjections=True)


tx, ty = driving_data.LoadWholeSet()

numImg = driving_data.num_test_images

print( numImg ) 
#while(cv2.waitKey(10) != ord('q')):

for i in range(len(ty)):
#    full_image = scipy.misc.imread("driving_dataset/" + str(i) + ".jpg", mode="RGB")
#    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
#    ind = index[i]
    image = tx[i]
    label = ty[i][0]
        
    deg =  model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.})[0][0] #* 180.0 / scipy.pi 
    resFile.write(`deg` + "," + `label` + "\n")
    print(deg, "------", i)




    # cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    # smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    # M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    # dst = cv2.warpAffine(img,M,(cols,rows))
    # cv2.imshow("steering wheel", dst)
    # i += 1 

    
    '''
    fi.turnOffInjections()
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi 
    golden = degrees 

    fi.turnOnInjections()
    ti.faultTypes.sequentialFIinit()

    bf = datetime.datetime.now()
    totalFI = 0.
    sdcCount = 0  
    while(ti.faultTypes.isKeepDoingFI):
        degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi 

        totalFI += 1
        resFile.write(`abs(degrees - golden)` + ",")
	print(i, totalFI)
    resFile.write("\n")

    '''
        
#        cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
#        #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
#        #and the predicted angle
#        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
#        M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
#        dst = cv2.warpAffine(img,M,(cols,rows))
#        cv2.imshow("steering wheel", dst)
        
    
    
#cv2.destroyAllWindows()
