from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf 
import input_data
import model
import glob
import re
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import scipy.misc
import glob

def get_one_image(img_dir):


    image = Image.open(img_dir)

    image = image.resize([150, 150])
    image = np.array(image)
    return image

def evaluate_one_image():
    '''Test one image against the saved models and parameters
    '''
    
    # you need to change the directories to yours.
    train_dir = '/home/yyy/Downloads/BOT/test/N_0_1001_1_2.jpg'

    image_array = get_one_image(train_dir)
    
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2
        
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 150, 150, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
        
        logit = tf.nn.softmax(logit)
        
        x = tf.placeholder(tf.float32, shape=[150, 150, 3])
        
        # you need to change the directories to yours.
        logs_train_dir = '/home/yyy/Downloads/BOT/train_model/' 
                       
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            
            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index==0:
                print('This is a cancer with possibility %.6f' %prediction[:, 0])
            else:
                print('This is a normal with possibility %.6f' %prediction[:, 1])
                
def evaluate_one_image_np(image_array):  
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2
        
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 150, 150, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
        
        logit = tf.nn.softmax(logit)
        
        x = tf.placeholder(tf.float32, shape=[150, 150, 3])
        
        # you need to change the directories to yours.
        logs_train_dir = '/home/yyy/Downloads/BOT/train_model/' 
                       
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            
            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index==0:
                print('This is a cancer with possibility %.6f' %prediction[:, 0])
                return 0
            else:
                print('This is a normal with possibility %.6f' %prediction[:, 1])
                return 1               
#             return prediction
# evaluate_one_image()      
          
num_pic=0      
for file in glob.glob('/home/yyy/Downloads/BOT/2017/*tiff'):  
    
    image = Image.open(file)
    image = np.array(image)     
    lower_red = np.array([10,10,10])
    upper_red = np.array([240, 240, 240])
    pic_yuantu_thr = cv2.inRange(image, lower_red, upper_red)
      
#     out_p=np.zeros(shape=(2048,2048,3))
    out_c=np.zeros(shape=(2048,2048))
        
    for x in range(0,1873,30):
        for y in range(0,1873,30):
            mask=pic_yuantu_thr[int(x):int(x+150),int(y):int(y+150)]
            white_pixel_cnt = cv2.countNonZero(mask) 
              
            if white_pixel_cnt > ((150 * 150) * 0.1): 
                yuantu_mask=image[int(x):int(x+150),int(y):int(y+150)]
                print("x",x,"y",y,"num_pic",str(num_pic))
                P=evaluate_one_image_np(yuantu_mask)  
#                 out_p[int(x):int(x+150),int(y):int(y+150),0:2]=P  
                   
                max_index = np.argmax(P)  
  
                if(P>0.5):
                    P=0
                    out_c[int(x):int(x+150),int(y):int(y+150)]=P 
                else:
                    P=255 
                    out_c[int(x):int(x+150),int(y):int(y+150)]=P    

    kernel=np.ones((40,40),np.uint8)
# closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
    dilation=cv2.dilate(out_c,kernel,iterations=1)
    open=cv2.erode(dilation,kernel,iterations=1)
    open=cv2.erode(open,kernel,iterations=1)
    dilation=cv2.dilate(open,kernel,iterations=1)
    f_name=file[29:]
    f_name=f_name[:-5]
    
    num_pic=num_pic+1
    scipy.misc.imsave('/home/yyy/Downloads/BOT/test/'+f_name+'.jpg',dilation)
    scipy.misc.imsave('/home/yyy/Downloads/BOT/test/'+f_name+'_yuanshi'+'.jpg',out_c)
    print(str(num_pic),f_name)
                 
                
def print_pic(pic):
    plt.figure("show_pic")
    plt.imshow(pic)
    plt.show()

# for file in glob.glob('/home/yyy/Downloads/BOT/2017/*tiff'):  
#     print(file)              
#     img=cv2.imread(file)
#     kernel=np.ones((40,40),np.uint8)
# # closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
#     dilation=cv2.dilate(img,kernel,iterations=1)
#     open=cv2.erode(dilation,kernel,iterations=1)
# 
#     open=cv2.erode(open,kernel,iterations=1)
#     dilation=cv2.dilate(open,kernel,iterations=1)
    
    
#     print_pic(img)
#     print_pic(dilation)              
                
                
                
                
                