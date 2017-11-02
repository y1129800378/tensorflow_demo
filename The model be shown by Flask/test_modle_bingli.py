
import os
import numpy as np
import tensorflow as tf 
import input_data_bingli as input_data
import model_self_bingli as model

#%% Evaluate one image
# when training, comment the following codes.
 
#  
from PIL import Image
import matplotlib.pyplot as plt

possible1=0
possible2=0
possible3=0

loabel=''
def get_one_image(train):
    image = Image.open(train)
    plt.imshow(image)
    image = image.resize([36, 36])
    image = np.array(image)
    return image
train_dir = '/home/yyy/Downloads/fl2.jpg' 
def evaluate_one_image(train_dir):
    '''Test one image against the saved models and parameters
    '''
     
    # you need to change the directories to yours.
    
    logs_train_dir = '/home/yyy/Downloads/save_data'

    image_array = get_one_image(train_dir)
    print("test")
    plt.imshow(image_array)
#     plt.show()
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 3
         
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 36, 36, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
         
        logit = tf.nn.softmax(logit)
         
        x = tf.placeholder(tf.float32, shape=[36, 36, 3])
         
        # you need to change the directories to yours.
        logs_train_dir = '/home/yyy/Downloads/save_data' 
                        
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
            possible1=prediction[:, 0]
            possible2=prediction[:, 1] 
            possible3=prediction[:, 2]
            if max_index==0:
                loabel='This is a CLL with possibility %.6f' %(prediction[:, 0]-0.012)
                print('This is a 1 with possibility %.6f' %prediction[:, 0])
            elif max_index==1:
                loabel='This is a FL with possibility %.6f' %(prediction[:, 1]-0.013)
                print('This is a 2 with possibility %.6f' %prediction[:, 1])
            elif max_index==2:
                loabel='This is a MCL with possibility %.6f' %(prediction[:, 2]-0.012)
                print('This is a 3 with possibility %.6f' %prediction[:, 2])

    return possible1,possible2,possible3
#%%
evaluate_one_image(train_dir)




