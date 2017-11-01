
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
#%%

# you need to change this to your data directory


def get_files():
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    cancer = []
    label_cancer = []
    normal = []
    label_normal = []
    file_dir1='/home/yyy/Downloads/BOT/patch_pic/'
    file_dir2='/home/yyy/Downloads/BOT/patch_pic_normal/'
    for file in os.listdir(file_dir1):
        cancer.append(file_dir1 +"/"+ file)
        label_cancer.append(0)
    for file in os.listdir(file_dir2):
        normal.append(file_dir2 +"/"+ file)
        label_normal.append(1)
        
    print('There are %d cancer\nThere are %d normal' %(len(cancer), len(normal)))
    
    image_list = np.hstack((cancer, normal))
    label_list = np.hstack((label_cancer, label_normal))
    
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    
    
    return image_list, label_list


#%%

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    
    ######################################
    # data argumentation should go to here
    ######################################
    
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    
    # if you want to test the generated batches of images, you might want to comment the following line.
    image = tf.image.per_image_standardization(image)
    
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)
    
    #you can also use shuffle_batch 
#    image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                      batch_size=BATCH_SIZE,
#                                                      num_threads=64,
#                                                      capacity=CAPACITY,
#                                                      min_after_dequeue=CAPACITY-1)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch


 
#%% TEST
# To test the generated batches of images
# When training the model, DO comment the following codes





#  
# BATCH_SIZE = 2
# CAPACITY = 256
# IMG_W = 150
# IMG_H = 150
#  
#  train
# image_list, label_list = get_files()
# image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#  
# with tf.Session() as sess:
#     i = 0
#      
#     #��������  ���г���  è����ս03 02��11 
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#      
#     try:
#         while not coord.should_stop() and i<1:
#              
#             img, label = sess.run([image_batch, label_batch])
#              
#             # just test one batch
#             for j in np.arange(BATCH_SIZE):
#                 print('label: %d' %label[j])
#                 plt.imshow(img[j,:,:,:])
#                 plt.show()
#             i+=1
#              
#     except tf.errors.OutOfRangeError:
#         print('done!')
#     finally:
#         coord.request_stop()
#     coord.join(threads)
# 
# 
# #%%





    
