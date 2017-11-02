
import tensorflow as tf
import numpy as np
import os


def get_files(file_dir):
    CLL = []
    label_CLL = []
    FL = []
    label_FL = []
    MCL =[]
    label_MCL=[]
    for file in os.listdir(file_dir):
        name = file
        if name[0]=='C':
            CLL.append(file_dir +"/"+ file)
            label_CLL.append(int(0))
        elif name[0]=='F':
            FL.append(file_dir +"/"+ file)
            label_FL.append(int(1))
        elif name[0]=='M':
            MCL.append(file_dir +"/"+ file)
            label_MCL.append(int(2))
    print('There are %d CLL\nThere are %d FL\nThere are %d MCL' %(len(CLL), len(FL),len(MCL)))
    
    image_list = np.hstack((CLL, FL,MCL))
    label_list = np.hstack((label_CLL, label_FL,label_MCL))
    
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    
    label_list = [int(i) for i in label_list]
    return image_list, label_list
 
 
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
     
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)    #this one our picture has same size
     
    # if you want to test the generated batches of images, you might want to comment the following line.
    #biao zhun hua
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


# # 
# import matplotlib.pyplot as plt
#   
# BATCH_SIZE = 2
# CAPACITY = 256
# IMG_W = 36
# IMG_H = 36
#  
# train_dir = 'C:/Users/yinyy/Desktop/save_jpg'
#   
# image_list, label_list = get_files(train_dir)
# image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#   
# with tf.Session() as sess:
#     i = 0
#       

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
# #%%





    
