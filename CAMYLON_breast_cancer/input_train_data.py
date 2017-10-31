
import tensorflow as tf
import numpy as np
import os


def get_files():
    Tumor=[]
    label_Tumor=[]
    Normal=[]
    label_Normal=[]
    
    file_dir1='/raid/CAMELYON/label-1'
    for file in os.listdir(file_dir1):
        name = file
        if name[0]=='t':
            Tumor.append(file_dir1 +"/"+ file)
            label_Tumor.append(int(0))
        elif name[0]=='n':
            Normal.append(file_dir1 +"/"+ file)
            label_Normal.append(int(1))
   
    file_dir2='/raid/CAMELYON/tumor-label-0'        
    for file in os.listdir(file_dir2):
        name = file
        if name[0]=='t':
            Tumor.append(file_dir2 +"/"+ file)
            label_Tumor.append(int(0))
        elif name[0]=='n':
            Normal.append(file_dir2 +"/"+ file)
            label_Normal.append(int(1))
 
    file_dir3='/raid/CAMELYON/normal-label-0'
    for file in os.listdir(file_dir3):
        name = file
        if name[0]=='t':
            Tumor.append(file_dir3 +"/"+ file)
            label_Tumor.append(int(0))
        elif name[0]=='n':
            Normal.append(file_dir3 +"/"+ file)
            label_Normal.append(int(1))
            
    file_dir4='/raid/CAMELYON/use-mask-label-1'
    for file in os.listdir(file_dir4):
        name = file
        if name[0]=='t':
            Tumor.append(file_dir4 +"/"+ file)
            label_Tumor.append(int(0))
        elif name[0]=='n':
            Normal.append(file_dir4 +"/"+ file)
            label_Normal.append(int(1))
    
    print('There are %d Tumor\nThere are %d Normal\n' %(len(Tumor), len(Normal)))
    

    image_list=np.hstack((Tumor,Normal))
    label_list = np.hstack((label_Tumor, label_Normal))
    
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    
    label_list = [int(i) for i in label_list]
    return image_list, label_list
# get_files() 
 
def get_batch(image, label, image_W, image_H, batch_size, capacity):

     
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
 
    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
     
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

     
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)    #this one our picture has same size
     
    # if you want to test the generated batches of images, you might want to comment the following line.
    #biao zhun hua
    image = tf.image.per_image_standardization(image)
     
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
     
    return image_batch, label_batch




    
