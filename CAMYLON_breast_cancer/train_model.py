import os
import numpy as np
import tensorflow as tf
import input_train_data as input_data
import model
from tensorflow.python.framework.dtypes import float32, float32_ref
from numpy import float64

# with tf.device('/gpu:3')
#%%

N_CLASSES = 2
IMG_W = 256  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 256
BATCH_SIZE = 32
CAPACITY = 2000
MAX_STEP = 5000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.00001 # with current parameters, it is suggested to use learning rate<0.0001



#%%
def run_training(path_modle_save):
    
    # you need to change the directories to yours.
    
    logs_train_dir = path_modle_save
    logs_train_dir2 = path_modle_save
    train, train_label = input_data.get_files()
    
    train_batch, train_label_batch = input_data.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE, 
                                                          CAPACITY)      
    
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)        
    train_op = model.trainning(train_loss, learning_rate)
    train__acc = model.evaluation(train_logits, train_label_batch)
       
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    times_floag=0
    
    writer1=tf.summary.FileWriter(logs_train_dir2,sess.graph)
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])
               
            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))

            if step % 1000 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
             
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
            
            if tra_loss<0.01:
                times_floag=times_floag+1
                if times_floag>1000:
                    checkpoint_path = os.path.join(logs_train_dir, 'good_finish_model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)   
                    print("good_finish")
                    break
            tf.summary.scalar('train_loss',tra_loss)
            writer1.add_summary(summary_str,step)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()

if __name__ == '__main__':   
    run_training("/home/yyy/new_model")






