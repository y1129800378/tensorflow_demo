
# coding: utf-8

# In[1]:

import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import argparse
# In[2]:
# model_save_path="/home/yyy/workspace2/get_patch/NBI"
# predict_img_path="/home/yyy/workspace2/pic_data/NBI/"
# save_pic_tmp=predict_img_path+"/tmp/"
# size=120

parser=argparse.ArgumentParser(description='Download and draw pic.')
parser.add_argument('--model_save_path',default='/home/yyy/Downloads/lymphoma/model/')
parser.add_argument('--predict_img_path',default='/home/yyy/Downloads/lymphoma/test/')
parser.add_argument('--size',type=int,default=150)
args=parser.parse_args()
model_save_path=args.model_save_path
predict_img_path=args.predict_img_path
size=args.size
save_pic_tmp=predict_img_path+"/tmp/"

lines = tf.gfile.GFile(model_save_path+'/output_labels.txt').readlines()
uid_to_human = {}
#一行一行读取数据
for uid,line in enumerate(lines) :
    #去掉换行符
    line=line.strip('\n')
    uid_to_human[uid] = line

def id_to_string(node_id):
    if node_id not in uid_to_human:
        return ''
    return uid_to_human[node_id]


#创建一个图来存放google训练好的模型
with tf.gfile.FastGFile(model_save_path+'/model.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    #遍历目录    
#     for root,dirs,files in os.walk('retrain/images/'):
    for root,dirs,files in os.walk(predict_img_path):
        if(files==[]):
            print("dont find files to predicte")
            break

        for file in files:
            #载入图片
            cll=0
            fl=0
            mcl=0	
            if(file[-8:]=="_ROI.jpg") or (file[-13:]=="_Predicte.jpg"):
                pass
            else:
                try:
                    img=cv2.imread(root+file)
                    size_pic=img.shape
                    width=size_pic[0]
                    height=size_pic[1]

                    for x_wigth in range(0,width,int(size)):
                        for y_height in range(0,height,int(size)):
                            if (width-x_wigth)<size or (height-y_height)<size:
                                pass
                            else:
                                img_patch=img[int(x_wigth):int(x_wigth+size),int(y_height):int(y_height+size)]
                                isExists=os.path.exists(save_pic_tmp)
                                if not isExists:
                                    os.makedirs(save_pic_tmp) 
                                cv2.imwrite(save_pic_tmp+"tmp.jpg",img_patch)
                                image_data = tf.gfile.FastGFile(os.path.join(save_pic_tmp,"tmp.jpg"), 'rb').read()
                                predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})#图片格式是jpg格式
                                predictions = np.squeeze(predictions)#把结果转为1维数据

                                top_k = predictions.argsort()[::-1] 
                                
                                for node_id in top_k:
                                    human_string=id_to_string(node_id)               
                                    score=predictions[node_id]
#                                     print("%s (score=%.5f)" % (human_string,score))   	
#                                 print("x_wigth:",x_wigth)
#                                 print("y_height:",y_height)
                                if(top_k[0]==1):
                                    cll=cll+1
                                elif(top_k[0]==0):
                                    fl=fl+1
                                elif(top_k[0]==2):
                                    mcl=mcl+1
                except:
                    print("finish")
            result=[cll,fl,mcl]
            lists=["cll","fl","mcl"]   
            print(file," result is:",lists[result.index(max(result))])  
            os.remove(save_pic_tmp+"tmp.jpg")

                         


# In[ ]:




# In[ ]:


