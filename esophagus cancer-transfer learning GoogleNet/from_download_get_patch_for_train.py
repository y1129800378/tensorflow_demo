# _*_coding:utf-8_*_
import os
import re
import glob
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests
import argparse
from email.policy import default

def print_pic(pic):
    plt.figure("show_pic")
    plt.imshow(pic)
    plt.show()
 
#      path='/home/yyy/workspace2/pic_data/'
#     path_get_patch='/home/yyy/workspace2/get_patch/'
#     size=120
def catch_pic_for_train(path,path_get_patch,size):
    isExists=os.path.exists(path)
    if not isExists:
        print("dont find download data")
        return       
        isExists=os.path.exists(mask_save_dir)
#     isExists=os.path.exists(path_get_patch)
#     if not isExists:
#         os.makedirs(path_get_patch) 
#         os.makedirs(path_get_patch+"/normal/") 
#         os.makedirs(path_get_patch+"/cancel/") 
    ls = os.listdir(path)
    pic_ls=[]
    class_name=""
    for i in ls:
        print(i)
        if(i=="NBI"):
            pic_ls = os.listdir(path+"/"+"NBI")
            class_name="NBI"
        elif(i=="NBI,放大"):
            pic_ls = os.listdir(path+"/"+"NBI,放大")
            class_name="NBI_amplification"
        elif(i=="NBI,放大,低倍"):
            pic_ls = os.listdir(path+"/"+"NBI,放大,低倍")
            class_name="NBI_amplification_low"    
        elif(i=="NBI,放大,高倍"):
            pic_ls = os.listdir(path+"/"+"NBI,放大,高倍")
            class_name="NBI_amplification_high"         
        elif(i=="NBI,近景"):
            pic_ls = os.listdir(path+"/"+"NBI,近景")
            class_name="NBI_amplification_high"         
        elif(i=="NBI,远景"):
            pic_ls = os.listdir(path+"/"+"NBI,远景")
            class_name="NBI_amplification_far"  
        elif(i=="no_label"):
            pic_ls = os.listdir(path+"/"+"no_label")
            class_name="no_label"      
        elif(i=="白光"):
            pic_ls = os.listdir(path+"/"+"白光")
            class_name="white"   
        elif(i=="白光,近景"):
            pic_ls = os.listdir(path+"/"+"白光,近景")
            class_name="white_close"   
        elif(i=="白光,远景"):
            pic_ls = os.listdir(path+"/"+"白光,远景")
            class_name="white_far"    
        elif(i=="碘染色"):
            pic_ls = os.listdir(path+"/"+"碘染色")
            class_name="dian_stain"
        
        for pic in pic_ls:
            if pic[-8:]!="_ROI.jpg":                
                img=cv2.imread(path+"/"+i+"/"+pic)  
                img_mask=cv2.imread(path+"/"+i+"/"+pic[:-4]+"_ROI.jpg") 
#                print_pic(img_mask)
#                print_pic(img)
                print("get_patch_from:"+path+"/"+i+"/"+pic)
                size_pic=img.shape
                width=size_pic[0]
                height=size_pic[1]
                for x_wigth in range(0,width,int(size/10)):
                    for y_height in range(0,height,int(size/10)):
                        if (width-x_wigth)<size or (height-y_height)<size:
                            pass
                        else:
                            mask=img_mask[int(x_wigth):int(x_wigth+size),int(y_height):int(y_height+size)]
                            mask=cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
                            mask = cv2.inRange(mask, 0, 254)
                            white_pixel_cnt = cv2.countNonZero(mask)
                           
                            isExists=os.path.exists(path_get_patch+class_name+"/cancel/")
                            if not isExists:
                                os.makedirs(path_get_patch+class_name+"/cancel/") 
                            isExists=os.path.exists(path_get_patch+class_name+"/normal/")
                            if not isExists:
                                os.makedirs(path_get_patch+class_name+"/normal/") 
                            if white_pixel_cnt > ((size * size) * 0.5):
                                mask_img=img[int(x_wigth):int(x_wigth+size),int(y_height):int(y_height+size)]
                                cv2.imwrite(path_get_patch+class_name+"/normal/"+pic+"_"+str(x_wigth)+"_"+str(y_height)+"normal.jpg",mask_img)
#                                print_pic(mask_img)
                            if white_pixel_cnt < ((size * size) * 0.5):
                                mask_img=img[int(x_wigth):int(x_wigth+size),int(y_height):int(y_height+size)]
                                cv2.imwrite(path_get_patch+class_name+"/cancel/"+pic+"_"+str(x_wigth)+"_"+str(y_height)+"cancel.jpg",mask_img)
#                                print_pic(mask_img)
#                 print_pic(img)   
#                 print_pic(img_mask)        
parser=argparse.ArgumentParser(description='Download and draw pic.')
parser.add_argument('--path',type=str,default='/home/yyy/workspace2/pic_data/')
parser.add_argument('--path_get_patch',type=str,default='/home/yyy/workspace2/get_patch/')
parser.add_argument('--size',type=int,default=120)
args=parser.parse_args()
path=args.path
path_get_patch=args.path_get_patch
size=args.size
#     path_get_patch='/home/yyy/workspace2/get_patch/'
#     size=120
catch_pic_for_train(path,path_get_patch,size)
  

            
            
        





            
    
    
    
