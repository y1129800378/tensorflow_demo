# _*_coding:utf-8_*_
import os
import re
import glob
# import cv2
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
 
def catch_pic_for_train():
    path='C:/Users/yinyy/Desktop/新建文件夹'
    path_get_patch='C:/Users/yinyy/Desktop/新建文件夹2'
    isExists=os.path.exists(path)
    if not isExists:
        print("dont find download data")
        return       
    ls = os.listdir(path)
    for i in ls:
        print(i)
        if(i=="NBI"):
            pic_ls = os.listdir(path+"/"+"NBI")
            print(pic_ls)
        elif(i=="NBI,放大"):
            pic_ls = os.listdir(path+"/"+"NBI,放大")
            print(pic_ls)
        elif(i=="NBI,放大,低倍"):
            pic_ls = os.listdir(path+"/"+"NBI,放大,低倍")
            print(pic_ls)            
        elif(i=="NBI,近景"):
            pic_ls = os.listdir(path+"/"+"NBI,近景")
            print(pic_ls)             
        elif(i=="NBI,远景"):
            pic_ls = os.listdir(path+"/"+"NBI,远景")
            print(pic_ls) 
        elif(i=="no_label"):
            pic_ls = os.listdir(path+"/"+"no_label")
            print(pic_ls)      
        elif(i=="白光"):
            pic_ls = os.listdir(path+"/"+"白光")
            print(pic_ls)    
        elif(i=="白光,近景"):
            pic_ls = os.listdir(path+"/"+"白光,近景")
            print(pic_ls)  
        elif(i=="白光,远景"):
            pic_ls = os.listdir(path+"/"+"白光,远景")
            print(pic_ls)        
        elif(i=="碘染色"):
            pic_ls = os.listdir(path+"/"+"碘染色")
            print(pic_ls)  
                  
parser=argparse.ArgumentParser(description='Download and draw pic.')
parser.add_argument('--csv_file',default='/home/yyy/workspace2/prod_dg_annotation_20171012.csv')
parser.add_argument('--mask_save_dir',default='/home/yyy/workspace2/pic_data/')
args=parser.parse_args()
csv_file=args.csv_file
mask_save_dir=args.mask_save_dir

catch_pic_for_train()
# read_csv_download_pic_and_draw_mask('/home/yyy/workspace2/prod_dg_annotation_20171012.csv',"/home/yyy/workspace2/pic_data2/")        

            
            
        





            
    
    
    