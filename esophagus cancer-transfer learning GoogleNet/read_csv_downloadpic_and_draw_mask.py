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

def read_csv_download_pic_and_draw_mask(csv_file,mask_save_dir):
    isExists=os.path.exists(mask_save_dir)
    if not isExists:
        os.makedirs(mask_save_dir)       
#     isExists=os.path.exists(mask_save_dir+"/NBI")
#     if not isExists:
#         os.makedirs(mask_save_dir+"/NBI")     
#     isExists=os.path.exists(mask_save_dir+"/NBI_magnification")
#     if not isExists:
#         os.makedirs(mask_save_dir+"/NBI_magnification")             
#     isExists=os.path.exists(mask_save_dir+"/lodine_staining")
#     if not isExists:
#         os.makedirs(mask_save_dir+"/lodine_staining")              
#     isExists=os.path.exists(mask_save_dir+"/white_light")
#     if not isExists:
#         os.makedirs(mask_save_dir+"/white_light")   
#     isExists=os.path.exists(mask_save_dir+"/white_light_magnification")
#     if not isExists:
#         os.makedirs(mask_save_dir+"/white_light_magnification")  

    f = open(csv_file)
    flag=0
    for row in f:
        if(flag==0):  #the first line is header,so Abandon it
            flag=1 
        else:
            csv_perline_str=str(row)
            
            data1_num=csv_perline_str.find('|')
            data2_num=csv_perline_str.find('|',data1_num+1)
            data3_num=csv_perline_str.find('|',data2_num+1)
            data4_num=csv_perline_str.find('|',data3_num+1)
            data5_num=csv_perline_str.find('|',data4_num+1)
            data6_num=csv_perline_str.find('|',data5_num+1)
            data7_num=csv_perline_str.find('|',data6_num+1)
            
            csv_perline_str_data1_imageId=csv_perline_str[0:data1_num] 
            csv_perline_str_data2_bucket=csv_perline_str[data1_num+1:data2_num] 
            csv_perline_str_data3_key=csv_perline_str[data2_num+1:data3_num] 
            csv_perline_str_data4_shape_type=csv_perline_str[data3_num+1:data4_num] 
            csv_perline_str_data5_shape_Desc=csv_perline_str[data4_num+1:data5_num] 
            csv_perline_str_data6_comment=csv_perline_str[data5_num+1:data6_num] 
            csv_perline_str_data7_tag=csv_perline_str[data6_num+1:data7_num] 
    
            point_num=csv_perline_str_data5_shape_Desc.count(',')
            point_x_location=[]
            point_y_location=[]
            comma_location=0        
            space_location=0
            Next_space_location=0
            Next_comma_location=0
            left_point_x=0
            left_point_y=0
            wight_c=0
            height_c=0
            if csv_perline_str_data4_shape_type=="4":
                for i in range(point_num):
                    if i==point_num-1:
                        Next_comma_location=csv_perline_str_data5_shape_Desc.find(',',comma_location+1)
                        Next_space_location=len(csv_perline_str_data5_shape_Desc)
                    else:
                        Next_comma_location=csv_perline_str_data5_shape_Desc.find(',',comma_location+1)
                        Next_space_location=csv_perline_str_data5_shape_Desc.find(' ',space_location+1)
                    point_x_location.append(float(csv_perline_str_data5_shape_Desc[space_location:Next_comma_location]))
                    point_y_location.append(float(csv_perline_str_data5_shape_Desc[Next_comma_location+1:Next_space_location]))
                    comma_location=Next_comma_location
                    space_location=Next_space_location    
            if csv_perline_str_data4_shape_type=="1" or csv_perline_str_data4_shape_type=="2" or csv_perline_str_data4_shape_type=="3":
                point_num=4 
                space_location=csv_perline_str_data5_shape_Desc.find(' ',1)
                left_point_x=float(csv_perline_str_data5_shape_Desc[0:space_location])
                Next_space_location=csv_perline_str_data5_shape_Desc.find(' ',space_location+1)
                left_point_y=float(csv_perline_str_data5_shape_Desc[space_location+1:Next_space_location])
                space_location=csv_perline_str_data5_shape_Desc.find(' ',Next_space_location+1)
                wight_c=float(csv_perline_str_data5_shape_Desc[Next_space_location+1:space_location])
                Next_space_location=csv_perline_str_data5_shape_Desc.find(' ',space_location+1)
                height_c=float(csv_perline_str_data5_shape_Desc[space_location+1:])
    #             if csv_perline_str_data4_shape_type=="1":
    
    #             if csv_perline_str_data4_shape_type=="2" or csv_perline_str_data4_shape_type=="3":
                point_x_location.append(left_point_x)
                point_x_location.append(left_point_x+wight_c)
                point_x_location.append(left_point_x+wight_c)
                point_x_location.append(left_point_x)
                point_y_location.append(left_point_y)
                point_y_location.append(left_point_y)
                point_y_location.append(left_point_y+height_c)
                point_y_location.append(left_point_y+height_c)
            pic_save_path=''
            if csv_perline_str_data7_tag=="":
                isExists=os.path.exists(mask_save_dir+"/no_label")
                if not isExists:
                    os.makedirs(mask_save_dir+"/no_label")  
                pic_save_path=mask_save_dir+"/no_label/"
            else: 
                isExists=os.path.exists(mask_save_dir+"/"+csv_perline_str_data7_tag)
                if not isExists:
                    os.makedirs(mask_save_dir+"/"+csv_perline_str_data7_tag) 
                pic_save_path=mask_save_dir+"/"+csv_perline_str_data7_tag+"/"
            
            
            if(os.path.exists(pic_save_path+"/"+csv_perline_str_data3_key)):
                pass
            else:
                down_pic=requests.get("https://s3.cn-north-1.amazonaws.com.cn/"+csv_perline_str_data2_bucket+"/"+csv_perline_str_data3_key)
                save_pic=open(pic_save_path+csv_perline_str_data3_key,"wb")
                save_pic.write(down_pic.content)
                save_pic.close()
            
            img=cv2.imread(pic_save_path+csv_perline_str_data3_key) 
            size_pic=img.shape
            width=size_pic[0]
            height=size_pic[1]
            out_c=np.zeros(shape=(width,height))   
            xld_contours=[]
            for i in range(point_num):
                xld_contours.append([[point_x_location[i],point_y_location[i]]])
            draw_ROI_contours=[np.array(xld_contours,dtype=int)]
            pic2=cv2.drawContours(out_c,draw_ROI_contours,-1,(255,255,255),-1)
            cv2.imwrite(pic_save_path+csv_perline_str_data3_key[:-4]+"_ROI.jpg",pic2)
    #         print_pic(pic2)

            print(csv_perline_str_data3_key)
       
            
parser=argparse.ArgumentParser(description='Download and draw pic.')
parser.add_argument('--csv_file',default='/home/yyy/workspace2/prod_dg_annotation_20171012.csv')
parser.add_argument('--mask_save_dir',default='/home/yyy/workspace2/pic_data/')
args=parser.parse_args()
csv_file=args.csv_file
mask_save_dir=args.mask_save_dir
# read_csv_download_pic_and_draw_mask('/home/yyy/workspace2/prod_dg_annotation_20171012.csv',"/home/yyy/workspace2/pic_data2/")        
read_csv_download_pic_and_draw_mask(csv_file,mask_save_dir)
            
            
        





            
    
    
    