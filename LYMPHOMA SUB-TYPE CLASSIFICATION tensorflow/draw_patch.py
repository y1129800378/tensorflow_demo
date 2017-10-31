import cv2
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def print_pic(pic):
    plt.figure("show_pic")
    plt.imshow(pic)
    plt.show()

parser=argparse.ArgumentParser(description='Download and draw pic.')
parser.add_argument('--path',default='/home/yyy/Downloads/lymphoma/')
parser.add_argument('--path_save_patch',default='/home/yyy/Downloads/lymphoma_patch/')
parser.add_argument('--size',type=int,default=150)
args=parser.parse_args()
path=args.path
path_save_patch=args.path_save_patch
size=args.size

isExists=os.path.exists(path_save_patch)
if not isExists:
    os.makedirs(path_save_patch) 
isExists=os.path.exists(path_save_patch+"/CLL/")
if not isExists:
    os.makedirs(path_save_patch+"/CLL/") 
isExists=os.path.exists(path_save_patch+"/FL/")
if not isExists:
    os.makedirs(path_save_patch+"/FL/") 
isExists=os.path.exists(path_save_patch+"/MCL/")
if not isExists:
    os.makedirs(path_save_patch+"/MCL/") 
    
for root,dirs,files in os.walk(path+"/CLL/"):
    for pic in files:
        print(pic)
        img=cv2.imread(path+"/CLL/"+pic)
        size_pic=img.shape
        width=size_pic[0]
        height=size_pic[1]
        for x in range(0,width,size):
            for y in range(0,height,size):
                if (height-y)<size or (width-x)<size:
                    pass
                else:
                    mask=img[x:x+size,y:y+size]
#                     print_pic(mask)
                    m=cv2.getRotationMatrix2D((size/2,size/2),90,1)
                    mask_change=cv2.warpAffine(mask,m,(size,size))
#                     print_pic(mask_change)
                    cv2.imwrite(path_save_patch+"/CLL/"+pic[:-4]+"_"+str(x)+"_"+str(y)+"_1.jpg",mask)
                    cv2.imwrite(path_save_patch+"/CLL/"+pic[:-4]+"_"+str(x)+"_"+str(y)+"_2.jpg",mask_change)
                    
for root,dirs,files in os.walk(path+"/FL/"):
    for pic in files:
        print(pic)

        try:
            img=cv2.imread(path+"/FL/"+pic)
            size_pic=img.shape
            width=size_pic[0]
            height=size_pic[1]
            for x in range(0,width,size):
                for y in range(0,height,size):
                    if (height-y)<size or (width-x)<size:
                        pass
                    else:
                        mask=img[x:x+size,y:y+size]
#                        print_pic(mask)
                        m=cv2.getRotationMatrix2D((size/2,size/2),90,1)
                        mask_change=cv2.warpAffine(mask,m,(size,size))
#                     print_pic(mask_change)
                        cv2.imwrite(path_save_patch+"/FL/"+pic[:-4]+"_"+str(x)+"_"+str(y)+"_1.jpg",mask)
                        cv2.imwrite(path_save_patch+"/FL/"+pic[:-4]+"_"+str(x)+"_"+str(y)+"_2.jpg",mask_change)   
        except:
            print(pic)
        
                    
for root,dirs,files in os.walk(path+"/MCL/"):
    for pic in files:
        print(pic)
        img=cv2.imread(path+"/MCL/"+pic)
        size_pic=img.shape
        width=size_pic[0]
        height=size_pic[1]
        for x in range(0,width,size):
            for y in range(0,height,size):
                if (height-y)<size or (width-x)<size:
                    pass
                else:
                    mask=img[x:x+size,y:y+size]
#                     print_pic(mask)
                    m=cv2.getRotationMatrix2D((size/2,size/2),90,1)
                    mask_change=cv2.warpAffine(mask,m,(size,size))
#                     print_pic(mask_change)
                    cv2.imwrite(path_save_patch+"/MCL/"+pic[:-4]+"_"+str(x)+"_"+str(y)+"_1.jpg",mask)
                    cv2.imwrite(path_save_patch+"/MCL/"+pic[:-4]+"_"+str(x)+"_"+str(y)+"_2.jpg",mask_change)                       
                    
                    
                    