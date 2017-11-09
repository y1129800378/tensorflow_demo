import cv2
import numpy as np

def draw_mask(x_position_list,y_position_list,size_wight,size_height,mask_save_place):
    xld_contours=[]
    mask=np.zeros(shape=(size_wight,size_height))
    for i in range(len(x_position_list)):
        xld_contours.append([[x_position_list[i],y_position_list[i]]])
    draw_mask_contours=[np.array(xld_contours,dtype=int)]  
    mask=cv2.drawContours(mask,draw_mask_contours,-1,(255,255,255),-1)
    cv2.imwrite(mask_save_place+"/mask.jpg",mask)
    
#for example:
x_position_list=[5,10,5,0]
y_position_list=[0,5,10,5]
draw_mask(x_position_list,y_position_list,21,21,"/home/yyy/workspace2/case7_lym_classification/")
    
    