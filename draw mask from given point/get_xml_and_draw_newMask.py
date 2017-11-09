import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from numpy import set_printoptions

def print_pic(pic):
    plt.figure("show_pic")
    plt.imshow(pic)
    plt.show()

    
def draw_mask_multiRegion(x_allRegion_location,y_allRegion_location,size_wight,size_height,mask_save_place):
    masks=np.zeros(shape=(size_wight,size_height))
    cv2.imwrite(mask_save_place,masks)
    for i in range(len(x_allRegion_location)):
        masks=cv2.imread(mask_save_place)
        x_position_list=x_allRegion_location[i]
        y_position_list=y_allRegion_location[i]
        xld_contours=[]
        for j in range(len(x_position_list)):
            xld_contours.append([[x_position_list[j],y_position_list[j]]])
        xld_contours=[np.array(xld_contours,dtype=int)]  
        masks=cv2.drawContours(masks,xld_contours,-1,(255,255,255),-1)
#         print_pic(masks)
        cv2.imwrite(mask_save_place,masks)   
 
def get_location_file(readPic_path):    
    
    img=cv2.imread(readPic_path)

    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    kernel=np.ones((60,60),np.uint8)
    open=cv2.erode(img,kernel,iterations=1)
    img=cv2.dilate(open,kernel,iterations=1)

    _,contour,_=cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    x_location=[]
    y_location=[]
    x_allRegion_location=[]
    y_allRegion_location=[]

    for i in range(len(contour)):
#which x,y 
        for j in range(len(contour[i])):
            x_location.append(contour[i][j][0][0])
            y_location.append(contour[i][j][0][1])
        x_allRegion_location.append(x_location)
        y_allRegion_location.append(y_location)
        
        x_location=[]
        y_location=[]
    
    draw_mask_multiRegion(x_allRegion_location,y_allRegion_location,img.shape[0],img.shape[1],"/home/yyy/workspace2/prediction_test3/mmexport1502715474300_new_pic.jpg")

get_location_file("/home/yyy/workspace2/prediction_test2/mmexport1502715474300.jpg_Predicte.jpg")


with open(r'/home/yyy/workspace2/prediction_test3/count_nuclei.xml','w') as txt_data:
# with open(r'/home/yyy/Desktop/count_nuclei.xml','w') as txt_data:
    img=cv2.imread("/home/yyy/workspace2/prediction_test2/mmexport1502715474300.jpg_Predicte.jpg")

    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    kernel=np.ones((10,10),np.uint8)
    open=cv2.erode(img,kernel,iterations=1)
    img=cv2.dilate(open,kernel,iterations=1)

    _,con,_=cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    txt_data.write('''<?xml version="1.0"?>'''+"\r\n")
    txt_data.write(''' <ASAP_Annotations>'''+"\r\n")
    txt_data.write(''' <Annotations>'''+"\r\n")
    for i in range(len(con)):
        txt_data.write(''' <Annotation Name="_'''+str(i)+'''" Type="Polygon" PartOfGroup="_0" Color="#F4FA58">'''+"\r\n")
        txt_data.write(''' <Coordinates>'''+"\r\n")    
     
        order=0
        for j in range(len(con[i])):
            txt_data.write(''' <Coordinate Order="'''+str(order)+'''" X="'''+str(con[i][j][0][0])+'''" Y="'''+str(con[i][j][0][1])+'''" /> '''+"\r\n") 
            order=order+1
#             print(x,y)    
        txt_data.write(''' </Coordinates>'''+"\r\n") 
        txt_data.write(''' </Annotation>'''+"\r\n")        
        
    txt_data.write(''' </Annotations>'''+"\r\n")  
    txt_data.write(''' <AnnotationGroups>'''+"\r\n")             
    txt_data.write(''' <Group Name="_0" PartOfGroup="None" Color="#00ff00">'''+"\r\n")  
    txt_data.write(''' <Attributes />'''+"\r\n")  
    txt_data.write(''' </Group>'''+"\r\n")   
    txt_data.write(''' </AnnotationGroups>'''+"\r\n")   
    txt_data.write(''' </ASAP_Annotations>'''+"\r\n")        
        