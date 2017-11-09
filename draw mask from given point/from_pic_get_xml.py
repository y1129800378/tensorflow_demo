import cv2
import numpy as np
import argparse
parser=argparse.ArgumentParser(description='get xml file.')
parser.add_argument('--path_saveXml',default='/home/yyy/workspace2/prediction_test3/count_nuclei.xml')
parser.add_argument('--path_readPic',default='/home/yyy/workspace2/prediction_test2/mmexport1502715474300.jpg_Predicte.jpg')
args=parser.parse_args()
path_readPic=args.path_readPic
path_saveXml=args.path_saveXml


with open(path_saveXml,'w') as txt_data:
    img=cv2.imread(path_readPic)

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
        