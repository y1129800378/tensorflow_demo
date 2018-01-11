import mxnet as mx
from mxnet import ndarray as nd
import numpy as np
import cv2
def print_pic(img):
    cv2.namedWindow("Image")  
    cv2.imshow("Image", img)   
    cv2.waitKey (0)   
pic=cv2.imread("C:/Users/316/Desktop/roi.jpg")
pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY) 
# print(a)
print_pic(pic)

wight_x=pic.shape[1]
height_y=pic.shape[0]

# #23 pic[22]
# pic_np_tempSave=[]
# pic_np_tempSave_perline=[]
# pic_np=[]
# for y_height in range(0,height_y):
#     pic_np_tempSave=np.array(pic[y_height])
#     for elements in range(0,len(pic_np_tempSave)):
#         elements_content=pic_np_tempSave[elements]
#         pic_np_tempSave_perline.append(elements_content)
#     pic_np.append(pic_np_tempSave_perline) 
#     #clean
#     pic_np_tempSave_perline=[]  
#     pic_np_tempSave=[] 
# print("pic_np",len(pic_np))
# x0=[[pic_np]]

x0=[[pic]]

y0 = [[0,0,0,pic.shape[1]-1,pic.shape[0]-1]]
 
x0=nd.array(x0)
y0=nd.array(y0)
 
print(x0.shape,y0.shape)
 
x = mx.symbol.Variable('x')
y = mx.symbol.Variable('y')
 
print(x0)
 
 
# a=mx.symbol.ROIPooling(name="roi",data=x, rois=y, pooled_size=(pic.shape[0]*4,pic.shape[1]*4),spatial_scale=1)
a=mx.symbol.ROIPooling(name="roi",data=x, rois=y, pooled_size=(500,1000),spatial_scale=1)
e = a.bind(mx.cpu(), {'x': x0, 'y':y0})
 
z= e.forward()

z=z[0].asnumpy()
print("z",z[0][0])
print(len(z[0][0]))

ROIPooling_pic=[]

for i in range(0,len(z[0][0])):
    numbers = [ int(x) for x in z[0][0][i] ]
    ROIPooling_pic.append(numbers)
    a=np.array(ROIPooling_pic,dtype=np.uint8)

print_pic(a)