from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf 
import glob
import re
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import scipy.misc
import glob

# with open(r'/home/yyy/Downloads/BOT/10_yuan_txt.txt','w') as txt_d:
#     i=0
#     for file in glob.glob('/home/yyy/Downloads/BOT/test/*_yuanshi.jpg'):  
#         
# #         print(file)
#         img=cv2.imread(file) 
#         grayimg=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# #     pic = cv2.inRange(grayimg, 0, 254)
#         white_pixel_cnt_yuantu = cv2.countNonZero(grayimg)
# #         print(white_pixel_cnt_yuantu)
#         if white_pixel_cnt_yuantu > ((2048 * 2048) * 0.10):
#             txt_d.write(file[29:-12]+".tiff "+"P")
#             i=i+1
#         else:
#             txt_d.write(file[29:-12]+".tiff "+"N")
#             i=i+1
#     print(str(i))


with open(r'/home/yyy/Downloads/BOT/0.000032_txt_huanghang.txt','w') as txt_d:
    i=0
    for file in glob.glob('/home/yyy/Downloads/BOT/test/*_yuanshi.jpg'):  
         
#         print(file)
        img=cv2.imread(file) 
        grayimg=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#     pic = cv2.inRange(grayimg, 0, 254)
        white_pixel_cnt_yuantu = cv2.countNonZero(grayimg)
#         print(white_pixel_cnt_yuantu)
        if white_pixel_cnt_yuantu > ((2048 * 2048) * 0.000032):
            txt_d.write(file[29:-12]+".tiff "+"P\r\n")
            i=i+1
        else:
            txt_d.write(file[29:-12]+".tiff "+"N\r\n")
            i=i+1
    print(str(i))
