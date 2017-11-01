import glob
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def print_pic(pic):
    plt.figure("show_pic")
    plt.imshow(pic)
    plt.show()
def draw_full_pic(pic,path):
    grayimg=cv2.cvtColor(pic,cv2.COLOR_RGB2GRAY)
    pic = cv2.inRange(grayimg, 0, 254)
   
    _, contours, _ = cv2.findContours(pic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pic2=cv2.drawContours(pic,contours,-1,(255,255,255),-1)
    cv2.imwrite(path,pic2)
    return pic2
#     cv2.imwrite("/home/yyy/Downloads/BOT/1.png",pic2)

def catch_roi(pic,pic_yuantu,j):
#     
    i=0
    grayimg=cv2.cvtColor(pic,cv2.COLOR_RGB2GRAY)
    pic = cv2.inRange(grayimg, 0, 254)
#     print_pic(pic)
    
#     print_pic(pic_yuantu)
    patch_hsv = cv2.cvtColor(pic_yuantu, cv2.COLOR_BGR2HSV)
#     print_pic(patch_hsv)
    lower_red = np.array([10,10,10])
    upper_red = np.array([240, 240, 240])
    pic_yuantu_thr = cv2.inRange(pic_yuantu, lower_red, upper_red)
#     print_pic(pic_yuantu)
    
    for x in range(0,2017,50):
        for y in range(0,2017,50):
            mask=pic[int(x):int(x+150),int(y):int(y+150)]
            white_pixel_cnt = cv2.countNonZero(mask)
            
#             print(x,y,white_pixel_cnt)
            #smaller more accurecy
            if white_pixel_cnt < ((150 * 150) * 0.2):
                mask_yuantu=pic_yuantu_thr[int(x):int(x+150),int(y):int(y+150)]
                white_pixel_cnt_yuantu = cv2.countNonZero(mask_yuantu)
                if white_pixel_cnt_yuantu > ((150 * 150) * 0.50):
                    i=i+1
                    mask_yuantu=pic_yuantu[int(x):int(x+150),int(y):int(y+150)]
                    cv2.imwrite("/home/yyy/Downloads/BOT/patch_pic/P_"+str(x)+"_"+str(y)+str(i)+"_"+str(j)+".jpg",mask_yuantu)
                    print(x,y)
#                 print_pic(mask)m
def get_nomal(pic):
    i=0
    for x in range(0,2017,100):
        for y in range(0,2017,100):
            mask=pic[int(x):int(x+150),int(y):int(y+150)]
            lower_red = np.array([10,10,10])
            upper_red = np.array([240, 240, 240])
            pic_yuantu_thr = cv2.inRange(mask, lower_red, upper_red)            
            white_pixel_cnt_yuantu = cv2.countNonZero(pic_yuantu_thr)
            if white_pixel_cnt_yuantu > ((150 * 150) * 0.1):
                i=i+1
                cv2.imwrite("/home/yyy/Downloads/BOT/patch_pic_normal/N_"+str(x)+"_"+str(y)+str(i)+"_"+str(i)+".jpg",mask)
                M=cv2.getRotationMatrix2D((75,75),180,1)
                mask2=cv2.warpAffine(mask,M,(150,150))
                cv2.imwrite("/home/yyy/Downloads/BOT/patch_pic_normal/N_"+str(x)+"_"+str(y)+str(i)+"_"+str(i)+"_2"+".jpg",mask2)
                
                          
                     


# img=cv2.imread('/home/yyy/Downloads/BOT/label_jpg/2017-06-09_18.08.16.ndpi.16.14788_15256.2048x2048.jpg') 
# # print_pic(img)
# pic_yuantu=cv2.imread('/home/yyy/Downloads/BOT/2017-06-09_18.08.16.ndpi.16.14788_15256.2048x2048.tiff') 
# print_pic(pic_yuantu)

# path='/home/yyy/Downloads/BOT/training2/non_cancer_subset00/'
# for file in glob.glob(path+'*tiff'):
#     img=cv2.imread(file) 
#     get_nomal(img)
#     







# #get pic cancer
# path='/home/yyy/Downloads/BOT/training2/cancer/'
# j=0
# for file in glob.glob(path+'*jpg'):
#     j=j+1
#     img=cv2.imread(file) 
# # 
#     pic_yuantu=cv2.imread(file[:-3]+"tiff") 
# #     print_pic(pic_yuantu)
#     catch_roi(img,pic_yuantu,j)





# #patch find
path_svg='/home/yyy/Downloads/BOT/2017-06-09_18.08.16.ndpi.16.14788_15256.2048x2048.png'
path='/home/yyy/Downloads/BOT/'
path='/home/yyy/Downloads/BOT/label_jpg/'
i=0
for file in glob.glob(path+'*jpg'):
    img=cv2.imread(file) 
    draw_full_pic(img,file)
    i=i+1
    print(file,i)


# not use function
#     a=pic.shape
#     empty=np.zeros((a[0],a[1]))
#     print_pic(empty)



    