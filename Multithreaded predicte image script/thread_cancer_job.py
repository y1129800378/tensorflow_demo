import threading
from threading import Timer
import time,os
# import random
import glob
import argparse
import os

def read_txt_list():
    img_list=open(txt_path)
    txt_content=img_list.readlines()
    img_list.close()
    content=[]
    for line in txt_content:
        content.append(line[:-1])
    return content
def write_txt_list(list_content):
    img_list=open(txt_path,'a')
    img_list.write(str(list_content)+'\n')
    img_list.close()

class load_pic_thread(threading.Thread):
    def run(self):
        while True:
#             print('load_pic_thread is %s threads is start'%(threading.current_thread())) 
            gSem.acquire()
            for pic_file in glob.glob(img_path):
                if pic_file in need_prediction_pic_list:
#                     print("not add tuple in list",need_prediction_pic_list)
                    pass
                else:
                    if pic_file not in have_been_prediction_pic_list:
                        need_prediction_pic_list.append(pic_file)
#                         write_txt_list(pic_file)
                        print("add tuple in list ,the list can be seen as follow:",need_prediction_pic_list)
            if len(need_prediction_pic_list)>0:
                gSem.notify_all()
                gSem.wait()
            print("now need prediction pic list is %s"%(need_prediction_pic_list))
#             time.sleep(0.1)
            gSem.release()
#             print('load_pic_thread is %s threads is drop out.....'%(threading.current_thread())) 
            
class run_prediction(threading.Thread):            
    def run(self):
        while True:
#             print('run_prediction is %s threads is start'%(threading.current_thread()))
#             print('need prediction pic list is:',need_prediction_pic_list)
            gSem.acquire()
            if(len(need_prediction_pic_list)==0):
                gSem.notify_all()
                gSem.wait()
            file_need_prediction=need_prediction_pic_list[0]
            prediction(file_need_prediction)
            have_been_prediction_pic_list.append(file_need_prediction)
            write_txt_list(file_need_prediction)
            need_prediction_pic_list.remove(file_need_prediction)
            gSem.notify_all()
            gSem.wait()
            gSem.release()
#             print('run_prediction is %s threads is drop out.....'%(threading.current_thread())) 
def prediction(list):
    print("start_do_prediction %s please wait......... "%(list))
    time.sleep(3)
    print("end_do_prediction")    
    
               
def main_function():
    parser=argparse.ArgumentParser(description='')
    parser.add_argument('--path',default='/home/yyy/Downloads/test/')
    parser.add_argument('--format',default='jpg')
    args=parser.parse_args()
    path=args.path
    format=args.format
    global need_prediction_pic_list
    global have_been_prediction_pic_list
    global gSem
    global img_path
    global txt_path
    img_path=path+'*'+format
    txt_path=path+"predicted images.txt"
    need_prediction_pic_list=[]
    isExists=os.path.exists(txt_path)
    if isExists:
        have_been_prediction_pic_list=read_txt_list()
    else:
        have_been_prediction_pic_list=[]
#     gSem=threading.Semaphore(2)
    gSem=threading.Condition()
    
    load_pic_thread().start()
    run_prediction().start()
    

if __name__=='__main__':

    main_function()
#     

    
