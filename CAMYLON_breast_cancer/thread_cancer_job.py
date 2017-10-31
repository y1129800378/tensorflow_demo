import threading
from threading import Timer
import time,os
import random
import glob

class load_pic_thread(threading.Thread):
    def run(self):
        while True:
            print('load_pic_thread is %s threads is start'%(threading.current_thread())) 
            gSem.acquire()
            for pic_file in glob.glob('/home/yyy/Downloads/test/*jpg'):
                if pic_file in need_prediction_pic_list:
                    print("not add tuple in list",need_prediction_pic_list)
                else:
                    if pic_file not in have_been_prediction_pic_list:
                        need_prediction_pic_list.append(pic_file)
                        print("add tuple in list!!!!",need_prediction_pic_list)
            if len(need_prediction_pic_list)>0:
                gSem.notify_all()
                gSem.wait()
            print("now need_prediction_pic_list is %s ...!!!"%(need_prediction_pic_list))
            time.sleep(0.5)
            gSem.release()
            print('load_pic_thread is %s threads is drop out.....'%(threading.current_thread())) 
            
class run_prediction(threading.Thread):            
    def run(self):
        while True:
            print('run_prediction is %s threads is start'%(threading.current_thread()))
            print('need_prediction_pic_list',need_prediction_pic_list)
            gSem.acquire()
            if(len(need_prediction_pic_list)==0):
                gSem.notify_all()
                gSem.wait()
            file_need_prediction=need_prediction_pic_list[0]
            prediction(file_need_prediction)
            have_been_prediction_pic_list.append(file_need_prediction)
            need_prediction_pic_list.remove(file_need_prediction)
            gSem.notify_all()
            gSem.wait()
            gSem.release()
            print('run_prediction is %s threads is drop out.....'%(threading.current_thread())) 
def prediction(list):
    print("start_do_prediction %s"%(list))
    time.sleep(3)
    print("end_do_prediction")    
    
               
def main_function():
    global need_prediction_pic_list
    global have_been_prediction_pic_list
    global gSem
    need_prediction_pic_list=[]
    have_been_prediction_pic_list=[]
#     gSem=threading.Semaphore(2)
    gSem=threading.Condition()
    
    load_pic_thread().start()
    run_prediction().start()
    

if __name__=='__main__':
    
    main_function()
    



