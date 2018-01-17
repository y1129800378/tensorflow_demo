import os 
import shutil
pic_need_select="C:/Users/316/Desktop/down/tiantianPT_savePICfromVideo"
picGet_savePath="C:/Users/316/Desktop/down/tiantianPT_label"
if not os.path.exists(picGet_savePath):
    os.makedirs(picGet_savePath)
for files in os.listdir(pic_need_select):
    if (files[-7:-4] == "000"):
        shutil.copyfile(pic_need_select+"/"+files,picGet_savePath+"/"+files)