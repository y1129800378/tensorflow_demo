import os
import re
import pdb
import random
# def addword(theIndex, word, pagenumber):  
#       if word in theIndex:  
#             theIndex[word].append(pagenumber)  
#       else:  
#             theIndex[word] = [pagenumber] 

def addWord(theIndex,word,pagenumber): 
    theIndex.setdefault(word, [ ]).append(pagenumber)

dicts={}
train_list = []
select_list = []
rm_list = []
for lines in open("/data2/datasets/video/logo/data/VOC2007/ImageSets/Main/trainval.txt") :
    if (lines[-2:] == "\n\r") or (lines[-2:] == "\r\n"):
        train_list.append(lines[:-2])
    else:
        train_list.append(lines[:-1])
        
for files in os.listdir("/data2/datasets/video/logo/data/VOC2007/Annotations/"):
    if files[:-4] in train_list:
        f = open('/data2/datasets/video/logo/data/VOC2007/Annotations/'+files,'r')
        txts=f.read()
        name_kind = re.findall('<name>(.*?)</name>',txts)

        if name_kind[0] == '8_changba_1':  #13_huajiao_4   31_yinyuetai_1
           # pdb.set_trace()
            ID_num=files[:-4].find('_')
            ID=files[:-4][:ID_num]
            addWord(dicts,ID,files[:-4])

        if name_kind[0] == '13_huajiao_4':  #13_huajiao_4   31_yinyuetai_1
            ID_num=files[:-4].find('_')
            ID=files[:-4][:ID_num]
            addWord(dicts,ID,files[:-4])
#
        if name_kind[0] == '31_yinyuetai_1':  #13_huajiao_4   31_yinyuetai_1
            ID_num=files[:-4].find('_')
            ID=files[:-4][:ID_num]
            addWord(dicts,ID,files[:-4])

for list_files in dicts:
    if len(dicts[list_files]) > 4:
#        pdb.set_trace() 
        tp_list =  dicts[list_files]
        del tp_list[1]
        del tp_list[2]
        rm_list.extend(tp_list)
print(len(train_list)-len(rm_list))
final_list=list(set(train_list).difference(set(rm_list)))
print(len(final_list))     


for _ in range(0,len(final_list)):
    num = int(random.uniform(0, len(final_list)))
    item = final_list.pop(len(final_list)-1)    
    final_list.insert(num, item)

print(len(final_list))
print(len(rm_list))

with open("/data2/datasets/video/logo/data/VOC2007/ImageSets/trainval.txt","w") as txt:
    for files in final_list:
        txt.write(files+'\n')




