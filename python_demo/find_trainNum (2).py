import os
import re
import pdb
import shutil
def addWord(theIndex,word,pagenumber): 
    theIndex.setdefault(word, [ ]).append(pagenumber)
kind_list=[]
d={}
# def addword(theIndex, word, pagenumber):  
#       if word in theIndex:  
#             theIndex[word].append(pagenumber)  
#       else:  
#             theIndex[word] = [pagenumber] 
train_list=[]
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

        for i in range(0,len(name_kind)):
#            if len(name_kind) >1 :
#                pdb.set_trace()
            try:
                name_kinds=name_kind[i]
            except:
                continue
            addWord(d,name_kinds,1)
            if name_kinds in kind_list:
                pass
            else:
                kind_list.append(name_kinds)

#        pdb.set_trace()
        if name_kind[0] == '6_vue_3':
            shutil.copyfile('/data2/datasets/video/logo/data/VOC2007/Annotations/'+files,'/data2/datasets/video/logo/data/VOC2007/vue/'+files)
            shutil.copyfile('/data2/datasets/video/logo/data/VOC2007/JPEGImages/'+files[:-3]+"jpg",'/data2/datasets/video/logo/data/VOC2007/vue/'+files[:-3]+"jpg")
print(len(kind_list))

for kind_lists in kind_list:
    print(kind_lists,len(d[kind_lists]))


