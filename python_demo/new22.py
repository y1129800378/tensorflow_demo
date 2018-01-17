import os
import re

def addWord(theIndex,word,pagenumber): 
    theIndex.setdefault(word, [ ]).append(pagenumber)
kind_list=[]
d={}
for file in os.listdir("/data2/datasets/video/logo/data/VOC2007/Annotations/"):
    f = open('/data2/datasets/video/logo/data/VOC2007/Annotations/'+file,'r')
    txts=f.read()
    name_kind = re.findall('<name>(.*?)</name>',txts)
    try:
        name_kind=name_kind[0]
    except:
        continue
    addWord(d,name_kind,1)
    if name_kind in kind_list:
        pass
    else:
        kind_list.append(name_kind)
print(len(kind_list))

for kind_lists in kind_list:
    print(kind_lists,len(d[kind_lists]))
