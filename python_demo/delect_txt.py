import random

big_txt="C:/Users/316/Desktop/trainval.txt"
small_txt="C:/Users/316/Desktop/momo.txt"
final_txt=big_txt

big_txt_list=[]
small_txt_list=[]
for lines in open(big_txt):  
    if (lines[-2:] == "\n\r") or (lines[-2:] == "\r\n"):
        big_txt_list.append(lines[:-2])
    elif(lines[-1:] == "\n"):
        big_txt_list.append(lines[:-1])
    else:
        print("Read big txt failed!")
        break
        
for lines2 in open(small_txt):  
    if (lines2[-2:] == "\n\r") or (lines2[-2:] == "\r\n"):
        small_txt_list.append(lines2[:-2])
    elif(lines2[-1:] == "\n"):
        small_txt_list.append(lines2[:-1])
    else:
        print("Read small txt failed!")
        break
for _ in range(0,len(big_txt_list)):
    num = int(random.uniform(0, len(big_txt_list)))
    item = big_txt_list.pop(len(big_txt_list)-1)    
    big_txt_list.insert(num, item)
    
num_delet=0
print("big txt lenth:",len(big_txt_list))
print("small txt lenth:",len(small_txt_list))

with open(final_txt,"w") as txt:
    for write_files in big_txt_list:
        if write_files not in small_txt_list :
            txt.write(write_files+"\n")
        else:
            num_delet=num_delet+1
    print("Delet number:",num_delet)