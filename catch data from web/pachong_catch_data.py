# *-coding:utf-8-*-
import re
import requests
import importlib
import sys
importlib.reload(sys)
import _csv
with open('C:/Users/yinyy/Desktop/Second_type111.csv','w',newline='') as csvfile:
    writer =_csv.writer(csvfile)
    writer.writerow(["gender","type","PRESENCE OF RETINOPATHY","duration","hba1c%","systolic","1_year","5_year","10_year"]) 
    url='http://retinarisk.com/calculator/bout_interactive.php'
    for PRESENCE in range(0,2):
        if PRESENCE==0:
            Static='Yes'      #0
        else:
            Static='No'       #1
        for gender in range(0,2):
            if gender==0:
                Static2='Male'     #0
            else:
                Static2='Female'   #1
         
            for hbac in range(0,13,2):
                for DURATION in range(0,40,4):
                    for SYSTOLIC in range(60,180,6):       
                        header={
                        'Accept':'*/*',
                        'Accept-Encoding':'gzip, deflate',
                        'Accept-Language':'zh-CN,zh;q=0.8',
                        'Connection':'keep-alive',
                        'Content-Length':'143',
                        'Content-Type':'application/x-www-form-urlencoded; charset=UTF-8',
                        'Cookie':'PHPSESSID=3i8ehti6j9s55ugp6tp7kgvar4; __utma=91566745.440636138.1497512826.1497953282.1498007595.3; __utmc=91566745; __utmz=91566745.1498007595.3.3.utmcsr=risk.is|utmccn=(referral)|utmcmd=referral|utmcct=/; _icl_current_language=en',
                        'Host':'retinarisk.com',
                        'Origin':'http://retinarisk.com',
                        'Referer':'http://retinarisk.com/risk-calculator/?lang=en',
                        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.86 Safari/537.36',
                        'X-Requested-With':'XMLHttpRequest'
                        }
                        post_data = {
                        'inter_gender':Static2,
                        'inter_type':'Type1',
                        'inter_dr':Static,
                        'duration':DURATION,
                        'hba1c':hbac,
                        'inter_hba1c_type':'%',
                        'abg':'',
                        'inter_abg_type':'mmol/L',
                        'systolic':SYSTOLIC,
                        'diastolic':'40'}
                        try:  
                            post_html = requests.post(url,data=post_data,headers=header)
                            post_html=post_html.content
                            post_html=post_html.decode('UTF-8')
                            catch_data=re.findall('riskPercentage":(.*?),"r',post_html)
#                           catch_data2=re.search('riskPercentage":(.*?),"r',post_html)
                            catch_data2=catch_data[0]
                            catch_data2=catch_data2.strip().lstrip().rstrip(']')
                            Str1=catch_data2[1:catch_data2.index(',')]          #第一年
                            catch_data3=catch_data2[catch_data2.index(',')+1:len(catch_data2)]
                            Str2=catch_data3[0:catch_data3.index(',')]           #第五年
                            Str3=catch_data3[catch_data3.index(',')+1:len(catch_data3)]   #第十年
                            writer.writerow([gender,'type1',PRESENCE,DURATION,hbac,SYSTOLIC,Str1,Str2,Str3]) 
#                           writer.writerow([catch_data,catch_data2,catch_data3,Str1,Str2,Str3])
                        except:
                            print("a problem",'Static=',Static,'Static2=',Static2,'hbac=',hbac,'DURATION=',DURATION,'SYSTOLIC=',SYSTOLIC)
                            pass
print("finish")