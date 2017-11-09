# *-coding:utf-8-*-
import os
import sys
import importlib
importlib.reload(sys)
import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import _csv
browser=webdriver.Firefox()
browser.get('http://www.meduniwien.ac.at/nephrogene/data/riskcalc/renal_v4/renal.html')
browser.find_element_by_id('ui-accordion-accordion-header-0').click()

# 
with open('C:/Users/yinyy/Desktop/PythonData1.csv','w',newline='') as csvfile:
    writer =_csv.writer(csvfile)
    writer.writerow(["UACR_mg/g","egfr","age","alive_with","death"]) 
    for UACR in range(1,300):
        for Age in range(55,84):
            for egfr in range(60,120):          
                browser.find_element_by_id('uacr_ren').send_keys(str(UACR))
                browser.find_element_by_id('uacr_g_ren').send_keys(Keys.SPACE)
   
                browser.find_element_by_id('egfr_ren').send_keys(str(egfr))
                browser.find_element_by_id('age_ren').send_keys(str(Age))
   
                browser.find_element_by_id('chkDisclaimer_ren').send_keys(Keys.SPACE)
                browser.find_element_by_id('female_one_ren').send_keys(Keys.SPACE)
   
                browser.find_element_by_id('calculate_button_ren').send_keys(Keys.ENTER)
            
                P1=browser.find_element_by_id('wrenal_ren').get_attribute('value')
                P2_death=browser.find_element_by_id('dead_ren').get_attribute('value')
      
                browser.find_element_by_id('uacr_ren').send_keys(Keys.CONTROL+'a')
                browser.find_element_by_id('uacr_ren').send_keys(Keys.DELETE)
                browser.find_element_by_id('egfr_ren').send_keys(Keys.CONTROL+'a')
                browser.find_element_by_id('egfr_ren').send_keys(Keys.DELETE)
                browser.find_element_by_id('age_ren').send_keys(Keys.CONTROL+'a')
                browser.find_element_by_id('age_ren').send_keys(Keys.DELETE)
                writer.writerow([UACR,egfr,Age,P1,P2_death])
      




