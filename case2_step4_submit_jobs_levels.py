#!/usr/bin/env python
import os
import glob
import subprocess
import commands


basefile_solver='/home/yyy/Desktop/case2_epi/bin_data_save/BASE-alexnet_solver_ada.prototxt'
basefile_train='/home/yyy/Desktop/case2_epi/bin_data_save/BASE-alexnet_traing_32w_db.prototxt'
basefile_qsub='/home/yyy/Desktop/case2_epi/bin_data_save/BASE-qsub.pbs'
workingdir='/home/yyy/Desktop/case2_epi/bin_data_save/'


f = open(basefile_solver, 'r')
template_solver_text=f.read()
f.close()


f = open(basefile_train, 'r')
template_train_text=f.read()
f.close()


f = open(basefile_qsub, 'r')
template_qsub=f.read()
f.close()



os.chdir(workingdir)

for kfoldi in xrange(1,6):
    stata,out=commands.getstatusoutput("wc -l /home/yyy/Desktop/case2_epi/train_data_package/test_w32_%d.txt"%(kfoldi))
    print ("11first_out:",out,"stata:",stata)
    numiter=int(out[1].split()[0])/128
    specific_solver_text=template_solver_text % {'kfoldi': kfoldi,'numiter': numiter}
    specific_train_text=template_train_text %  {'kfoldi': kfoldi}
    specific_qsub=template_qsub %   {'kfoldi': kfoldi}	
    foutname=basefile_solver
    foutname=foutname.replace('BASE',"%d" %(kfoldi))
    fout = open(foutname,'w')
    fout.write(specific_solver_text)
    fout.close()				
    foutname=basefile_train
    foutname=foutname.replace('BASE',"%d" %(kfoldi))
    fout = open(foutname,'w')
    fout.write(specific_train_text)
    fout.close()			
		#sp = subprocess.Popen(["qsub",""], shell=False, stdin=subprocess.PIPE)
		#print sp.communicate(specific_qsub)
		#sp.wait()
    stata,out=commands.getstatusoutput("/usr/local/caffe-caffe-0.15/build/tools/caffe train --solver=/home/yyy/Desktop/case2_epi/bin_data_save/%d-alexnet_solver_ada.prototxt"%(kfoldi))
    print ("second_out:",out,"stata:",stata)
