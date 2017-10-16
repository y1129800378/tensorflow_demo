#!/bin/bash
#PBS -l walltime=320:00:00
#PBS -l nodes=1:ppn=1:gpus=1:exclusive_process
#PBS -M axj232@case.edu
#PBS -q gpufermi
#PBS -N %(kfoldi)d_nuclei_level
#PBS -m bea
#PBS -o "%(kfoldi)d_nuclei_level_output.out"
#PBS -j oe


output=`nvidia-smi -q -d "PIDS" --id=0`
if ! [[ $output == *"None"* ]];
then
   export CUDA_VISIBLE_DEVICES=1
fi



cd /home/yyy/Desktop/case2_epi/bin_data_save

/usr/local/caffe-caffe-0.15/build/tools/caffe train --solver=/home/yyy/Desktop/case2_epi/bin_data_save/%(kfoldi)d-alexnet_solver_ada.prototxt


exit


