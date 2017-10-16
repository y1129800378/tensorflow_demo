#!/bin/bash
cd /home/yyy/Desktop/case2_epi/subs/
for kfoldi in {1..5}
do
echo "doing fold $kfoldi"
/usr/local/caffe-caffe-0.15/build/tools/convert_imageset -shuffle -backend leveldb   /home/yyy/Desktop/case2_epi/subs/ /home/yyy/Desktop/case2_epi/train_data_package/train_w32_${kfoldi}.txt /home/yyy/Desktop/case2_epi/bin_data_save/DB_train_${kfoldi} &
/usr/local/caffe-caffe-0.15/build/tools/convert_imageset -shuffle -backend leveldb   /home/yyy/Desktop/case2_epi/subs/  /home/yyy/Desktop/case2_epi/train_data_package/test_w32_${kfoldi}.txt /home/yyy/Desktop/case2_epi/bin_data_save/DB_test_${kfoldi} &
done




FAIL=0
for job in `jobs -p`
do
    echo $job
    wait $job || let "FAIL+=1"
done




echo "number failed: $FAIL"

cd /home/yyy/Desktop/case2_epi/bin_data_save/

for kfoldi in {1..5}
do
echo "doing fold $kfoldi"
/usr/local/caffe-caffe-0.15/build/tools/compute_image_mean /home/yyy/Desktop/case2_epi/bin_data_save/DB_train_${kfoldi} DB_train_w32_${kfoldi}.binaryproto -backend leveldb  &
done



