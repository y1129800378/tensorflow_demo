python3 read_csv_downloadpic_and_draw_mask.py --csv_file /home/yyy/workspace2/prod_dg_annotation_20171012.csv --mask_save_dir /home/yyy/workspace2/pic_data/ 

echo "download picture and draw mask success"
python3 from_download_get_patch_for_train.py --path /home/yyy/workspace2/pic_data/ --path_get_patch /home/yyy/workspace2/get_patch/ --size 120

echo "catch patch success"
echo "begin train model"

python3 retrain.py --image_dir /home/yyy/workspace2/get_patch/NBI/ --bottleneck_dir /home/yyy/workspace2/get_patch/NBI/bottleneck/ --how_many_training_steps 1500  --model_dir /home/yyy/workspace2/read_csv_and_draw_pic_py3/inception_tensorflow_donload/ --output_graph /home/yyy/workspace2/get_patch/NBI/output_graph.pb --output_labels /home/yyy/workspace2/get_patch/NBI/output_labels.txt

