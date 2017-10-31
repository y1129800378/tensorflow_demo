

python3 draw_patch.py --path /home/yyy/Downloads/lymphoma/ --path_save_patch /home/yyy/Downloads/lymphoma_patch/ --size 150

mkdir /home/yyy/Downloads/lymphoma/bottleneck/
mkdir /home/yyy/Downloads/lymphoma/model/

python3 retrain.py --image_dir /home/yyy/Downloads/lymphoma_patch/ --bottleneck_dir /home/yyy/Downloads/lymphoma/bottleneck/ --how_many_training_steps 3500  --model_dir /home/yyy/workspace2/read_csv_and_draw_pic_py3/inception_tensorflow_donload/ --output_graph /home/yyy/Downloads/lymphoma/model/model.pb  --output_labels /home/yyy/Downloads/lymphoma/model/output_labels.txt

