训练模型时使用run_training.sh脚本
其中：
python3 read_csv_downloadpic_and_draw_mask.py --csv_file 加csv读取地址 --mask_save_dir 加下载图像储存位置 
python3 from_download_get_patch_for_train.py -- 加下载图像储存位置  --path_get_patch 加切片储存位置--size 切片尺寸
python3 retrain.py --image_dir 加切片储存位置(具体到子文件夹) --bottleneck_dir 打包图像存储位置 --how_many_training_steps 1训练步数  --model_dir 预训练模型存储位置 --output_graph 迁移学习后模型存储位置 --output_labels 标签存储位置

预测样本时使用predicte.sh脚本
其中：
python3 predicte.py --model_save_path 迁移学习后模型存储位置 --predict_img_path 待预测样本位置 --size 切片尺寸

此外 该程序运行还需inception v3的预训练模型  
此模型可以在我的github迁移学习哪一章的 下载模型.py 运行下载
