# 简介：	
该工程分为两个部分：

一.训练模型 

二.使用已有模型预测

1.训练模型可直接运行train_model.py。其中run_training函数入口参数为模型保存地址。

2.使用已有模型预测，可直接运行test_model_to_prediction.py。

其中prediction_and_get_xml函数入口参数依次为：

（1）需要预测的图片地址

（2）需要预测的图片名称

（3）中间截图的保存地址

（4）已有模型的读取路径

最后得到roi对应的xml文件，xml文件保存路径在test_model_to_prediction.py文件中第116行根据自己需要修改。


使用环境：

python3.5 

opencv3.0

openslide

tensorflow 0.12.1(GPU版本)

提示：

由于opencv对于titan x框架及CUDA 8.0不友好，安装过程极其复杂，建议使用yyy账号进行测试。
