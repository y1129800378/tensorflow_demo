
import os
import numpy as np
import tensorflow as tf 
import input_train_data as input_data
import  model
import glob
import re
from PIL import Image
import matplotlib.pyplot as plt
import get_patch

possible1=0
possible2=0


loabel=''
def get_one_image(train):
    image = Image.open(train)
    plt.imshow(image)
    image = image.resize([256, 256])
    
    image = np.array(image)
#     plt.show()
    return image

def evaluate_one_image(path_image,path_model):
    '''Test one image against the saved models and parameters
    '''  
    # you need to change the directories to yours.
    image_array = get_one_image(path_image)
    print("test")
    plt.imshow(image_array)
#     plt.show()
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2
         
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 256, 256, 3])

        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
         
        logit = tf.nn.softmax(logit)
         
        x = tf.placeholder(tf.float32, shape=[256,256, 3])
                                
        saver = tf.train.Saver()
         
        with tf.Session() as sess:
             
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(path_model)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
             
            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            possible1=prediction[:, 0]
            possible2=prediction[:, 1] 
            if max_index==0:
                loabel=1
                print('This is a tumor with possibility %.6f' %prediction[:, 0])
            elif max_index==1:
                loabel=2
                print('This is a normal with possibility %.6f' %prediction[:, 1])
           
    return loabel
#%%
 


def prediction_and_get_xml(path_image_needPrediction,name_image_needPrediction,path_patch_image_save_place,path_model):
    get_patch.get_patch_pic(path_image_needPrediction,name_image_needPrediction,path_patch_image_save_place)

    
    Tumor_list=[]
    paths = glob.glob(os.path.join(path_patch_image_save_place, '*'))
    paths.sort()  
    for jgp_path in paths:
 
        num=evaluate_one_image(jgp_path,path_model)
        if num==1:
            Tumor_list.append(jgp_path)
 
    print("Tumor_list",Tumor_list)


#     data_list=['/raid/CAMELYON/test_test002/normal__12928_100480.jpg', '/raid/CAMELYON/test_test002/normal__13184_100736.jpg', '/raid/CAMELYON/test_test002/normal__13440_100736.jpg', '/raid/CAMELYON/test_test002/normal__13696_100736.jpg', '/raid/CAMELYON/test_test002/normal__14464_98944.jpg', '/raid/CAMELYON/test_test002/normal__15232_100736.jpg', '/raid/CAMELYON/test_test002/normal__17792_100480.jpg', '/raid/CAMELYON/test_test002/normal__18560_102016.jpg', '/raid/CAMELYON/test_test002/normal__18816_106368.jpg', '/raid/CAMELYON/test_test002/normal__19584_102784.jpg', '/raid/CAMELYON/test_test002/normal__19840_102784.jpg', '/raid/CAMELYON/test_test002/normal__20608_119680.jpg', '/raid/CAMELYON/test_test002/normal__20864_119424.jpg', '/raid/CAMELYON/test_test002/normal__21376_104576.jpg', '/raid/CAMELYON/test_test002/normal__22656_106624.jpg', '/raid/CAMELYON/test_test002/normal__23680_106368.jpg', '/raid/CAMELYON/test_test002/normal__23936_107392.jpg', '/raid/CAMELYON/test_test002/normal__24960_117376.jpg', '/raid/CAMELYON/test_test002/normal__25216_108160.jpg', '/raid/CAMELYON/test_test002/normal__25216_117632.jpg', '/raid/CAMELYON/test_test002/normal__25472_108160.jpg', '/raid/CAMELYON/test_test002/normal__26496_79232.jpg', '/raid/CAMELYON/test_test002/normal__28032_121216.jpg', '/raid/CAMELYON/test_test002/normal__28288_110976.jpg', '/raid/CAMELYON/test_test002/normal__28288_120960.jpg', '/raid/CAMELYON/test_test002/normal__28288_121472.jpg', '/raid/CAMELYON/test_test002/normal__28288_121728.jpg', '/raid/CAMELYON/test_test002/normal__28544_110720.jpg', '/raid/CAMELYON/test_test002/normal__28544_120448.jpg', '/raid/CAMELYON/test_test002/normal__28544_121216.jpg', '/raid/CAMELYON/test_test002/normal__29056_73344.jpg', '/raid/CAMELYON/test_test002/normal__29056_79232.jpg', '/raid/CAMELYON/test_test002/normal__29312_110720.jpg', '/raid/CAMELYON/test_test002/normal__29312_119680.jpg', '/raid/CAMELYON/test_test002/normal__29568_111744.jpg', '/raid/CAMELYON/test_test002/normal__29568_119680.jpg', '/raid/CAMELYON/test_test002/normal__29824_119680.jpg', '/raid/CAMELYON/test_test002/normal__30336_118656.jpg', '/raid/CAMELYON/test_test002/normal__30336_118912.jpg', '/raid/CAMELYON/test_test002/normal__30592_117888.jpg', '/raid/CAMELYON/test_test002/normal__30592_118144.jpg', '/raid/CAMELYON/test_test002/normal__30592_77440.jpg', '/raid/CAMELYON/test_test002/normal__30592_78208.jpg', '/raid/CAMELYON/test_test002/normal__30592_78464.jpg', '/raid/CAMELYON/test_test002/normal__30592_84608.jpg', '/raid/CAMELYON/test_test002/normal__30848_117376.jpg', '/raid/CAMELYON/test_test002/normal__30848_117888.jpg', '/raid/CAMELYON/test_test002/normal__30848_118656.jpg', '/raid/CAMELYON/test_test002/normal__30848_118912.jpg', '/raid/CAMELYON/test_test002/normal__30848_77184.jpg', '/raid/CAMELYON/test_test002/normal__30848_77440.jpg', '/raid/CAMELYON/test_test002/normal__31104_116864.jpg', '/raid/CAMELYON/test_test002/normal__31104_117120.jpg', '/raid/CAMELYON/test_test002/normal__31104_118144.jpg', '/raid/CAMELYON/test_test002/normal__31104_118400.jpg', '/raid/CAMELYON/test_test002/normal__31104_76672.jpg', '/raid/CAMELYON/test_test002/normal__31104_76928.jpg', '/raid/CAMELYON/test_test002/normal__31104_77184.jpg', '/raid/CAMELYON/test_test002/normal__31360_116864.jpg', '/raid/CAMELYON/test_test002/normal__31360_118400.jpg', '/raid/CAMELYON/test_test002/normal__31360_77184.jpg', '/raid/CAMELYON/test_test002/normal__31616_116864.jpg', '/raid/CAMELYON/test_test002/normal__31616_118400.jpg', '/raid/CAMELYON/test_test002/normal__31872_163712.jpg', '/raid/CAMELYON/test_test002/normal__31872_163968.jpg', '/raid/CAMELYON/test_test002/normal__31872_164224.jpg', '/raid/CAMELYON/test_test002/normal__31872_164992.jpg', '/raid/CAMELYON/test_test002/normal__31872_165504.jpg', '/raid/CAMELYON/test_test002/normal__31872_165760.jpg', '/raid/CAMELYON/test_test002/normal__31872_166016.jpg', '/raid/CAMELYON/test_test002/normal__32128_163456.jpg', '/raid/CAMELYON/test_test002/normal__32128_163712.jpg', '/raid/CAMELYON/test_test002/normal__32128_163968.jpg', '/raid/CAMELYON/test_test002/normal__32128_164224.jpg', '/raid/CAMELYON/test_test002/normal__32128_164480.jpg', '/raid/CAMELYON/test_test002/normal__32128_164736.jpg', '/raid/CAMELYON/test_test002/normal__32128_164992.jpg', '/raid/CAMELYON/test_test002/normal__32128_165248.jpg', '/raid/CAMELYON/test_test002/normal__32128_165504.jpg', '/raid/CAMELYON/test_test002/normal__32128_165760.jpg', '/raid/CAMELYON/test_test002/normal__32128_166016.jpg', '/raid/CAMELYON/test_test002/normal__32128_166272.jpg', '/raid/CAMELYON/test_test002/normal__32384_163200.jpg', '/raid/CAMELYON/test_test002/normal__32384_163456.jpg', '/raid/CAMELYON/test_test002/normal__32384_163712.jpg', '/raid/CAMELYON/test_test002/normal__32384_163968.jpg', '/raid/CAMELYON/test_test002/normal__32384_164224.jpg', '/raid/CAMELYON/test_test002/normal__32384_164480.jpg', '/raid/CAMELYON/test_test002/normal__32384_164736.jpg', '/raid/CAMELYON/test_test002/normal__32384_164992.jpg', '/raid/CAMELYON/test_test002/normal__32384_165248.jpg', '/raid/CAMELYON/test_test002/normal__32384_165504.jpg', '/raid/CAMELYON/test_test002/normal__32384_165760.jpg', '/raid/CAMELYON/test_test002/normal__32384_166016.jpg', '/raid/CAMELYON/test_test002/normal__32384_166272.jpg', '/raid/CAMELYON/test_test002/normal__32640_163200.jpg', '/raid/CAMELYON/test_test002/normal__32640_163456.jpg', '/raid/CAMELYON/test_test002/normal__32640_163712.jpg', '/raid/CAMELYON/test_test002/normal__32640_163968.jpg', '/raid/CAMELYON/test_test002/normal__32640_164224.jpg', '/raid/CAMELYON/test_test002/normal__32640_164480.jpg', '/raid/CAMELYON/test_test002/normal__32640_164736.jpg', '/raid/CAMELYON/test_test002/normal__32640_164992.jpg', '/raid/CAMELYON/test_test002/normal__32640_165248.jpg', '/raid/CAMELYON/test_test002/normal__32640_165504.jpg', '/raid/CAMELYON/test_test002/normal__32640_165760.jpg', '/raid/CAMELYON/test_test002/normal__32640_166016.jpg', '/raid/CAMELYON/test_test002/normal__32640_166272.jpg', '/raid/CAMELYON/test_test002/normal__32640_166528.jpg', '/raid/CAMELYON/test_test002/normal__32640_166784.jpg', '/raid/CAMELYON/test_test002/normal__32896_159872.jpg', '/raid/CAMELYON/test_test002/normal__32896_163200.jpg', '/raid/CAMELYON/test_test002/normal__32896_163456.jpg', '/raid/CAMELYON/test_test002/normal__32896_163712.jpg', '/raid/CAMELYON/test_test002/normal__32896_163968.jpg', '/raid/CAMELYON/test_test002/normal__32896_164224.jpg', '/raid/CAMELYON/test_test002/normal__32896_164480.jpg', '/raid/CAMELYON/test_test002/normal__32896_164736.jpg', '/raid/CAMELYON/test_test002/normal__32896_164992.jpg', '/raid/CAMELYON/test_test002/normal__32896_165248.jpg', '/raid/CAMELYON/test_test002/normal__32896_165504.jpg', '/raid/CAMELYON/test_test002/normal__32896_165760.jpg', '/raid/CAMELYON/test_test002/normal__32896_166016.jpg', '/raid/CAMELYON/test_test002/normal__32896_166272.jpg', '/raid/CAMELYON/test_test002/normal__32896_166528.jpg', '/raid/CAMELYON/test_test002/normal__33152_110464.jpg', '/raid/CAMELYON/test_test002/normal__33152_159104.jpg', '/raid/CAMELYON/test_test002/normal__33152_163456.jpg', '/raid/CAMELYON/test_test002/normal__33152_163712.jpg', '/raid/CAMELYON/test_test002/normal__33152_163968.jpg', '/raid/CAMELYON/test_test002/normal__33152_164224.jpg', '/raid/CAMELYON/test_test002/normal__33152_164992.jpg', '/raid/CAMELYON/test_test002/normal__33152_165248.jpg', '/raid/CAMELYON/test_test002/normal__33152_165504.jpg', '/raid/CAMELYON/test_test002/normal__33152_165760.jpg', '/raid/CAMELYON/test_test002/normal__33152_166016.jpg', '/raid/CAMELYON/test_test002/normal__33152_166272.jpg', '/raid/CAMELYON/test_test002/normal__33408_163200.jpg', '/raid/CAMELYON/test_test002/normal__33408_163968.jpg', '/raid/CAMELYON/test_test002/normal__33408_164992.jpg', '/raid/CAMELYON/test_test002/normal__33408_165248.jpg', '/raid/CAMELYON/test_test002/normal__33408_165504.jpg', '/raid/CAMELYON/test_test002/normal__33408_165760.jpg', '/raid/CAMELYON/test_test002/normal__33408_166016.jpg', '/raid/CAMELYON/test_test002/normal__33408_166272.jpg', '/raid/CAMELYON/test_test002/normal__33408_166528.jpg', '/raid/CAMELYON/test_test002/normal__33408_166784.jpg', '/raid/CAMELYON/test_test002/normal__33664_163456.jpg', '/raid/CAMELYON/test_test002/normal__33664_163968.jpg', '/raid/CAMELYON/test_test002/normal__33664_164992.jpg', '/raid/CAMELYON/test_test002/normal__33664_165504.jpg', '/raid/CAMELYON/test_test002/normal__33664_165760.jpg', '/raid/CAMELYON/test_test002/normal__33664_166016.jpg', '/raid/CAMELYON/test_test002/normal__33664_166272.jpg', '/raid/CAMELYON/test_test002/normal__33664_166528.jpg', '/raid/CAMELYON/test_test002/normal__33664_166784.jpg', '/raid/CAMELYON/test_test002/normal__33920_164224.jpg', '/raid/CAMELYON/test_test002/normal__33920_165504.jpg', '/raid/CAMELYON/test_test002/normal__33920_165760.jpg', '/raid/CAMELYON/test_test002/normal__33920_166016.jpg', '/raid/CAMELYON/test_test002/normal__33920_166272.jpg', '/raid/CAMELYON/test_test002/normal__33920_166784.jpg', '/raid/CAMELYON/test_test002/normal__33920_167040.jpg', '/raid/CAMELYON/test_test002/normal__34176_164992.jpg', '/raid/CAMELYON/test_test002/normal__34176_165248.jpg', '/raid/CAMELYON/test_test002/normal__34176_165504.jpg', '/raid/CAMELYON/test_test002/normal__34176_165760.jpg', '/raid/CAMELYON/test_test002/normal__34176_166016.jpg', '/raid/CAMELYON/test_test002/normal__34176_166272.jpg', '/raid/CAMELYON/test_test002/normal__34176_166528.jpg', '/raid/CAMELYON/test_test002/normal__34176_166784.jpg', '/raid/CAMELYON/test_test002/normal__34432_165248.jpg', '/raid/CAMELYON/test_test002/normal__34432_165760.jpg', '/raid/CAMELYON/test_test002/normal__34432_166016.jpg', '/raid/CAMELYON/test_test002/normal__34432_166272.jpg', '/raid/CAMELYON/test_test002/normal__34432_166784.jpg', '/raid/CAMELYON/test_test002/normal__34432_167040.jpg', '/raid/CAMELYON/test_test002/normal__34688_166016.jpg', '/raid/CAMELYON/test_test002/normal__34688_166272.jpg', '/raid/CAMELYON/test_test002/normal__34688_166528.jpg', '/raid/CAMELYON/test_test002/normal__34688_166784.jpg', '/raid/CAMELYON/test_test002/normal__34944_100736.jpg', '/raid/CAMELYON/test_test002/normal__34944_164992.jpg', '/raid/CAMELYON/test_test002/normal__34944_165760.jpg', '/raid/CAMELYON/test_test002/normal__34944_166272.jpg', '/raid/CAMELYON/test_test002/normal__34944_166784.jpg', '/raid/CAMELYON/test_test002/normal__35200_159360.jpg', '/raid/CAMELYON/test_test002/normal__35200_159616.jpg', '/raid/CAMELYON/test_test002/normal__35200_164992.jpg', '/raid/CAMELYON/test_test002/normal__35200_165504.jpg', '/raid/CAMELYON/test_test002/normal__35200_166016.jpg', '/raid/CAMELYON/test_test002/normal__35200_166528.jpg', '/raid/CAMELYON/test_test002/normal__35200_166784.jpg', '/raid/CAMELYON/test_test002/normal__35456_165248.jpg', '/raid/CAMELYON/test_test002/normal__35456_165504.jpg', '/raid/CAMELYON/test_test002/normal__35712_165248.jpg', '/raid/CAMELYON/test_test002/normal__37760_93824.jpg', '/raid/CAMELYON/test_test002/normal__41344_98432.jpg', '/raid/CAMELYON/test_test002/normal__54656_124160.jpg', '/raid/CAMELYON/test_test002/normal__54656_124416.jpg', '/raid/CAMELYON/test_test002/normal__55424_124672.jpg', '/raid/CAMELYON/test_test002/normal__55936_125184.jpg', '/raid/CAMELYON/test_test002/normal__59008_125952.jpg', '/raid/CAMELYON/test_test002/normal__60288_119808.jpg', '/raid/CAMELYON/test_test002/normal__60544_119552.jpg', '/raid/CAMELYON/test_test002/normal__60544_132864.jpg', '/raid/CAMELYON/test_test002/normal__60544_133120.jpg', '/raid/CAMELYON/test_test002/normal__60544_133376.jpg', '/raid/CAMELYON/test_test002/normal__60800_114944.jpg', '/raid/CAMELYON/test_test002/normal__60800_133120.jpg', '/raid/CAMELYON/test_test002/normal__61056_129792.jpg', '/raid/CAMELYON/test_test002/normal__61312_114432.jpg', '/raid/CAMELYON/test_test002/normal__62592_119296.jpg', '/raid/CAMELYON/test_test002/normal__64128_113152.jpg', '/raid/CAMELYON/test_test002/normal__64384_112640.jpg', '/raid/CAMELYON/test_test002/normal__64384_112896.jpg', '/raid/CAMELYON/test_test002/normal__64384_113152.jpg', '/raid/CAMELYON/test_test002/normal__64640_112896.jpg', '/raid/CAMELYON/test_test002/normal__64640_113152.jpg', '/raid/CAMELYON/test_test002/normal__64640_129792.jpg', '/raid/CAMELYON/test_test002/normal__64896_113152.jpg', '/raid/CAMELYON/test_test002/normal__65152_112384.jpg', '/raid/CAMELYON/test_test002/normal__65152_115200.jpg', '/raid/CAMELYON/test_test002/normal__65408_112640.jpg', '/raid/CAMELYON/test_test002/normal__65920_128768.jpg', '/raid/CAMELYON/test_test002/normal__66176_130048.jpg', '/raid/CAMELYON/test_test002/normal__66944_132864.jpg', '/raid/CAMELYON/test_test002/normal__67200_126208.jpg', '/raid/CAMELYON/test_test002/normal__67200_130304.jpg', '/raid/CAMELYON/test_test002/normal__67200_130560.jpg', '/raid/CAMELYON/test_test002/normal__67456_130560.jpg', '/raid/CAMELYON/test_test002/normal__67712_127232.jpg', '/raid/CAMELYON/test_test002/normal__67712_131072.jpg', '/raid/CAMELYON/test_test002/normal__67712_132096.jpg', '/raid/CAMELYON/test_test002/normal__67712_132352.jpg', '/raid/CAMELYON/test_test002/normal__67968_131840.jpg', '/raid/CAMELYON/test_test002/normal__68224_130048.jpg', '/raid/CAMELYON/test_test002/normal__68224_130560.jpg', '/raid/CAMELYON/test_test002/normal__68480_132864.jpg', '/raid/CAMELYON/test_test002/normal__68736_104192.jpg', '/raid/CAMELYON/test_test002/normal__68736_129280.jpg', '/raid/CAMELYON/test_test002/normal__68736_130048.jpg', '/raid/CAMELYON/test_test002/normal__68992_127744.jpg', '/raid/CAMELYON/test_test002/normal__70016_134400.jpg', '/raid/CAMELYON/test_test002/normal__70272_134656.jpg', '/raid/CAMELYON/test_test002/normal__70528_134656.jpg', '/raid/CAMELYON/test_test002/normal__70784_134400.jpg', '/raid/CAMELYON/test_test002/normal__70784_134656.jpg', '/raid/CAMELYON/test_test002/normal__71040_134656.jpg', '/raid/CAMELYON/test_test002/normal__71296_134400.jpg', '/raid/CAMELYON/test_test002/normal__71296_134656.jpg', '/raid/CAMELYON/test_test002/normal__71552_112640.jpg', '/raid/CAMELYON/test_test002/normal__71808_126208.jpg', '/raid/CAMELYON/test_test002/normal__72064_126208.jpg', '/raid/CAMELYON/test_test002/normal__72064_126464.jpg', '/raid/CAMELYON/test_test002/normal__72320_124160.jpg', '/raid/CAMELYON/test_test002/normal__72576_124160.jpg', '/raid/CAMELYON/test_test002/normal__72832_127488.jpg', '/raid/CAMELYON/test_test002/normal__73088_103168.jpg', '/raid/CAMELYON/test_test002/normal__73088_127488.jpg', '/raid/CAMELYON/test_test002/normal__73600_127232.jpg', '/raid/CAMELYON/test_test002/normal__74368_110336.jpg', '/raid/CAMELYON/test_test002/normal__74624_101376.jpg', '/raid/CAMELYON/test_test002/normal__74624_104960.jpg', '/raid/CAMELYON/test_test002/normal__74880_101376.jpg', '/raid/CAMELYON/test_test002/normal__75392_109312.jpg', '/raid/CAMELYON/test_test002/normal__75392_109568.jpg', '/raid/CAMELYON/test_test002/normal__75392_136448.jpg', '/raid/CAMELYON/test_test002/normal__79744_118272.jpg', '/raid/CAMELYON/test_test002/normal__80768_119808.jpg', '/raid/CAMELYON/test_test002/normal__80768_127232.jpg', '/raid/CAMELYON/test_test002/normal__81024_111360.jpg', '/raid/CAMELYON/test_test002/normal__81280_110592.jpg', '/raid/CAMELYON/test_test002/normal__82816_111104.jpg', '/raid/CAMELYON/test_test002/normal__84352_109056.jpg', '/raid/CAMELYON/test_test002/normal__84352_112128.jpg', '/raid/CAMELYON/test_test002/normal__84608_111616.jpg', '/raid/CAMELYON/test_test002/normal__84608_112128.jpg', '/raid/CAMELYON/test_test002/normal__84864_127488.jpg', '/raid/CAMELYON/test_test002/normal__86144_113408.jpg', '/raid/CAMELYON/test_test002/normal__87680_121856.jpg']
    data_list=Tumor_list
    i=0
    datax_list=[]
    datay_list=[]
    print(len(data_list))
    j=0

    for str_data in data_list:
        strx=re.findall("normal__(.*?)_",str_data)[0]
        stry=re.findall("normal__(.*?).jpg",str_data)[0]
        stry_true=stry.replace(strx+"_","")
        strx_true=int(strx)
        stry_true=int(stry_true) 
        data_x=strx_true
        data_y=stry_true      
        datax_list.append(data_x)
        datay_list.append(data_y)
        i=i+1
    biao_ji=0  
    #change your path  
    
    with open(r'/home/yyy/Desktop/read_roi_Tumor_test.xml','w') as txt_data:
        txt_data.write(''' <?xml version='1.0' encoding='utf-8'?>'''+"\r\n")
        txt_data.write("<ASAP_Annotations>"+"\r\n")
        txt_data.write('''  <Annotations>'''+"\r\n")

        for  x,y in zip(datax_list,datay_list):

            strx_true=int(x)
            stry_true=int(y) 
            data_x=strx_true
            data_y=stry_true      
            txt_data.write('''<Annotation Color="#64FE2E" Name="Annotation '''+str(biao_ji)+''' - Prob = 0.9974" PartOfGroup="True_Positives" Type="Dot">'''+"\r\n")
            txt_data.write("<Coordinates>"+"\r\n")
            txt_data.write('''<Coordinate Order="0" X="'''+str(data_x)+'''" Y="'''+str(data_y)+'''" />'''+"\r\n")
            txt_data.write("</Coordinates>"+"\r\n")
            txt_data.write("</Annotation>"+"\r\n")
            biao_ji=biao_ji+1
        txt_data.write('''</Annotations>'''+"\r\n")    
        txt_data.write(''' <AnnotationGroups>'''+"\r\n")    
        txt_data.write('''<Group Color="#64FE2E" Name="True_Positives" PartOfGroup="None" Type="Dot">'''+"\r\n")        
        txt_data.write('''<Attributes />'''+"\r\n")    
        txt_data.write('''</Group>'''+"\r\n")    
        txt_data.write('''<Group Color="#ff0000" Name="False_Positives" PartOfGroup="None" Type="Dot">'''+"\r\n")
        txt_data.write('''<Attributes />'''+"\r\n")  
        txt_data.write('''</Group>'''+"\r\n")  
        txt_data.write(''' </AnnotationGroups>'''+"\r\n")  
        txt_data.write('''</ASAP_Annotations> '''+"\r\n")  


if __name__ == '__main__':
    prediction_and_get_xml('/raid/CAMELYON/CAMELYON16/TrainingData/Train_Tumor/','tumor_110.tif','/home/yyy/get_patch_pic/','/home/yyy/save_camylon_train')
    





