#基于tensorflow2中的keras框架

##############数据提取###############

import h5py
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import keyboard
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib

# 打开h5文件
f = h5py.File('train_pre_data.h5', 'r')
data = f['data']
data = np.array(data)

#由于数据集的不规范，数据数据分为了两类，这里是第一类。
type1 = [0,3,5,7,8,12,14,17,19,21,24,26,28,29,32,33,34,35,37,38,39,41,44,46,49,52,54,55,57,59,60,62,65,69,72,74,75,76,77,80,82,83,84,86,89,91,92,93,94,96,98,102,103,104,105,106,107,109,110,113,117,120,125,129,130,131,132,133,134,135,136,145,146,147,150,152,155,158,159,163,166,167,169,171,174,175,176,177,178,179,182,183,184,186,187,188,189,194,196,198,199,201,204,205,206,208,213,215,217,220,222,223,225,226,227,228,229,231,232,233,234,235,237,239,244,247,250,251,253,260,261,263,266,268,270,271,272,275,277,278,279,281,282,283,286,288,290,291,294,296,297]

"""
   下面开始从原始的h5文件中提取我们需要的数据,需要说明的是，h5文件一共300个样本，每一个样本都是3D的数据，我采取的策略是，对于一个样本，
   我只取一张横截面图。并且每张截面图都是属于一种类型，之所以选取这个类型，是因为查询了一些医学方面的资料，得知这种类型的图片识别效果最好，
   具体内容详见说明文档
"""
count_pic = 0

train_file_csv = pd.read_csv("train_pre_label_validation.csv") # 读取train的csv文件，或者读取validation的csv文件，文件里的字段：【id】，【label】
label = train_file_csv['label']

train_id= train_file_csv['id']

for a in range(0,31): # 这里的range是填入(0,269)或者（0,31），因为一共300个数据集，我手动设置了，训练集269个数据，验证集31个数据
    
    location = str(label[a])
    train_id_dijige = train_id[a] #样本的编号
    t2 = 'validation' # 这里填入train或者validation,来明确是导入train文件夹，还是导入validation文件夹
    if train_id_dijige in type1: #如果样本编号属于第一种类型

        for k in range(42,43): #第一类型，从后看

            matplotlib.image.imsave('./image/'+t2+'/'+location+'/'+str(count_pic)+'.png',np.rot90((data[train_id_dijige][0])[:,k,:],k = 2))#导出到预先创建的文件夹
            count_pic = count_pic + 1

    else: #如果第二种类型

        for k in range(42,43): #第二种类型，从后看

            matplotlib.image.imsave('./image/'+t2+'/'+location+'/'+str(count_pic)+'.png',np.rot90((data[train_id_dijige][0])[:,k,:]))#导出到预先设定的文件夹
            count_pic = count_pic + 1

###########数据集的预处理，模型搭建，与预测##########
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from shutil import copyfile
import h5py
import numpy as np
import matplotlib
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
import scipy
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

#数据所在文件夹
base_dir = './image'
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')

#训练集
train_0_dir = os.path.join(train_dir,'0')
train_1_dir = os.path.join(train_dir,'1')
train_2_dir = os.path.join(train_dir,'2')

#验证集
validation_0_dir = os.path.join(validation_dir,'0')
validation_1_dir = os.path.join(validation_dir,'1')
validation_2_dir = os.path.join(validation_dir,'2')



#模型搭建
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (5,5), activation='relu',padding = 'SAME',input_shape=(79,79,3)),
    tf.keras.layers.MaxPooling2D((3, 3),strides = 2),
    
    #tf.keras.layers.Conv2D(16, (5,5),activation='relu',padding = 'SAME'),
    #tf.keras.layers.AvgPool2D((3,3),strides = 2),

    #tf.keras.layers.Conv2D(32, (5,5),activation='relu',padding = 'SAME'),
    #tf.keras.layers.AvgPool2D((3,3),strides = 2),
    
    #为全连接层准备
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
   
    tf.keras.layers.Dense(3,activation = 'softmax')
])

#模型编译
model.compile(optimizer = 'adam',
              loss ='categorical_crossentropy', 
              metrics = ['categorical_accuracy'])

#数据预处理
train_datagen = ImageDataGenerator(rescale = 1./255) #每个像素点除以255

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 10,
                                                    shuffle = True,
                                                    class_mode ='categorical', 
                                                    target_size = (79, 79)) # 指定resize成的大小，需要修改

validation_generator =  test_datagen.flow_from_directory( validation_dir,
                                                          batch_size  = 10,
                                                          class_mode  = 'categorical', 
                                                          target_size = (79,79))

print(validation_generator.class_indices)   # 输出对应的标签文件夹

checkpoint = ModelCheckpoint(filepath='./image/the_best_model_2.h5',monitor='val_loss',mode='min' ,save_best_only='True') #训练时保存val_loss最低的模型

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=10, min_lr=0.00000001) #训练时学习率衰减

history = model.fit(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch =40,
            epochs = 1000,
            validation_steps = 3,
            verbose = 1,
            callbacks = [checkpoint,reduce_lr])

#预测并生成最终的结果
predicted_list = [] #结果放在这里,最终结果

#一共232个预测对象，先把单个对象导入预先设置的文件夹，做成generator，进行预测，然后删除这个对象，再导入下一个待预测的对象，再预测，再删除，循环此操作。
#这里的预测对象，是选取的与前面训练集和验证集同类型的来自，测试集h5文件，的横截面图。
for i in range(0,232): #一共232个待预测对象
    src = './image/test1/'+str(i)+'.png'#这里需要修改
    dst = './image/test/2/'+str(i)+'.png'
    copyfile(src, dst) #将待预测的对象复制到预先设定的文件夹

    test_dir = os.path.join(base_dir,'test')
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    #生成test generator
    test_generator =  test_datagen.flow_from_directory( test_dir,
                                                          batch_size  = 10,
                                                          class_mode  = 'categorical', 
                                                          target_size = (79,79)) #这里需要修改size
    result = model.predict(test_generator) #开始预测
    predicted_list.append(np.argmax(result)) #输出possibility最大的标签
    dir_remove = './image/test/2/'+str(i)+'.png'
    os.remove(dir_remove) #删除待预测对象





