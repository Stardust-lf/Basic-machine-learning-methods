import os
import sys
import glob
import matplotlib.pyplot as plt

from tensorflow.keras.applications.densenet import DenseNet201,preprocess_input

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint

def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


IM_WIDTH, IM_HEIGHT = 224, 224  # densenet指定的图片尺寸

train_dir = 'C:\\Users\\StarDust\\PycharmProjects\\untitled2\\images\\trashfin_train'  # 训练集数据路径
val_dir = 'C:\\Users\\StarDust\\PycharmProjects\\untitled2\\images\\trashfin_test'  # 验证集数据
nb_classes = 4
nb_epoch = 3
batch_size = 32

nb_train_samples = get_nb_files(train_dir)  # 训练样本个数
nb_classes = len(glob.glob(train_dir + "/*"))  # 分类数
nb_val_samples = get_nb_files(val_dir)  # 验证集样本个数
print('验证集有',nb_val_samples,'个')
nb_epoch = int(nb_epoch)  # epoch数量
batch_size = int(batch_size) #每一批次图片的数量


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# 训练数据与测试数据
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size, class_mode='categorical',
    shuffle=True)


validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size, class_mode='categorical',
    shuffle=True)


# 添加新层
def add_new_last_layer(base_model, nb_classes):
    """
    添加最后的层
    输入
    base_model和分类数量
    输出
    新的keras的model
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


# 搭建模型
model = DenseNet201(include_top=False)
model = add_new_last_layer(model, nb_classes)
# model.load_weights('../model/checkpoint-02e-val_acc_0.82.hdf5')  第二次训练可以接着第一次训练得到的模型接着训练
model.compile(optimizer=SGD(lr=0.001, momentum=0.9, decay=0.0001, nesterov=True), loss='categorical_crossentropy',
              metrics=['accuracy'])

# 更好地保存模型 Save the model after every epoch.
output_model_file = 'C:\\Users\\StarDust\\PycharmProjects\\untitled2\\memory\\checkpoint-{epoch:02d}e-accuracy_{accuracy:.2f}.hdf5'
checkpoint = ModelCheckpoint(output_model_file, monitor='val-acc', verbose=1, save_best_only=True)

# 开始训练
history_ft = model.fit(
    train_generator,
    #steps_per_epoch=1, #一次epoch中处理图片的批次数，默认为训练集样本个数/训练集一批次图片数（32）
    epochs=nb_epoch,#迭代次数
    callbacks=[checkpoint],
    validation_data=validation_generator,
    #validation_steps=10)
    validation_steps=nb_val_samples)#测试集的图片数量


print(history_ft.history)#打印历史记录字典，键val_accuracy的值为测试集准确率
