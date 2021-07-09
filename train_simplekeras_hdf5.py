# importing libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten, GlobalMaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras import regularizers
import logging
import sys  
from h5imagegenerator import HDF5ImageGenerator
import albumentations as A  
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten, GlobalMaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras import regularizers
import logging
import sys
from focal_loss import BinaryFocalLoss
#tf.compat.v1.disable_eager_execution()

#train_data_dir = '/data2/Naresh/data/BACH/final_train_test_val/train'
#validation_data_dir = '/data2/Naresh/data/BACH/final_train_test_val/val'
train_data_dir = '/data2/Naresh/data/BACH/images_hdf5_new/train.h5'
validation_dir = '/data2/Naresh/data/BACH/images_hdf5_new/val.h5' 
nb_train_samples = 12864 
#nb_train_samples = 3072
nb_validation_samples = 3264
epochs = 10
batch_size = 64
img_width, img_height = 256, 256
num_classes=2
img_size=img_width
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# This must be fixed for multi-GPU
mirrored_strategy = tf.distribute.MirroredStrategy()
#mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
#mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
with mirrored_strategy.scope():
    model = 'notcustom'
    if model == 'custom':
        model = Sequential()
        model.add(Conv2D(32, (2, 2), input_shape = input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size =(2, 2)))
          
        model.add(Conv2D(32, (2, 2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size =(2, 2)))
          
        model.add(Conv2D(64, (2, 2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size =(2, 2)))
          
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
    else:
        input_tensor = Input(shape=(img_size, img_size, 3))
        model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False, input_tensor=input_tensor, input_shape=input_shape)
        base_model = model
        x = base_model.output
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Flatten()(x)
        out = Dense(num_classes, activation='softmax')(x)
        base_model.trainable = False  
        model = Model(inputs=input_tensor, outputs=out)
    model.compile(loss ='binary_crossentropy',
                         optimizer ='rmsprop',
                       metrics =['accuracy'])

#my_augmenter = Compose([ rescale = 1. / 255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True])
my_augmenter = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomContrast(limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.RandomBrightness(limit=0.2, p=0.5)])   
#labels_encoding='hot',
                
train_generator = HDF5ImageGenerator(
        src=train_data_dir,
        X_key='image',
        y_key='label',
        scaler=True,
        batch_size=batch_size,
        augmenter=my_augmenter)
validation_generator = HDF5ImageGenerator(
        src=validation_dir,
        X_key='image',
        y_key='label',
        scaler=True,
        batch_size=batch_size,
        augmenter=my_augmenter)

# num_img=0    
# for image, label in train_generator:
    # print("Image shape: ", image.shape)
    # #print("Image shape: ", image["anchor"].numpy().shape)
    # #print("Image shape: ", image["neg_img"].numpy().shape)
    # #print("Label: ", label.numpy().shape)
    # print("Label: ", label.shape)
    # sys.exit(0)
    # #num_img=num_img+1
# #print(num_img)  
# sys.exit(0)  

    
model.fit_generator(train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs, validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size,workers=1, use_multiprocessing=False, verbose=1)
  
model.save_weights('model_saved.h5')  
sys.exit(0)  
train_datagen = ImageDataGenerator(
                rescale = 1. / 255,
                 shear_range = 0.2,
                  zoom_range = 0.2,
            horizontal_flip = True)
  
test_datagen = ImageDataGenerator(rescale = 1. / 255)
  
train_generator = train_datagen.flow_from_directory(train_data_dir,
                              target_size =(img_width, img_height),
                     batch_size = batch_size, class_mode ='binary')
  
validation_generator = test_datagen.flow_from_directory(
                                    validation_data_dir,
                   target_size =(img_width, img_height),
          batch_size = batch_size, class_mode ='binary')
  
model.fit_generator(train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs, validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)
  
model.save_weights('model_saved.h5')
