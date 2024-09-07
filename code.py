from tensorflow.keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

image_directory = './segm/unlabel/'
mask_directory = './segm/label/'

image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.
mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

images_ = os.listdir(image_directory)
images = sorted(images_)
SIZE = 80

for i, image_name in enumerate(images):
  if (image_name.split('.')[1] == 'jpg'):    #Remember enumerate method adds a counter and returns the enumerate object
      image = cv2.imread(image_directory+image_name, 0)
      image = Image.fromarray(image)
      image = image.resize((SIZE, SIZE))
      image_dataset.append(np.array(image))

masks_ = os.listdir(mask_directory)
masks = sorted(masks_)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(mask_directory+image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))

#Normalize images
image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1),3)
#D not normalize masks, just rescale to 0 to 1.
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255

print(image_dataset.shape)
print(mask_dataset.shape)

from sklearn.model_selection import train_test_split
X_trainv, X_test, y_trainv, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.15)

X_trainv.shape

import random
import numpy as np
image_number = random.randint(0, len(X_trainv))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_trainv[image_number], (SIZE,SIZE)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_trainv[image_number], (SIZE,SIZE)), cmap='gray')
plt.show()

IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS

from re import VERBOSE
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda

def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.MeanIoU(num_classes=2) , tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    # model.summary()

    return model
def whole():
  model = simple_unet_model(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
  history = model.fit(X_train, y_train,
                      batch_size = 16,
                      verbose=0,
                      epochs=100,
                      validation_data=(X_val, y_val),
                      shuffle=False)
  loss, acc, AUC, IoU, precision, recall = model.evaluate(X_val, y_val)
  # printing
  print(f"loss = {loss:0.6f}")
  print(f"Accuracy = {(acc * 100.0):3.2f} %")
  print(f"Mean Area of SGs = {AUC:0.6f}")
  print(f"Mean IoU value is {IoU:0.6f}")
  print(f"Precision and Recall values are {precision:0.6f} and {recall:0.6f}")
  #plot the training and validation accuracy and loss at each epoch
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(1, len(loss) + 1)
  plt.figure(figsize=(12,6))
  plt.subplot(1,2,1)
  plt.plot(epochs, loss, 'g', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  plt.subplot(1,2,2)
  plt.plot(epochs, acc, 'g', label='Training acc')
  plt.plot(epochs, val_acc, 'r', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  # plt.tight_layout()
  plt.show()
  return model

spl = 4
test_img_number = random.randint(0, len(X_test))

kf = KFold(n_splits=spl)
for train, test in kf.split(X_trainv,y_trainv):
  X_train = np.array([X_trainv[i] for i in train])
  X_val = np.array([X_trainv[i] for i in test])
  y_train = np.array([y_trainv[i] for i in train])
  y_val = np.array([y_trainv[i] for i in test])
  model = whole()
  #Prediction
  # test_img_number = random.randint(0, len(X_test))
  test_img = X_test[test_img_number]
  ground_truth=y_test[test_img_number]
  test_img_norm=test_img[:,:,0][:,:,None]
  test_img_input=np.expand_dims(test_img_norm, 0)
  prediction = (model.predict(test_img_input)[0,:,:,0] > 0.2).astype(np.uint8)
  plt.figure(figsize=(12, 6))
  plt.subplot(231)
  plt.title('Testing Image')
  plt.imshow(test_img[:,:,0], cmap='gray')
  plt.subplot(232)
  plt.title('Testing Label')
  plt.imshow(ground_truth[:,:,0], cmap='gray')
  plt.subplot(233)
  plt.title('Prediction on test image')
  plt.imshow(prediction, cmap='gray')
  plt.show()

# model = simple_unet_model(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)

# history = model.fit(X_train, y_train,
#                     batch_size = 16,
#                     verbose=0,
#                     epochs=100,
#                     validation_data=(X_val, y_val),
#                     shuffle=True)

# loss, acc, AUC, IoU, precision, recall = model.evaluate(X_val, y_val)
# print("loss = ",loss)
# print("Accuracy = ", (acc * 100.0), "%")
# print(f"Mean Area of SGs = {AUC}")
# print(f"Mean IoU value is {IoU}")
# print(f"Precision and Recall values are {precision} and {recall}")

# #plot the training and validation accuracy and loss at each epoch
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.figure()
# plt.plot(epochs, loss, 'g', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# plt.plot(epochs, acc, 'g', label='Training acc')
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# test_img_number = random.randint(0, len(X_test))
# test_img = X_test[test_img_number]
# ground_truth=y_test[test_img_number]
# test_img_norm=test_img[:,:,0][:,:,None]
# test_img_input=np.expand_dims(test_img_norm, 0)
# prediction = (model.predict(test_img_input)[0,:,:,0] > 0.2).astype(np.uint8)

# test_img_other = cv2.imread('/content/drive/MyDrive/segm/unlabel/1 (53).jpg', 0)
# test_img_other=cv2.resize(test_img_other,(IMG_HEIGHT,IMG_WIDTH))
# test_img_other_norm = np.expand_dims(normalize(np.array(test_img_other), axis=1),2)

# test_img_other_norm=test_img_other_norm[:,:,0][:,:,None]
# test_img_other_input=np.expand_dims(test_img_other_norm, axis=0)

# test_img_other_input.shape

# prediction_other = (model.predict(test_img_other_input)[0,:,:,0] > 0.2).astype(np.uint8)

# plt.figure(figsize=(12, 6))
# plt.subplot(231)
# plt.title('Testing Image')
# plt.imshow(test_img[:,:,0], cmap='gray')
# plt.subplot(232)
# plt.title('Testing Label')
# plt.imshow(ground_truth[:,:,0], cmap='gray')
# plt.subplot(233)
# plt.title('Prediction on test image')
# plt.imshow(prediction, cmap='gray')
# plt.subplot(234)
# plt.title('External Image')
# plt.imshow(test_img_other, cmap='gray')
# plt.subplot(236)
# plt.title('Prediction of external Image')
# plt.imshow(prediction_other, cmap='gray')
# plt.show()