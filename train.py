from random import randint
from typing import Tuple

from unet_model import UNet_model
from PIL import Image 
import cv2
import numpy as np 
import os 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.utils import normalize

dataset_dir = "./lgg-mri-segmentation/kaggle_3m/"
images_dataset = []
masks_dataset = []
IMG_SIZE = 256

def generate_dataset(path : str):
    for d in os.listdir(path):
        if os.path.isdir(path+d):
           imgs_path = path + d 
           for img in os.listdir(imgs_path):
               image = cv2.imread(imgs_path + '/' + img,0)
               image = Image.fromarray(image)
               image = image.resize((IMG_SIZE, IMG_SIZE))
               if img[len(img) - 8 : len(img)- 4] != 'mask':
                   images_dataset.append(np.array(image))
               else:
                    masks_dataset.append(np.array(image))

        else: 
            print(f"{d} is not a directory")
            return



generate_dataset(dataset_dir)

images_dataset = np.expand_dims(normalize(np.array(images_dataset), axis=1),3)
masks_dataset = np.expand_dims((np.array(masks_dataset)), 3)/255
X_train , X_test, Y_train, Y_test = train_test_split(images_dataset, masks_dataset, test_size=0.10, random_state=0)

def show_random_image() -> None:
    img_n = randint(0,len(X_train))
    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.imshow(np.reshape(X_train[img_n], (IMG_SIZE, IMG_SIZE)), cmap='gray')
    plt.show()

IMG_HEIGHT = images_dataset.shape[1]
IMG_WIDHT = images_dataset.shape[2]
IMG_CHANNELS = images_dataset.shape[3]



def plot_loss(history):
    loss = history.history['accuracy']
    val_loss = history.history['val_accuracy']
    epochs = range(1, len(loss)+1)
    plt.plot(epochs, loss, 'y', label="training Accuracy")
    plt.plot(epochs, val_loss,'r', label='validation Accuracy')
    plt.title('Training and validation Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    return 

def predict():
    model = UNet_model(IMG_HEIGHT, IMG_WIDTH=IMG_WIDHT, IMG_CHANNELS=IMG_CHANNELS)
    model.load_weights('brain_tumor.hdf5')
    test_img_number = randint(0, len(X_test))
    test_img = X_test[test_img_number]
    ground_truth=Y_test[test_img_number]
    test_img_norm=test_img[:,:,0][:,:,None]
    test_img_input=np.expand_dims(test_img_norm, 0)
    prediction = (model.predict(test_img_input)[0,:,:,0] > 0.2).astype(np.uint8)

    test_img_other = cv2.imread('lgg-mri-segmentation/kaggle_3m/TCGA_HT_A616_19991226/TCGA_HT_A616_19991226_1.tif', 0)
    #test_img_other = cv2.imread('data/test_images/img8.tif', 0)
    test_img_other_norm = np.expand_dims(normalize(np.array(test_img_other), axis=1),2)
    test_img_other_norm=test_img_other_norm[:,:,0][:,:,None]
    test_img_other_input=np.expand_dims(test_img_other_norm, 0)

    #Predict and threshold for values above 0.5 probability
    #Change the probability threshold to low value (e.g. 0.05) for watershed demo.
    prediction_other = (model.predict(test_img_other_input)[0,:,:,0] > 0.2).astype(np.uint8)

    plt.figure(figsize=(16, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,0], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='gray')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(prediction, cmap='gray')
    plt.subplot(234)
    plt.title('External Image')
    plt.imshow(test_img_other, cmap='gray')
    plt.subplot(235)
    plt.title('Prediction of external Image')
    plt.imshow(prediction_other, cmap='gray')
    plt.show()

def train_model(batch_size , epochs , verbose):
    model = UNet_model(IMG_HEIGHT, IMG_WIDTH=IMG_WIDHT, IMG_CHANNELS=IMG_CHANNELS)

    history = model.fit(
        X_train, Y_train, 
        batch_size=batch_size,
        verbose=verbose,
        epochs=epochs,
        validation_data=(X_test,Y_test),
        shuffle=False
    )
    model.save('brain_tumor.hdf5')

    a, acc = model.evaluate(X_test, Y_test)

    print(f"Accuracy = {(acc * 100.0)}%")
    plot_loss(history)
    loss = history.history['loss']
    model.save('brain_tumor.hdf5')
    return acc, history, model

def main():

    model = UNet_model(IMG_HEIGHT, IMG_WIDTH=IMG_WIDHT, IMG_CHANNELS=IMG_CHANNELS)

    history = model.fit(
        X_train, Y_train, 
        batch_size=16,
        verbose=1,
        epochs=10,
        validation_data=(X_test,Y_test),
        shuffle=False
    )
    model.save('brain_tumor.hdf5')

    a, acc = model.evaluate(X_test, Y_test)

    print(f"Accuracy = {(acc * 100.0)}%")
    plot_loss(history)
    loss = history.history['loss']
    predict()
    return 


if __name__ == '__main__':
    main()