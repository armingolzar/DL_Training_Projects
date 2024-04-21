import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from joblib import dump
plt.style.use('ggplot')


def load_data():
    all_data = []
    all_label = []
    for index, item in enumerate(glob.glob('./train/*.*')):
        img = cv2.imread(item)
        img_re = cv2.resize(img, (32, 32))
        img_re = img_re/255.0
        all_data.append(img_re)
        
        
        label = item.split('/')[2].split('.')[0]
        all_label.append(label)

        if index% 100 == 0 :
            print('[INFO]: {}/25000 processed'.format(index))

    all_data = np.array(all_data)  
    dataTrain, dataTest, labelTrain, labelTest = train_test_split(all_data, all_label, test_size=0.2, random_state=42)
    
    return dataTrain, dataTest, labelTrain, labelTest


def CNN():
    net = models.Sequential([
                            layers.Conv2D(64, (3, 3), activation='relu', input_shape = (32, 32, 3)),
                            layers.MaxPool2D(),
                            layers.Conv2D(32, (3, 3), activation='relu'),
                            layers.MaxPool2D(),
                            layers.Flatten(),
                            layers.Dense(100, activation='relu'),
                            layers.Dense(50, activation='relu'),
                            layers.Dense(10, activation='relu'),
                            layers.Dense(2, activation='softmax')
                            ])

    net.compile(
                optimizer='Adam',
                metrics=['accuracy'],
                loss = 'sparse_categorical_crossentropy')

    return net


def visualization(H):
    plt.plot(H.history['accuracy'], label = 'train_accuracy')
    plt.plot(H.history['val_accuracy'], label = 'test_accuracy')
    plt.plot(H.history['loss'], label = 'train_loss')
    plt.plot(H.history['val_loss'], label = 'test_loss')
    plt.xlabel('epochs')
    plt.ylabel('accuracy/loss')
    plt.title('cat v dog classification')
    plt.legend()
    plt.show() 


x_train, x_test, y_train, y_test = load_data()
net = CNN()
H = net.fit(x_train, y_train, batch_size=32, validation_data=(x_test, y_test), epochs=30)
dump(H, 'catVdogCNN.h5')
visualization(H)

