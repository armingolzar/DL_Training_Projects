from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def prepracess_data():

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    
    return x_train, y_train, x_test, y_test

def Neural_network():
    net = models.Sequential([
                    layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)),
                    layers.MaxPool2D(),
                    layers.Conv2D(64, (3, 3), activation = 'relu'),
                    layers.Flatten(),
                    layers.Dense(10, activation = 'softmax') 
                    ])
    net.compile(optimizer = 'Adam',
            metrics = ['accuracy'],
            loss = 'sparse_categorical_crossentropy')
    return net

def visualization(H):
    plt.plot(H.history['accuracy'], label = 'train_accuracy')
    plt.plot(H.history['val_accuracy'], label = 'test_accuracy')
    plt.plot(H.history['loss'], label = 'train_loss')
    plt.plot(H.history['val_loss'], label = 'test_loss')
    plt.xlabel('epochs')
    plt.ylabel('accuracy/loss')
    plt.title('cifar classification')
    plt.legend()
    plt.show() 

x_train, y_train, x_test, y_test = prepracess_data()
net = Neural_network()
H = net.fit(x_train, y_train, batch_size = 32, validation_data=(x_test, y_test), epochs=50)
visualization(H)