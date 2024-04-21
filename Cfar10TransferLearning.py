from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import SGD, Adam
import cv2 
plt.style.use('ggplot')


def prepare_data():
    LE = LabelBinarizer()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    (x_train, x_test) = x_train/255.0, x_test/255.0
    y_train = LE.fit_transform(y_train)
    y_test = LE.transform(y_test)
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = prepare_data()
    

x_train_final = []
x_test_final = []
for img in x_train:
    img_re = cv2.resize(img, (224,224), interpolation=cv2.INTER_CUBIC)
    x_train_final.append(img_re)

for img in x_test:
    img_re = cv2.resize(img, (224,224), interpolation=cv2.INTER_CUBIC)
    x_test_final.append(img_re)

for image in x_train_final:
    print(image.shape)

baseModel = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3)
                )

baseModel.trainable = False




def Nural_Network():
    net = models.Sequential([
                              layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
                            layers.BatchNormalization(),
                            layers.Conv2D(32, (3, 3), activation='relu'),
                            layers.BatchNormalization(),
                            layers.MaxPool2D(),
            
            
                            layers.Conv2D(64, (3, 3), activation='relu'),
                            layers.BatchNormalization(),
                            layers.Conv2D(64, (3, 3), activation='relu'),
                            layers.BatchNormalization(),
                            layers.MaxPool2D(),

                            baseModel,
                            layers.MaxPool2D((4, 4)),
                            layers.Flatten(),
                            layers.Dense(64, activation='relu'),
                            layers.BatchNormalization(),
                            layers.Dense(10, activation='softmax')
                            ])
    opt = Adam(lr= 0.001, decay= 0.001/25)
        
    net.compile(optimizer=opt,
                metrics = ['accuracy'],
                loss = ['categorical_crossentropy']
            )
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



aug = ImageDataGenerator(rotation_range=20,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode='nearest')


   
net = Nural_Network()
print(net.summary())

net.fit(aug.flow(x_train_final, y_train, batch_size = 8),
        steps_per_epoch= len(x_train_final)//8,
        validation_data=(x_test_final, y_test),
        epochs = 10)

visualization(net)