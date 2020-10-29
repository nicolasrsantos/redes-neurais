'''
    Exercicio 4 - tensorflow/keras
    Redes Neurais SCC5809 - ICMC-USP 2020

    Alunos:     Nícolas Roque dos Santos
                Tales Somensi
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def get_model():
    # model = tf.keras.Sequential(
    #     [
    #         tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    #         tf.keras.layers.Dense(64, activation='relu'),
    #         tf.keras.layers.Dense(10)
    #     ]
    # )
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu'),  # 16 filters, kernel_size = size of filters (must be odd)
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(32, kernel_size=(7, 7), activation='relu'),  # 32 filters, kernel_size = size of filters (must be odd)
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10),
            tf.keras.layers.Softmax()
        ]
    )
    # print(model.summary(), '\n')
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

def get_files():
    imgs_X = []
    imgs_y = []
    # import files for digit '1'
    folder_path = 'test_data/1/'
    for img in os.listdir(folder_path):
        img = os.path.join(folder_path, img)
        img = tf.keras.preprocessing.image.load_img(img, target_size=(28,28), color_mode='grayscale')
        img = tf.keras.preprocessing.image.img_to_array(img)
        imgs_X.append(img)
        imgs_y.append(1)
    # import files for digit '2'
    folder_path = 'test_data/2/'
    for img in os.listdir(folder_path):
        img = os.path.join(folder_path, img)
        img = tf.keras.preprocessing.image.load_img(img, target_size=(28,28), color_mode='grayscale')
        img = tf.keras.preprocessing.image.img_to_array(img)
        imgs_X.append(img)
        imgs_y.append(2)
    # import files for digit '4'
    folder_path = 'test_data/4/'
    for img in os.listdir(folder_path):
        img = os.path.join(folder_path, img)
        img = tf.keras.preprocessing.image.load_img(img, target_size=(28,28), color_mode='grayscale')
        img = tf.keras.preprocessing.image.img_to_array(img)
        imgs_X.append(img)
        imgs_y.append(4)
    # import files for digit '7'
    folder_path = 'test_data/7/'
    for img in os.listdir(folder_path):
        img = os.path.join(folder_path, img)
        img = tf.keras.preprocessing.image.load_img(img, target_size=(28,28), color_mode='grayscale')
        img = tf.keras.preprocessing.image.img_to_array(img)
        imgs_X.append(img)
        imgs_y.append(7)

    imgs_X = np.array(imgs_X)
    imgs_y = np.array(imgs_y)
    # print('imgs_X.shape', imgs_X.shape)
    # print('imgs_y.shape', imgs_y.shape)

    return imgs_X, imgs_y

def main():

    num_epochs = 2

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    # scale images to the 0~1 range
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    # reshape images from (28, 28) to (28, 28, 1)
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    # get the model
    model = get_model()
    print('\ntraining model for %d epochs:\n' % (num_epochs))
    # train
    model.fit(X_train, y_train, epochs=num_epochs)
    # open images for inference
    imgs_X, imgs_y = get_files()
    # evaluate trained model
    test_loss, test_acc = model.evaluate(imgs_X, imgs_y, verbose=0)
    print('\nevaluation:\tloss: %.4f\taccuracy: %.4f' % (test_loss, test_acc))
    # predict = model.predict(imgs_X)

if __name__ == '__main__':
    main()


























#
