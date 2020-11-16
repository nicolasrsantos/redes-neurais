import os
import numpy as np
import sklearn as sk
import tensorflow as tf

def get_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(256, 256, 1)),
            tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu'),  # 16 filters, kernel_size = size of filters (must be odd)
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(32, kernel_size=(7, 7), activation='relu'),  # 32 filters, kernel_size = size of filters (must be odd)
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(7),
            tf.keras.layers.Softmax()
        ]
    )
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_files(path):
    imgs_X = []
    imgs_y = []
    img_class = 0
    for folder in os.listdir(path):
        for img in os.listdir(path + folder):
            img = os.path.join(path + folder, img)
            img = tf.keras.preprocessing.image.load_img(img, target_size=(256,256), color_mode='grayscale')
            img = tf.keras.preprocessing.image.img_to_array(img)
            imgs_X.append(img)
            imgs_y.append(img_class)
        img_class += 1

    imgs_X = np.array(imgs_X)
    imgs_y = np.array(imgs_y)
    
    return imgs_X, imgs_y

def main():
    X_train, y_train = get_files("train/")
    X_val, y_val = get_files("val/")

    num_epochs = 10

    # scale images to the 0~1 range
    X_train = X_train.astype('float32') / 255
    #X_val = X_val.astype('float32') / 255
    # reshape images from (28, 28) to (28, 28, 1)
    #X_train = np.expand_dims(X_train, -1)
    #X_val = np.expand_dims(X_val, -1)
    # get the model
    model = get_model()
    print('\ntraining model for %d epochs:\n' % (num_epochs))
    # train
    model.fit(X_train, y_train, epochs=num_epochs)
    # open images for inference
    #imgs_X, imgs_y = get_files()
    # evaluate trained model
    test_loss, test_acc = model.evaluate(X_val, y_val, verbose=0)
    print('\nevaluation:\tloss: %.4f\taccuracy: %.4f' % (test_loss, test_acc))
    # predict = model.predict(imgs_X)


if __name__ == '__main__':
    main()