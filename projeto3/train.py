import numpy as np
import sklearn as sk
import tensorflow as tf

def limit_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def train_and_save(path_to_model, path_to_train, path_to_val, epochs, steps_per_epoch, validation_steps, batch_size_train, batch_size_val):

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
    train_it = datagen.flow_from_directory(path_to_train, class_mode='categorical', batch_size=batch_size_train, target_size=(160,160))
    val_it = datagen.flow_from_directory(path_to_val, class_mode='categorical', batch_size=batch_size_val, target_size=(160,160))

    model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)   # weights pre-trained for imagenet, include_top=False means fully-connected layers are not included on the top of the network
    model.trainable = False     # freeze all model layers
    model.summary()             # to verify there is no treinable params anymore

    x = model.output
    x = tf.keras.layers.Dense(2, activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    model = tf.keras.models.Model(inputs=model.input, outputs=x)
    model.summary()

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # print(model.input.shape)

    model.fit(train_it, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=val_it, validation_steps=validation_steps)
    model.save(path_to_model)

def main():
    # function to alocate memory as it is used
    limit_gpu()

    # hiperparameters
    epochs = 20
    steps_per_epoch = 16#64
    validation_steps = 16
    batch_size_train = 32
    batch_size_val = 32
    path_to_model = 'saved_model/'
    path_to_train = 'cats_and_dogs_filtered/train/'
    path_to_val = 'cats_and_dogs_filtered/validation/'

    # train and saves the model
    train_and_save(path_to_model, path_to_train, path_to_val, epochs, steps_per_epoch, validation_steps, batch_size_train, batch_size_val)

if __name__ == "__main__":
    main()
