import numpy as np
import sklearn as sk
import tensorflow as tf

def get_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(tf.keras.layers.Conv2D(32, (7, 7), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(7, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    train_it = datagen.flow_from_directory('train/', class_mode='categorical', batch_size=64, color_mode='grayscale')
    val_it = datagen.flow_from_directory('val/', class_mode='categorical', batch_size=64, color_mode='grayscale')
    test_it = datagen.flow_from_directory('test/', class_mode='categorical', batch_size=64, color_mode='grayscale')
    
    model = get_model()
    model.fit(train_it, epochs=10, steps_per_epoch=64, validation_data=val_it, validation_steps=32)
    model.save("models/new_model")

    test_loss, test_acc = model.evaluate(test_it, steps=32)
    print('\nevaluation:\tloss: %.4f\taccuracy: %.4f' % (test_loss, test_acc))

if __name__ == "__main__":
    main()