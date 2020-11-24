import tensorflow as tf

def load_and_test(path_to_model, path_to_test, test_batch_size, test_steps):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
    test_it = datagen.flow_from_directory(path_to_test, class_mode='categorical', batch_size=test_batch_size, target_size=(160,160))
    model = tf.keras.models.load_model(path_to_model)

    test_loss, test_acc = model.evaluate(test_it, steps=test_steps)
    print('\nevaluation:\tloss: %.4f\taccuracy: %.4f' % (test_loss, test_acc))

def main():

    # hiperparameters
    test_steps = 1
    test_batch_size = 64
    path_to_model = 'saved_model/'
    path_to_test = 'test/'

    # load model and test
    load_and_test(path_to_model, path_to_test, test_batch_size, test_steps)

if __name__ == "__main__":
    main()
