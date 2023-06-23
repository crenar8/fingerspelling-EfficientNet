from keras.preprocessing.image import ImageDataGenerator
from efficientnet.tfkeras import EfficientNetB0
from tensorflow import keras

# Loading of the pre-trained model
model = keras.models.load_model('asl_fingerspelling_model.h5')


test_directory = 'dataset/test'
batch_size = 32
image_size = (224, 224)

# Test data generator
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    test_directory,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Evaluation of the trained model
loss, accuracy = model.evaluate(test_generator)

# Metrics printing
print('Loss:', loss)
print('Accuracy:', accuracy)
