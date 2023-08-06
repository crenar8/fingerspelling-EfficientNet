from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from efficientnet.tfkeras import EfficientNetB0


# Loading base EfficientNet pre-trained with ImageNet weights
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(200, 200, 3))

# Adding Global Average Pooling  Layer to reduce output size
x = keras.layers.GlobalAveragePooling2D()(base_model.output)

# Adding last Layer fully connected as an output
output = keras.layers.Dense(29, activation='softmax')(x)

# Creation of the final Model
model = keras.models.Model(inputs=base_model.input, outputs=output)

# Model compiling
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Creation of the ImageDataGenerator instance for the data augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=10,  # Image rotation
    width_shift_range=0.1,  # Horizontal shift of the images
    height_shift_range=0.1,  # Vertical shift of the images
    horizontal_flip=True,
    validation_split=0.2
)

# Dataset directories
train_directory = 'dataset/training'
test_directory = 'dataset/test'
batch_size = 32
image_size = (200, 200)

# Training data generator creation
train_generator = datagen.flow_from_directory(
    train_directory,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Validation data generator creation
validation_generator = datagen.flow_from_directory(
    test_directory,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Model Training
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Trained Model saving
model.save('asl_fingerspelling_model.h5')
