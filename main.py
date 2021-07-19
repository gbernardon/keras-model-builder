# Importing Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os


# Create 'models' folder, if it doesn't exist
if not os.path.isdir('models'):
    os.mkdir('models')

# Defining class names
class_1 = 'not_separated'
class_2 = 'separated'

# setting path to the main directory
main_dir = 'data'

# Setting path to the training directory
train_dir = os.path.join(main_dir, 'train')

# Setting path to the test directory
test_dir = os.path.join(main_dir, 'test')

# Directory with train broken tip images
train_class_1_dir = os.path.join(train_dir, class_1)

# Directory with train intact tip images
train_class_2_dir = os.path.join(train_dir, class_2)

# Directory with test broken tip image
test_class_1_dir = os.path.join(test_dir, class_1)

# Directory with test intact tip image
test_class_2_dir = os.path.join(test_dir, class_2)

# Creating a list of filenames in each directory
train_class_1_names = os.listdir(train_class_1_dir)


train_class_2_names = os.listdir(train_class_2_dir)


test_class_1_names = os.listdir(test_class_1_dir)


test_class_2_names = os.listdir(test_class_2_dir)


# Printing total number of images present in each set
print('Total number of images in training set:', len(train_class_1_names
                                                     + train_class_2_names))

print("Total number of images in test set:", len(test_class_1_names
                                                 + test_class_2_names))

# get the directory to each image file in the train set
class_1_pic = [os.path.join(train_class_1_dir, filename) for filename in train_class_1_names[:8]]
class_2_pic = [os.path.join(train_class_2_dir, filename) for filename in train_class_2_names[:8]]


# merge broken and intact lists
merged_list = class_1_pic + class_2_pic



# Data Preprocessing and Augmentation

# Generate training, testing and validation batches
dgen_train = ImageDataGenerator(rescale=1. / 255,
                                validation_split=0.2,  # using 20% of training data for validation
                                zoom_range=0.2,
                                horizontal_flip=True)

dgen_validation = ImageDataGenerator(rescale=1. / 255)

dgen_test = ImageDataGenerator(rescale=1. / 255)

# Parameters
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_MODE = 'categorical'

# Connecting the ImageDataGenerator objects to our dataset
train_generator = dgen_train.flow_from_directory(train_dir,
                                                 target_size=TARGET_SIZE,
                                                 subset='training',
                                                 batch_size=BATCH_SIZE,
                                                 class_mode=CLASS_MODE)

# CONFIRM if it should be dgen_validation.flow...
validation_generator = dgen_train.flow_from_directory(train_dir,
                                                      target_size=TARGET_SIZE,
                                                      subset='validation',
                                                      batch_size=BATCH_SIZE,
                                                      class_mode=CLASS_MODE)

test_generator = dgen_test.flow_from_directory(test_dir,
                                               target_size=TARGET_SIZE,
                                               batch_size=BATCH_SIZE,
                                               class_mode=CLASS_MODE)

# Get the class indices
print('\n Class indices:')
print(train_generator.class_indices)

# Get the image shape
print('\n Image shape:')
print(train_generator.image_shape)

# Building CNN Model
model = Sequential()
model.add(Conv2D(32, (5, 5), padding='same', activation='relu',
                 input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='sigmoid'))

model.summary()

# Compile the Model
model.compile(Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(train_generator,
                    epochs=20,
                    validation_data=validation_generator,
                    callbacks=[
                    # Stopping our training if val_accuracy doesn't improve after 20 epochs
                    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20),

                    # Saving the best weights of our model in the model directory

                    # We don't want to save just the weight, but also the model architecture
                    tf.keras.callbacks.ModelCheckpoint('models/model_{val_accuracy:.3f}.h5',
                                                       save_best_only=True,
                                                       save_weights_only=False,
                                                       monitor='val_accuracy')
])


# Performance Evaluation
history.history.keys()

# Plot graph between training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Training and Validation Losses')
plt.xlabel('epoch')

# Plot graph between training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training', 'Validation'])
plt.xlabel('epoch')

# loading the best performing model
model = tf.keras.models.load_model(' ') # choose model with best accuracy

# Getting test accuracy and loss
test_loss, test_acc = model.evaluate(test_generator)
print('Test loss: {} Test Acc: {}'.format(test_loss, test_acc))