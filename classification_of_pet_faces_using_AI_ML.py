import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as pp_1 
from tensorflow.keras.layers import RandomFlip, RandomRotation, Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

images_fp = r'C:\Users\Shree\OneDrive\Desktop\python scripts\images'
image_files = glob.glob(os.path.join(images_fp, '*.[jJ][pP][gG]')) + glob.glob(os.path.join(images_fp, '*.[jJ][pP][eE][gG]'))
image_names = [os.path.basename(file) for file in image_files]
print(f"Total image files found in directory: {len(image_names)}")
if len(image_names) == 0:
    print("Warning: No images found! Double-check your path or file extensions.")

def label_encode(label):
    if label == 'Abyssinian': return 0
    elif label == 'Bengal': return 1
    elif label == 'Birman': return 2
    elif label == 'Bombay': return 3
    elif label == 'British Shorthair': return 4
    elif label == 'Egyptian Mau': return 5
    elif label == 'american bulldog': return 6
    elif label == 'american pit bull terrier': return 7
    elif label == 'basset hound': return 8
    elif label == 'beagle': return 9
    elif label == 'boxer': return 10
    elif label == 'chihuahua': return 11
    elif label == 'english cocker spaniel': return 12
    elif label == 'english setter': return 13
    elif label == 'german shorthaired': return 14
    elif label == 'great pyrenees': return 15
    return None

features = []
labels = []
IMAGE_SIZE = (224, 224)

for name in image_names:
    label = ' '.join(name.split('_')[:-1:])
    label_encoded = label_encode(label)
    
    if label_encoded is not None:
        img = load_img(os.path.join(images_fp, name))
        img = tf.image.resize_with_pad(img_to_array(img, dtype='uint8'), *IMAGE_SIZE).numpy().astype('uint8')
        features.append(img)
        labels.append(label_encoded)

features_array = np.array(features)
labels_array = np.array(labels)
labels_one_hot = pd.get_dummies(labels_array)

num_classes = labels_one_hot.shape[1] 
print(f"Number of unique classes found in your dataset: {num_classes}")


# Split 80% train+val and 20% test
x_train, x_test, y_train, y_test = train_test_split(
    features_array, labels_one_hot, test_size=0.2, random_state=42
)
# Split the 80% into 60% train and 20% val
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.25, random_state=1
)

# making model 
data_augmentation = Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2)
])

prediction_layers = Dense(num_classes, activation='softmax')
resnet_model = ResNet50(include_top=False, pooling='avg', weights='imagenet')
resnet_model.trainable = False

inputs = Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = pp_1(x)  
x = resnet_model(x, training=False)
x = Dropout(0.2)(x)
outputs = prediction_layers(x)
model = Model(inputs, outputs)

#  COMPILATION & TRAINING
model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])

model_history = model.fit( x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=10)

#  VISUALIZATION 
acc = model_history.history['accuracy']
val_acc = model_history.history['val_accuracy']
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
epochs_range = range(10)

plt.figure(figsize=(15, 8))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='training accuracy')
plt.plot(epochs_range, val_acc, label='validation_accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='training loss')
plt.plot(epochs_range, val_loss, label='validation_loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# Show plots 
plt.show()

#  PREDICTION 
x_test_1 = np.array(x_test)
y_pred = model.predict(x_test_1)
print("Predictions complete. Shape of predictions:", y_pred.shape)