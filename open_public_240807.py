# Import necessary libraries
import os
import shutil
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define base paths
base_path = './SDNET2018'
deck_crack_path = os.path.join(base_path, 'D/CD')
deck_uncrack_path = os.path.join(base_path, 'D/UD')
pavement_crack_path = os.path.join(base_path, 'P/CP')
pavement_uncrack_path = os.path.join(base_path, 'P/UP')
wall_crack_path = os.path.join(base_path, 'W/CW')
wall_uncrack_path = os.path.join(base_path, 'W/UW')

# Create directories for train, validation, and test sets
base_dir = 'concrete_crack_split'
os.makedirs(base_dir, exist_ok=True)
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

# Function to split data into train, validation, and test sets
def prepare_data(src_dir, dest_dir, label, train_split=0.7, valid_split=0.2, test_split=0.1):
    files = os.listdir(src_dir)
    random.shuffle(files)
    total_files = len(files)
    train_files = files[:int(total_files * train_split)]
    valid_files = files[int(total_files * train_split):int(total_files * (train_split + valid_split))]
    test_files = files[int(total_files * (train_split + valid_split)):]

    # Create subdirectories
    train_label_dir = os.path.join(train_dir, label)
    valid_label_dir = os.path.join(valid_dir, label)
    test_label_dir = os.path.join(test_dir, label)

    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(valid_label_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)

    # Copy files
    for file in train_files:
        shutil.copy(os.path.join(src_dir, file), os.path.join(train_label_dir, file))
    for file in valid_files:
        shutil.copy(os.path.join(src_dir, file), os.path.join(valid_label_dir, file))
    for file in test_files:
        shutil.copy(os.path.join(src_dir, file), os.path.join(test_label_dir, file))

# Prepare data for all categories
prepare_data(deck_crack_path, 'crack', '1')
prepare_data(deck_uncrack_path, 'uncrack', '0')
prepare_data(pavement_crack_path, 'crack', '1')
prepare_data(pavement_uncrack_path, 'uncrack', '0')
prepare_data(wall_crack_path, 'crack', '1')
prepare_data(wall_uncrack_path, 'uncrack', '0')

# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)

valid_generator = valid_test_datagen.flow_from_directory(
    valid_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)

test_generator = valid_test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // valid_generator.batch_size,
    epochs=20
)

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(valid_generator, verbose=2)
print(f'Validation Accuracy: {val_accuracy}')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, verbose=2)
print(f'Test Accuracy: {test_accuracy}')

# Save the model
model.save('cnn_concrete_crack_model.h5')

# Plot training & validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

plt.show()
