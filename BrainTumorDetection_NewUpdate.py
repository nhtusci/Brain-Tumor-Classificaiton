import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define functions to create dataframes from directories
def train_df(tr_path):
    classes, class_paths = zip(*[(label, os.path.join(tr_path, label, image))
                                 for label in os.listdir(tr_path) if os.path.isdir(os.path.join(tr_path, label))
                                 for image in os.listdir(os.path.join(tr_path, label))])
    tr_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return tr_df

def test_df(ts_path):
    classes, class_paths = zip(*[(label, os.path.join(ts_path, label, image))
                                 for label in os.listdir(ts_path) if os.path.isdir(os.path.join(ts_path, label))
                                 for image in os.listdir(os.path.join(ts_path, label))])
    ts_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return ts_df

# Load data
tr_df = train_df('./Brain_Diseases_Dataset/Training')
ts_df = test_df('./Brain_Diseases_Dataset/Testing')

# Visualization of image counts
plt.figure(figsize=(15, 7))
ax = sns.countplot(data=tr_df, y='Class')
plt.title('Count of Training Images in Each Class', fontsize=20)
ax.bar_label(ax.containers[0])
plt.savefig('Count_train_Images_of_each_class.png')
plt.show()

plt.figure(figsize=(15, 7))
ax = sns.countplot(data=ts_df, y='Class', palette='viridis')
plt.title('Count of Testing Images in Each Class')
ax.bar_label(ax.containers[0])
plt.savefig('Count_test_Images_of_each_class.png')
plt.show()

# Seed for reproducibility
seed = 50

# Split validation and test set from test data
valid_df, ts_df = train_test_split(ts_df, train_size=0.5, random_state=seed, stratify=ts_df['Class'])

# Data Augmentation with more variation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
])

# ImageDataGenerators
batch_size = 32
img_size = (299, 299)

_gen = ImageDataGenerator(
    rescale=1/255,
    brightness_range=(0.8, 1.2),
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

valid_ts_gen = ImageDataGenerator(rescale=1/255)

# Flow from DataFrame
tr_gen = _gen.flow_from_dataframe(tr_df, x_col='Class Path', y_col='Class',
                                  batch_size=batch_size, target_size=img_size, class_mode='categorical')
valid_gen = valid_ts_gen.flow_from_dataframe(valid_df, x_col='Class Path', y_col='Class',
                                             batch_size=batch_size, target_size=img_size, class_mode='categorical')
ts_gen = valid_ts_gen.flow_from_dataframe(ts_df, x_col='Class Path', y_col='Class',
                                          batch_size=batch_size, target_size=img_size, class_mode='categorical', shuffle=False)

# Model Architecture
img_shape = (299, 299, 3)
base_model = tf.keras.applications.Xception(include_top=False, weights="imagenet",
                                            input_shape=img_shape, pooling='max')

# Freeze the first 100 layers to prevent overfitting on the training data
for layer in base_model.layers[:100]:
    layer.trainable = False

model = Sequential([
    data_augmentation,  # Apply data augmentation before the base model
    base_model,
    Flatten(),
    Dropout(rate=0.4),  # Increase dropout
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # Add L2 regularization
    Dropout(rate=0.4),  # Increase dropout
    Dense(4, activation='softmax')
])

# Compile Model
model.compile(Adamax(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy', Precision(), Recall()])

# Xây dựng mô hình trước khi hiển thị tóm tắt và biểu đồ
model.build((None,) + img_shape)

# Hiển thị tóm tắt của model
model.summary()

# Vẽ cấu trúc model
tf.keras.utils.plot_model(model, show_shapes=True)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Training
hist = model.fit(tr_gen,
                 epochs=30,
                 validation_data=valid_gen,
                 shuffle=False,
                 callbacks=[early_stopping, reduce_lr])

# History keys
hist.history.keys()
print(hist.history.keys())
#Visualize
tr_acc = hist.history['accuracy']
tr_loss = hist.history['loss']
tr_per = hist.history['precision']
tr_recall = hist.history['recall']


val_acc = hist.history['val_accuracy']
val_loss = hist.history['val_loss']
val_per = hist.history['val_precision']
val_recall = hist.history['val_recall']

index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]
index_acc = np.argmax(val_acc)
acc_highest = val_acc[index_acc]
index_precision = np.argmax(val_per)
per_highest = val_per[index_precision]
index_recall = np.argmax(val_recall)
recall_highest = val_recall[index_recall]

Epochs = [i + 1 for i in range(len(tr_acc))]
loss_label = f'Best epoch = {str(index_loss + 1)}'
acc_label = f'Best epoch = {str(index_acc + 1)}'
per_label = f'Best epoch = {str(index_precision + 1)}'
recall_label = f'Best epoch = {str(index_recall + 1)}'

plt.figure(figsize=(20, 12))
plt.style.use('fivethirtyeight')

plt.subplot(2, 2, 1)
plt.plot(Epochs, tr_loss, 'r', label='Training loss')
plt.plot(Epochs, val_loss, 'g', label='Validation loss')
plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(Epochs, tr_acc, 'r', label='Training Accuracy')
plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(Epochs, tr_per, 'r', label='Precision')
plt.plot(Epochs, val_per, 'g', label='Validation Precision')
plt.scatter(index_precision + 1, per_highest, s=150, c='blue', label=per_label)
plt.title('Precision and Validation Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(Epochs, tr_recall, 'r', label='Recall')
plt.plot(Epochs, val_recall, 'g', label='Validation Recall')
plt.scatter(index_recall + 1, recall_highest, s=150, c='blue', label=recall_label)
plt.title('Recall and Validation Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)

plt.suptitle('Model Training Metrics Over Epochs', fontsize=16)
plt.savefig('Model Training Metrics Over Epochs.png')
plt.show()

train_score = model.evaluate(tr_gen, verbose=1)
valid_score = model.evaluate(valid_gen, verbose=1)
test_score = model.evaluate(ts_gen, verbose=1)

print(f"Train Loss: {train_score[0]:.4f}")
print(f"Train Accuracy: {train_score[1]*100:.2f}%")
print('-' * 20)
print(f"Validation Loss: {valid_score[0]:.4f}")
print(f"Validation Accuracy: {valid_score[1]*100:.2f}%")
print('-' * 20)
print(f"Test Loss: {test_score[0]:.4f}")
print(f"Test Accuracy: {test_score[1]*100:.2f}%")
preds = model.predict(ts_gen)
y_pred = np.argmax(preds, axis=1)

# Create class_dict from the test generator
class_dict = {v: k for k, v in ts_gen.class_indices.items()}
labels = list(class_dict.values())  # This will contain class names for labels

# Confusion matrix visualization
cm = confusion_matrix(ts_gen.classes, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('Confusion matrix.png')
plt.show()

# Classification report
clr = classification_report(ts_gen.classes, y_pred, target_names=labels)
print(clr)

import matplotlib.pyplot as plt
import numpy as np

def plot_prediction_results(model, data_gen, class_dict, save_path="prediction_results.png", num_images=20):
    # Lấy các dự đoán và nhãn thực tế từ generator
    preds = model.predict(data_gen, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = data_gen.classes
    
    # Lấy ảnh và nhãn từ generator
    images = []
    for i in range(len(data_gen)):
        batch_images, _ = data_gen[i]
        images.extend(batch_images)
    
    # Chọn số lượng hình ảnh để hiển thị
    num_images = min(num_images, len(y_true))
    plt.figure(figsize=(15, 15))
    
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        
        # Lấy ảnh và nhãn dự đoán
        img = images[i]
        true_label = y_true[i]
        pred_label = y_pred[i]
        
        # Tạo tiêu đề hiển thị
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {class_dict[true_label]}\nPred: {class_dict[pred_label]}", color=color)
        
        # Hiển thị ảnh
        plt.imshow(img)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Biểu đồ đã được lưu tại: {save_path}")

# Sử dụng hàm
plot_prediction_results(model, ts_gen, class_dict, save_path="prediction_results.png", num_images=20)

#---------------------------------------
model.save('brain_tumor_classifier.h5')
print("Model saved as brain_tumor_classifier.h5")