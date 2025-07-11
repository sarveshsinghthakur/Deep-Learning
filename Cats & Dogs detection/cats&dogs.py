# -*- coding: utf-8 -*-
"""cats&dogs.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tu3d3fNiFT0NCMaOwPAHQfsUqIT6P0wj
"""

!pip install tensorflow opencv-python

import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import ResNet50

(ds_train, ds_test), ds_info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    with_info=True
)

IMG_SIZE = 128
BATCH_SIZE = 32

def preprocess(image,label):
  image = tf.image.resize(image,(IMG_SIZE,IMG_SIZE))
  image = tf.cast(image,tf.float32)/255.0
  return image,label

ds_train = ds_train.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def dataset_to_numpy(dataset):
    images, labels = [], []
    for image, label in dataset.unbatch():
        images.append(image.numpy())
        labels.append(label.numpy())
    return np.array(images), np.array(labels)

x_train, y_train = dataset_to_numpy(ds_train)
x_test, y_test = dataset_to_numpy(ds_test)

#model 1 : CNN
models = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128,activation='relu'),
    Dense(1,activation='sigmoid')
])

models.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = models.fit(x_train,y_train,epochs=5,validation_data=(x_test,y_test))

y_pred = models.predict(x_test)
y_pred1 = (y_pred > 0.5).astype("int32").flatten()
acc = np.mean(y_pred1 == y_test)
print(f"accuracy by manual method: {acc * 100:.2f} %\n")
print(f"accuracy by tf : {models.evaluate(x_test,y_test)[1]*100} % \n")
print(f"loss by tf : {models.evaluate(x_test,y_test)[0]*100} % \n")
print(f"classification report : {classification_report(y_pred1,y_test,target_names=['Cat', 'Dog'])}")

cm = confusion_matrix(y_test, y_pred1)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve - Cats vs Dogs')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()

def predict_cat_dog(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found.")
        return -1

    img = cv2.resize(img, (128, 128))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_array = img_rgb / 255.0
    img_array = img_array.reshape(1, 128, 128, 3)
    pred = models.predict(img_array)[0][0]
    label = "Dog" if pred >= 0.5 else "Cat"
    plt.imshow(img_rgb)
    plt.title(f"Predicted: {label} ({pred*100:.2f}) %")
    plt.axis('off')
    plt.show()

    return label
predict_cat_dog("/content/download.jpg")