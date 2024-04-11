import os
import cv2
import tensorflow as tf
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, Dropout, MaxPool2D, Activation, Flatten, Dense
from keras.models import Sequential, load_model
from keras.applications import VGG16, VGG19
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

root = r'morph_new'
label = 0


def preprocess(image_path):
    data = []
    for path in os.listdir(image_path):
        if path == 'fracture' or path == 'frac':
            label = 1
        else:
            label = 0

        for img_path in os.listdir(os.path.join(image_path, path)):
            im_path = os.path.join(os.path.join(image_path, path), img_path)
            img_arr = cv2.imread(im_path)
            img_arr = cv2.resize(img_arr, (100, 100))
            img_arr = img_arr/255

            data.append([img_arr, label])

    return data


data_train = preprocess(root)
#print(len(data))

np.random.RandomState(seed=42).shuffle(data_train)

X = []
Y = []

for image, label in data_train:
    X.append(image)
    Y.append(label)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

base_model = VGG16(include_top=False, input_shape=(100, 100, 3))
model = Sequential()

for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers:
    model.add(layer)

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['accuracy'])
his = model.fit(X_train, Y_train, epochs=30, validation_split=0.2)


#plotting of the losses
acc = his.history['accuracy']
val_acc = his.history['val_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.savefig('VGG16_accuracy_morph.png', bbox_inches='tight')
plt.figure()

loss = his.history['loss']
val_loss = his.history['val_loss']
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.savefig('VGG16_loss_morph.png', bbox_inches='tight')
plt.show()

model.save('vgg16_morph_new.h5')

#model = load_model('vgg19.h5')

"""img = cv2.imread('WhatsApp Image 2024-03-15 at 18.33.44_8c15f61f.jpg')
img = cv2.resize(img, (100, 100))
img = img/255
img = np.reshape(img, ((1,)+img.shape))

if model.predict(img) > 0.5:
    print('Fracture detected')
else:
    print('No fracture detected')

model = load_model('vgg16_morph.h5')"""


Y_pred = model.predict(X_test)

y_predict = []
for values in Y_pred:
    if values >= 0.5:
        values = 1
    else:
        values = 0
    y_predict.append(values)

# PLOTTING OF THE CONFUSION MATRIX

cnf_mat = confusion_matrix(Y_test, y_predict)
group_names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
group_counts = ["{0:0.0f}".format(value) for value in cnf_mat.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cnf_mat.flatten()/np.sum(cnf_mat)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2, 2)

sns.heatmap(cnf_mat/np.sum(cnf_mat), annot=labels, fmt='', cmap='viridis')
plt.title('Confusion Matrix for VGG16 morph_new', family='serif', size=20, pad=12)
plt.xlabel('Predicted Values', family='serif')
plt.ylabel('True Values', family='serif')
plt.show()