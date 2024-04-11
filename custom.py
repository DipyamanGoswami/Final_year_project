from keras.layers import Conv2D, MaxPool2D, AvgPool1D, Flatten, Dense, Dropout, BatchNormalization
import cv2
import numpy as np
import os
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns

root = r'morph_new'
label = 0


def preprocess(image_path):
    data = []
    for path in os.listdir(image_path):
        if path == 'frac' or path == 'fracture':
            label = 1
        else:
            label = 0

        for img_path in os.listdir(os.path.join(image_path, path)):
            im_path = os.path.join(os.path.join(image_path, path), img_path)
            img_arr = cv2.imread(im_path, 0)
            img_arr = cv2.resize(img_arr, (100, 100))
            img_arr = img_arr.reshape(img_arr.shape+(1,))
            img_arr = img_arr/255

            data.append([img_arr, label])

    return data

data_train = preprocess(root)

np.random.RandomState(seed=42).shuffle(data_train)

X = []
Y = []

for image, label in data_train:
    X.append(image)
    Y.append(label)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), input_shape=(100, 100, 1), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['accuracy'])

his = model.fit(X_train, Y_train, epochs=60, validation_split=0.2)

model.save('custom_gray_morph.h5')

#plotting of the losses
acc = his.history['accuracy']
val_acc = his.history['val_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.savefig('Custom_morph_accuracy_1.png', bbox_inches='tight')
plt.figure()

loss = his.history['loss']
val_loss = his.history['val_loss']
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.savefig('Custom_morph_loss_1', bbox_inches='tight')
plt.show()

# model = load_model('custom_gray_morph.h5')


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
plt.title('Confusion Matrix for Custom Model Morph', family='serif', size=15, pad=12)
plt.xlabel('Predicted Values', family='serif')
plt.ylabel('True Values', family='serif')
plt.show()