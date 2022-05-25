import pandas as pd
import numpy as np
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

df_train = pd.read_csv('sign_mnist_train.csv')
y_train = df_train['label'].to_numpy()
temp_train = df_train.drop(['label'], axis=1)
x_train = temp_train.to_numpy()
print(x_train.shape)
print(y_train.shape)

df_test = pd.read_csv('sign_mnist_test.csv')
y_test = df_test['label'].to_numpy()
temp_test = df_test.drop(['label'], axis=1)
x_test = temp_test.to_numpy()
print(x_test.shape)
print(y_test.shape)

index_to_letter = list('ABCDEFGHIKLMNOPQRSTUVWXY')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255
x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))
x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))

def Model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(25, activation='softmax'))
	return model

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=100)

model.save('model.h5')

pred = model.predict(x_test)

for i in range(10):
	temp = np.reshape(x_test[i], (28, 28))
	plt.imshow(temp)
	haha = np.argmax(pred[i])
	plt.title(index_to_letter[haha])
	plt.show()

acc_score = accuracy_score(x_test, pred)
print('Accuracy Score = ',acc_score)

plt.figure(figsize=(28,28))
plot_confusion_matrix(x_test, pred, classes = index_to_letter, normalize=True, title='Non-Normalized Confusion matrix')
plt.show()