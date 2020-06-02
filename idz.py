# ����������� ����������� ���������
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# ��������� ������
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# ������������ ������� ������� ������
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ����������� � �������� �� 0 �� 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# ��������� ����
model = keras.Sequential([
keras.layers.Flatten(input_shape=(28, 28)),
keras.layers.Dense(128, activation='relu'),
keras.layers.Dense(128, activation='relu'),
keras.layers.Dense(10, activation='softmax')
])

# ���������� ������
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

# ���������� ������
H = model.fit(train_images, train_labels, epochs=30, validation_data = (train_images, train_labels))

# ������ ��������
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\n�������� �� ����������� ������:', test_acc)
# ������������ ������
predictions = model.predict(test_images)

# �������� ��������
loss = H.history['loss']
val_loss = H.history['val_loss']
acc = H.history['accuracy']
val_acc = H.history['val_accuracy']
epochs = range(1, len(loss) + 1)

# ������ ������
plt.plot(epochs, loss, 'bo', label='Train loss')
plt.plot(epochs, val_loss, 'b', label='Valid loss')
plt.title('Train and valid loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()

# ������ ��������
plt.plot(epochs, acc, 'bo', label='Train acc')
plt.plot(epochs, val_acc, 'b', label='Valid acc')
plt.title('Train and valid accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()