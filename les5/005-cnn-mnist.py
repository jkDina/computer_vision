#!/usr/bin/env python
# coding: utf-8

# # Классификация MNIST сверточной сетью

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow import keras


# In[ ]:


train = np.loadtxt('../002/data/digit/train.csv', delimiter=',', skiprows=1)
test = np.loadtxt('../002/data/digit/test.csv', delimiter=',', skiprows=1)


# In[ ]:


# сохраняем разметку в отдельную переменную
train_label = train[:, 0]

# приводим размерность к удобному для обаботки виду
# добавляем размерность канала
train_img = np.resize(train[:, 1:], (train.shape[0], 28, 28, 1))
test_img = np.resize(test, (test.shape[0], 28, 28, 1))


# ## Визуализируем исходные данные

# In[ ]:


fig = plt.figure(figsize=(20, 10))
for i, img in enumerate(train_img[0:5, :], 1):
    subplot = fig.add_subplot(1, 5, i)
    plt.imshow(img[:,:,0], cmap='gray');
    subplot.set_title('%s' % train_label[i - 1]);


# ## Разбиваем выборку на обучение и валидацию

# In[ ]:


from sklearn.model_selection import train_test_split
y_train, y_val, x_train, x_val = train_test_split(
    train_label, train_img, test_size=0.2, random_state=42)


# ## Собираем сверточную сеть для обучения

# In[ ]:


seed = 123457
kernek_initializer = keras.initializers.glorot_normal(seed=seed)
bias_initializer = keras.initializers.normal(stddev=1., seed=seed)

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(6, 
                              kernel_size=(5, 5), 
                              padding='same', 
                              activation='relu', 
                              input_shape=x_train.shape[1:],
                              bias_initializer=bias_initializer,
                              kernel_initializer=kernek_initializer))

model.add(keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'))

model.add(keras.layers.Conv2D(16, 
                              kernel_size=(5, 5),
                              padding='valid',
                              activation='relu', 
                              bias_initializer=bias_initializer,
                              kernel_initializer=kernek_initializer))

model.add(keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(32, activation='relu',
                             bias_initializer=bias_initializer,
                             kernel_initializer=kernek_initializer))

model.add(keras.layers.Dense(10, activation='softmax',
                             bias_initializer=bias_initializer,
                             kernel_initializer=kernek_initializer))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# ## Выводим информацию о модели

# In[ ]:


model.summary()


# ## One hot encoding разметки

# In[ ]:


y_train_labels = keras.utils.to_categorical(y_train)


# In[ ]:


y_train[:10]


# In[ ]:


y_train_labels[:10]


# ## Запускаем обучение

# In[ ]:


model.fit(x_train, 
          y_train_labels,
          batch_size=32, 
          epochs=5,
          validation_split=0.2)


# ## Предсказываем класс объекта

# In[ ]:


pred_val = model.predict_classes(x_val)


# In[ ]:


pred_val[:10]


# ## Оцениваем качество решение на валидационной выборке

# In[ ]:


from sklearn.metrics import accuracy_score
print('Accuracy: %s' % accuracy_score(y_val, pred_val))


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_val, pred_val))


# In[ ]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_val, pred_val))


# ## Предсказания на тестовыйх данных

# In[ ]:


pred_test = model.predict_classes(test_img)


# ## Визуализируем предсказания

# In[ ]:


fig = plt.figure(figsize=(20, 10))
indices = np.random.choice(range(len(test_img)), 5)
img_prediction = zip(test_img[indices], pred_test[indices])
for i, (img, pred) in enumerate(img_prediction, 1):
    subplot = fig.add_subplot(1, 5, i)
    plt.imshow(img[...,0], cmap='gray');
    subplot.set_title('%d' % pred);


# ## Готовим файл для отправки

# In[ ]:


with open('submit.txt', 'w') as dst:
    dst.write('ImageId,Label\n')
    for i, p in enumerate(pred_test, 1):
        dst.write('%s,%d\n' % (i, p))


# In[ ]:


# Your submission scored 0.96814

