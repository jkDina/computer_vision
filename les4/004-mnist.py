#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


train = np.loadtxt('../../002/data/digit/train.csv', delimiter=',', skiprows=1)
test = np.loadtxt('../../002/data/digit/test.csv', delimiter=',', skiprows=1)


# In[3]:


# сохраняем разметку в отдельную переменную
train_label = train[:, 0]
# приводим размерность к удобному для обаботки виду
train_img = np.resize(train[:, 1:], (train.shape[0], 28, 28))
test_img = np.resize(test, (test.shape[0], 28, 28))


# ## Визуализируем исходные данные

# In[5]:


fig = plt.figure(figsize=(20, 10))
for i, img in enumerate(train_img[0:5], 1):
    subplot = fig.add_subplot(1, 7, i)
    plt.imshow(img, cmap='gray');
    subplot.set_title('%s' % train_label[i - 1]);


# ## Вычисляем X и Y составляющие градиента с помощью оператора Собеля

# In[6]:


train_sobel_x = np.zeros_like(train_img)
train_sobel_y = np.zeros_like(train_img)
for i in range(len(train_img)):
    train_sobel_x[i] = cv2.Sobel(train_img[i], cv2.CV_64F, dx=1, dy=0, ksize=3)
    train_sobel_y[i] = cv2.Sobel(train_img[i], cv2.CV_64F, dx=0, dy=1, ksize=3)

test_sobel_x = np.zeros_like(test_img)
test_sobel_y = np.zeros_like(test_img)
for i in range(len(test_img)):
    test_sobel_x[i] = cv2.Sobel(test_img[i], cv2.CV_64F, dx=1, dy=0, ksize=3)
    test_sobel_y[i] = cv2.Sobel(test_img[i], cv2.CV_64F, dx=0, dy=1, ksize=3)


# ## Вычисляем угол и длину вектора градиента

# In[7]:


train_g, train_theta = cv2.cartToPolar(train_sobel_x, train_sobel_y)
test_g, test_theta = cv2.cartToPolar(test_sobel_x, test_sobel_y)


# In[8]:


fig = plt.figure(figsize=(20, 10))
for i, img in enumerate(train_g[:5], 1):
    subplot = fig.add_subplot(1, 7, i)
    plt.imshow(img, cmap='gray');
    subplot.set_title('%s' % train_label[i - 1]);
    subplot = fig.add_subplot(3, 7, i)
    plt.hist(train_theta[i - 1].flatten(),
             bins=16, weights=train_g[i - 1].flatten())


# ## Вычисляем гистограммы градиентов

# In[9]:


# Гистограммы вычисляются с учетом длины вектора градиента
train_hist = np.zeros((len(train_img), 16))
for i in range(len(train_img)):
    hist, borders = np.histogram(train_theta[i],
                                 bins=16,
                                 range=(0., 2. * np.pi),
                                 weights=train_g[i])
    train_hist[i] = hist
    
test_hist = np.zeros((len(test_img), 16))
for i in range(len(test_img)):
    hist, borders = np.histogram(test_theta[i],
                                 bins=16,
                                 range=(0., 2. * np.pi),
                                 weights=test_g[i])
    test_hist[i] = hist


# ## Нормируем вектор гистограммы

# In[10]:


train_hist = train_hist / np.linalg.norm(train_hist, axis=1)[:, None]
test_hist = test_hist / np.linalg.norm(test_hist, axis=1)[:, None]


# ## Разбиваем выборку на обучение и валидацию

# In[11]:


from sklearn.model_selection import train_test_split
y_train, y_val, x_train, x_val = train_test_split(
    train_label, train_hist, test_size=0.2, random_state=42)


# ## Собираем полносвязную сеть для обучения

# In[13]:


model = keras.models.Sequential()

model.add(keras.layers.Dense(32,
                             input_dim=x_train.shape[1],
                             activation='relu'))

model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# ## Выводим информацию о модели

# In[17]:


model.summary()


# ## One hot encoding разметки

# In[21]:


y_train.shape


# In[19]:


from keras.utils import np_utils
y_train_labels = np_utils.to_categorical(y_train)


# ## Запускаем обучение

# In[ ]:


model.fit(x_train, y_train_labels, 
          batch_size=32, validation_split=0.2,
          epochs=10)


# ## Предсказываем класс объекта

# In[30]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


pred_val = model.predict_classes(x_val)


# In[ ]:


pred_proba = model.predict_proba(x_val)


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


pred_test = model.predict_classes(test_hist)


# ## Визуализируем предсказания

# In[ ]:


fig = plt.figure(figsize=(20, 10))
for i, img in enumerate(test_img[0:5], 1):
    subplot = fig.add_subplot(1, 7, i)
    plt.imshow(img, cmap='gray');
    subplot.set_title('%s' % pred_test[i - 1]);


# ## Готовим файл для отправки

# In[75]:


with open('submit.txt', 'w') as dst:
    dst.write('ImageId,Label\n')
    for i, p in enumerate(pred_test, 1):
        dst.write('%s,%s\n' % (i, p))


# In[ ]:


# Your submission scored 0.59843

