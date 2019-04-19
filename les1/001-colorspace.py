#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Загрузка изображения из файла
import cv2


# In[ ]:


img = cv2.imread('./lena.png', cv2.IMREAD_COLOR)


# In[ ]:


print('type: ', type(img))
print('shape: ', img.shape)


# In[ ]:


#Визуализация в matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
plt.imshow(img)


# In[ ]:


#изменяем порядок каналов для визуализации (b, g, r) -> (r, g, b)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[ ]:


#Переходим в пространство цветов HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# In[ ]:


plt.figure(1, figsize = (16, 4))
plt.subplot(141)
plt.imshow(img[...,::-1])
plt.subplot(142)
plt.imshow(img_hsv[...,0], cmap='gray')
plt.title('Hue')
plt.subplot(143)
plt.imshow(img_hsv[...,1], cmap='gray')
plt.title('Saturation')
plt.subplot(144)
plt.imshow(img_hsv[...,2], cmap='gray')
plt.title('Value');


# In[ ]:


plt.imshow(img_hsv)

