#!/usr/bin/env python
# coding: utf-8

# # Оператор Собеля

# In[ ]:


import cv2
import numpy as np


# In[ ]:


img = cv2.imread('lena.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[ ]:


img.dtype


# In[ ]:


plt.imshow(img[..., ::-1])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
plt.imshow(img_gray, cmap='gray')


# ## Оператор собеля для вычисления X и Y составляющих градиента

# In[ ]:


sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
    
sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])


# ## Вычисляем градиенты по осям

# In[ ]:


grad_x = cv2.filter2D(img_gray, cv2.CV_32F, sobel_x)
grad_y = cv2.filter2D(img_gray, cv2.CV_32F, sobel_y)


# In[ ]:


plt.figure(2, figsize=(12, 8))
plt.subplot(121)
plt.imshow(np.abs(grad_x)/np.max(np.abs(grad_x)), cmap='gray')
plt.title('Grad X')
plt.subplot(122)
plt.imshow(np.abs(grad_y)/np.max(np.abs(grad_y)), cmap='gray')
plt.title('Grad Y');


# ## Вычисляем суммарный градиент

# In[ ]:


grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
theta = np.arctan2(grad_y, grad_x)


# In[ ]:


plt.figure(2, figsize=(12, 8))
plt.subplot(122)
plt.imshow(img_gray, cmap='gray')
plt.subplot(121)
plt.imshow(grad, cmap='gray')

