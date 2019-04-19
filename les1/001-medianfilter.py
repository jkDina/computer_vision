#!/usr/bin/env python
# coding: utf-8

# # Медианный фильтр

# In[ ]:


import cv2
import numpy as np


# In[ ]:


img = cv2.imread('./lena.png')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
plt.imshow(img[...,::-1])


# In[ ]:


img.shape


# In[ ]:


def median_filter(img, ksize):
    result = np.zeros_like(img)
    for channel in range(img.shape[-1]):
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                result[row, col, channel] = img[row, col, channel]
                # TODO: определяем текущее положение окна в координатах изображения
                # TODO: вычислить медиану в окрестности х, y для окна размера ksize
    return result


# ## Добавляем случайный шум на изображение

# In[ ]:


noisy_img = img.astype(float) + np.random.uniform(img.astype(float))
noisy_img = np.uint8(255. * np.abs(noisy_img) / np.max(np.abs(noisy_img)))
plt.imshow(noisy_img[...,::-1])


# In[ ]:


from ipywidgets import interact

def median_filter_show(ksize = 5):
    filtered = median_filter(noisy_img, ksize)
    return plt.imshow(filtered[...,::-1]);

interact(median_filter_show, ksize = (1, 20, 1));

