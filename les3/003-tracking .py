#!/usr/bin/env python
# coding: utf-8

# # Пример работы трекера лица по видео с веб-камеры
# 
# для корректной работы примера необходимо установить пакет opencv-contrib: pip install --upgrade opencv-python opencv-contrib-python

# In[1]:


import cv2


# ## Загружает предобученную модель детектора лица

# In[2]:


face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')


# ## Запускаем процесс обработки видеопотка

# In[4]:


camera = cv2.VideoCapture(0)

kcf_tracker = None  # объект трекера инициализируется при первой детекции лица

while(1):
    ret, frame = camera.read()
    # frame = cv2.bilateralFilter(frame, 5, 50, 100)
    frame = cv2.flip(frame, 1)
    
    kcf_tracker_box = None  # результат работы трекера
    
    if kcf_tracker is not None:
        # обновляем трекер и получаем результат трекинга
        ok, box = kcf_tracker.update(frame)
        # сохраняем результат трекинга
        if ok:
            kcf_tracker_box = box
    
    # преобразуем изображение в чернобелый формат
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # запускаем детектор лиц
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)
    
    # инициализируем трекер первой детекцией
    if len(faces) != 0 and kcf_tracker is None:
        kcf_tracker = cv2.TrackerKCF_create()
        (x, y, w, h) = faces[0]
        kcf_tracker.init(frame, (x,y,w,h))
    
    for (x,y,w,h) in faces:
        # отрисовываем детекцию лиц
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0), 2)

    # отрисовываем результат трекера
    if kcf_tracker_box is not None: 
        (x, y, w, h) = map(int, kcf_tracker_box)
        cv2.rectangle(frame, (x,y),(x+w,y+h), (0, 0, 255), 2)
    
    cv2.imshow('Tracking example', frame)
    interrupt=cv2.waitKey(10)
    
    # выход по нажатию на клавишу 'q'
    if  interrupt & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




