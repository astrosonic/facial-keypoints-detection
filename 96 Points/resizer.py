#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import numpy as np
import pandas as pd


# In[67]:


class Resizer():
    def __init__(self,width,height,chin_ratio=1):
        self.width=width
        self.height=height
        self.chin_ratio=chin_ratio
        self.face=cv.CascadeClassifier("haarcascade_frontalface_default.xml")

    def _detect_face(self,image):
        faces=self.face.detectMultiScale(image)
        return faces
    
    def _crop_face(self,image,faces):
        images=[]
        for (x,y,z,w) in faces:
            images.append(image[y:y+int(w*self.chin_ratio),x:x+int(z*self.chin_ratio)])
        return images
    
    def _resize(self,images,data,faces):
        image=None
        for i in range(len(images)):
            if(min(data[0])>faces[i][0] and max(data[0])<(faces[i][0]+faces[i][2])):
                data[0]=data[0]-faces[i][0]
                data[1]=data[1]-faces[i][1]
                initial_shape=images[i].shape
                image=cv.resize(images[i],(self.width,self.height))
                data[0]=data[0]*(self.width*1.0/initial_shape[1])
                data[1]=data[1]*(self.height*1.0/initial_shape[0])
        data=data.values.reshape(-1,1)
        data=(data-(self.width/2))/(self.width/2)
        image=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        image=image/255.0
        image=image.astype(np.float32)
        return [image,data]
        
    def get_resized_withdata(self,image,data):
        #faces=self._detect_face(image)
        x=max(0,int(min(data[0])-50))
        y=max(0,int(min(data[1])-100))
        z=int(max(data[0])+50)-x
        w=int(max(data[1])+100)-y
        faces=np.array([[x,y,z,w]])        
        images=self._crop_face(image,faces)
        return self._resize(images,data,faces)
    
    def get_resized_withoutdata(self,image):
        faces=self._detect_face(image)
        images=self._crop_face(image,faces)
        for i in range(len(images)):
            images[i]=cv.resize(images[i],(self.width,self.height))
            images[i]=images[i]/255.0
            images[i]=images[i].astype(np.float32)
        return [images,faces]

    def get_original_data(self,data):
        data=(data*self.width/2)+(self.width/2)
        return data

