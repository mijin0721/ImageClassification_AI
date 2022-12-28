# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 00:26:46 2022

@author: kmj84
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import os
import pandas as pd 

model = tf.keras.models.load_model("saved_model2.h5")
#class_names = ['0','1','2','3','4']

dir = 'C:/Users/kmj84/OneDrive/바탕 화면/aipSources/test'

test100=[]
for filename in os.listdir(dir):
    img=Image.open(dir+'/'+filename)
    x=np.asarray(img.resize([128,128]))/255.0
    test100.append(x)

test10=np.asarray(test100)

pred=model.predict(test10)


n=len(test10)
names=os.listdir(dir)

prediction=[]
for i in range(n):
    pred1 = np.argmax(pred[i])
    if pred1 == 0: #고양이
        #print(names[i], '0')
        prediction.append(0)
    elif pred1 == 1: #강아지
        #print(names[i], '1')
        prediction.append(1)
    elif pred1 == 2: #코끼리
        #print(names[i], '2')
        prediction.append(2)
    elif pred1 == 3: #말
        #print(names[i], '3')
        prediction.append(3)
    else: #사자
        #print(names[i], '4')
        prediction.append(4)
        
df=pd.DataFrame(
    {"data":names,
    "label":prediction}
)        
df

df.sort_values("data")

df.to_csv("test.csv", index=False)

'''
pred = list(pred)

for i in range(len(pred)):
    pred[i] = list(pred[i])
    
result = []
for p in pred:
    result.append(p.index(max(p)))

df = pd.read_csv('C:/Users/kmj84/OneDrive/바탕 화면/aipSources/submission (4).csv')
df['label'] = result
df.to_csv('C:/Users/kmj84/OneDrive/바탕 화면/aipSources/submission (4).csv', index=False)
'''