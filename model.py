#!/usr/bin/env python
# coding: utf-8

# In[418]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import math 
from sklearn.metrics import mean_squared_error


# In[419]:


timestep = 90
days_predict = 30


# In[420]:


x_train = np.load("./data/x_train.npy")
y_train = np.load("./data/y_train.npy")
x_test = np.load("./data/x_test.npy")
y_test = np.load("./data/y_test.npy")
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[421]:


model = Sequential()
model.add(LSTM(50, return_sequences=True,input_shape=(timestep,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()


# In[422]:


#don't need mini-batches on such small data
model.fit(x_train,y_train,validation_data=(x_test,y_test), epochs=10,verbose=1)


# In[423]:


#needed to do here again, purely so can do scaler inverse 
#transform, to be able to calculuse RMSE
df = pd.read_csv('./data/ETH-CAD.csv')
df = df.iloc[1100:]
df_close = df.reset_index()['Close']

scaler = MinMaxScaler(feature_range=(0,1))
df_close = scaler.fit_transform(np.array(df_close).reshape(-1,1))


# In[424]:


train_predict = model.predict(x_train)
train_predict = scaler.inverse_transform(train_predict)
# test_predict = model.predict(x_test)
# test_predict = scaler.inverse_transform(test_predict)


# In[425]:


print(math.sqrt(mean_squared_error(y_train, train_predict)))
# print(math.sqrt(mean_squared_error(y_test, test_predict)))


# In[426]:


look_back=timestep
trainPredictPlot = np.empty_like(df_close)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
# testPredictPlot = np.empty_like(df_close)
# testPredictPlot[:, :] = np.nan
# testPredictPlot[len(train_predict)+(look_back*2)+1:len(df_close)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df_close))
plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
plt.show()


# In[427]:


training_size = int(len(df_close)*1)
test_size = int(len(df_close)) - training_size
train_data, test_data = df_close[0:training_size:],df_close[training_size:len(df_close),:1]
print(len(test_data))


# In[428]:


#len(test_data)-num_steps_back is the number to which you index here
x_input=train_data[(len(train_data))-timestep:].reshape(1,-1)
x_input.shape
temp_input=list(x_input)
temp_input=temp_input[0].tolist()
temp_input


# In[429]:


#just to make trying training all the data predicitons easier
test_data = train_data

lst_output=[]
n_steps=timestep
i=0
while(i<days_predict):
    
    if(len(temp_input)>timestep):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[430]:


print(len(df_close))


# In[431]:


day_new=np.arange(1,timestep + 1)
day_pred=np.arange(timestep + 1,timestep + 1 + days_predict)
#this num is len(df_close)-30
plt.plot(day_new,scaler.inverse_transform(df_close[len(df_close)-timestep:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[432]:


df_new=df_close.tolist()
df_new.extend(lst_output)
plt.plot(df_new[:])


# In[433]:


df_new=scaler.inverse_transform(df_new).tolist()
plt.plot(df_new)

