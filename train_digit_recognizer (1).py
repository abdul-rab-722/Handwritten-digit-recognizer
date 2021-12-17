#!/usr/bin/env python
# coding: utf-8

# The core features of the model are as follows −
# 
# - Input layer consists of (1, 8, 28) values.
# 
# - First layer, Conv2D consists of 32 filters and ‘relu’ activation function with kernel size, (3,3).
# 
# - Second layer, Conv2D consists of 64 filters and ‘relu’ activation function with kernel size, (3,3).
# 
# - Thrid layer, MaxPooling has pool size of (2, 2).
# 
# - Fifth layer, Flatten is used to flatten all its input into single dimension.
# 
# - Sixth layer, Dense consists of 128 neurons and ‘relu’ activation function.
# 
# - Seventh layer, Dropout has 0.5 as its value.
# 
# - Eighth and final layer consists of 10 neurons and ‘softmax’ activation function.
# 
# - Use categorical_crossentropy as loss function.
# 
# - Use Adadelta() as Optimizer.
# 
# - Use accuracy as metrics.
# 
# - Use 128 as batch size.
# 
# - Use 10 as epochs.

# ### Step 1 − Import the modules

# In[1]:


from tensorflow import keras 
import tensorflow
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Flatten 
from tensorflow.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.keras import backend as K 
import numpy as np


# ### Step 2 − Load data

# In[2]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# ### Step 3 − Process the data

# Let us change the dataset according to our model, so that it can be feed into our model.

# In[3]:


img_rows, img_cols = 28, 28 

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols) 
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols) 
    input_shape = (1, img_rows, img_cols)
else: 
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1) 
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1) 
    input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 
x_train /= 255 
x_test /= 255 

y_train = tensorflow.keras.utils.to_categorical(y_train, 10) 
y_test = tensorflow.keras.utils.to_categorical(y_test, 10)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# ### Step 4 − Create the model

# In[4]:


model = Sequential() 
model.add(Conv2D(32, kernel_size = (3, 3),  
   activation = 'relu', input_shape = input_shape)) 
model.add(Conv2D(64, (3, 3), activation = 'relu')) 
model.add(MaxPooling2D(pool_size = (2, 2))) 
model.add(Dropout(0.25))
model.add(Flatten()) 
model.add(Dense(128, activation = 'relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(10, activation = 'softmax'))
model.summary()


# ### Step 5 − Compile the model

# In[5]:


model.compile(loss = tensorflow.keras.losses.categorical_crossentropy, 
   optimizer = tensorflow.keras.optimizers.Adadelta(), metrics = ['accuracy'])


# In[6]:


hist = model.fit(
   x_train, y_train, 
   batch_size = 128, 
   epochs = 10, 
   verbose = 1, 
   validation_data = (x_test, y_test)
)

print("The model has successfully trained")

model.save('mnist.h5')
print("Saving the model as mnist.h5")


# In[7]:


hist.history


# ### Step 7 − Evaluate the model

# In[8]:


score = model.evaluate(x_test, y_test, verbose = 0) 

print('Test loss:', score[0]) 
print('Test accuracy:', score[1])


# ### Step 8 − Predict

# In[9]:


pred = model.predict(x_test) 
pred = np.argmax(pred, axis = 1)[:5] 
label = np.argmax(y_test,axis = 1)[:5] 

print(pred) 
print(label)


# In[ ]:




