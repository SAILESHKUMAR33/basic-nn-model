# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

A neural network is a computer program inspired by how our brains work. It's used to solve problems by finding patterns in data. Imagine a network of interconnected virtual "neurons." Each neuron takes in information, processes it, and passes it along.
A Neural Network Regression Model is a type of machine learning algorithm that is designed to predict continuous numeric values based on input data. It utilizes layers of interconnected nodes, or neurons, to learn complex patterns in the data. The architecture typically consists of an input layer, one or more hidden layers with activation functions, and an output layer that produces the regression predictions.
This model can capture intricate relationships within data, making it suitable for tasks such as predicting prices, quantities, or any other continuous numerical outputs.

## Neural Network Model

![image](https://github.com/SAILESHKUMAR33/basic-nn-model/assets/113497410/4db866a2-c5e9-4b5b-8d19-da3af1b88dd3)



## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
Developed by: Saileshkumar A

Reg no: 212222230126

```python
import pandas as pd
from google.colab import auth
import gspread
from google.auth import default
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

auth.authenticate_user()
creds,_=default()
gc=gspread.authorize(creds)

worksheet=gc.open('deeplearning').sheet1
data=worksheet.get_all_values()

data1=pd.DataFrame(data[1:],columns=data[0])
data1=data1.astype({'input':'float'})
data1=data1.astype({'output':'float'})
data1.head()

x=data1[['input']].values
y=data1[['output']].values



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=30)


scaler=MinMaxScaler()

scaler.fit(x_train)

x_train1=scaler.transform(x_train)




ai_brain=Sequential([Dense(7,activation="relu",input_shape=[1]),Dense(14,activation="relu"),Dense(1)])

ai_brain.compile(optimizer="rmsprop",loss="mse")


ai_brain.fit(x_train,y_train,epochs=3000)


loss=pd.DataFrame(ai_brain.history.history)

loss.plot()

x_test1=scaler.transform(x_test)
ai_brain.evaluate(x_test1,y_test)



x_n1=[[11]]
x_n1_1=scaler.transform(x_n1)
ai_brain.predict(x_n1_1)
```

## Dataset Information

![image](https://github.com/SAILESHKUMAR33/basic-nn-model/assets/113497410/cb95c7be-f1e9-4418-8faf-8ce36f5d4355)



## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/SAILESHKUMAR33/basic-nn-model/assets/113497410/3a092344-75e5-4596-a1bb-abba9df6f06b)




### Test Data Root Mean Squared Error

![image](https://github.com/SAILESHKUMAR33/basic-nn-model/assets/113497410/cebabc7a-b8ad-46a8-a990-97c0dc1a7e20)



### New Sample Data Prediction

![image](https://github.com/SAILESHKUMAR33/basic-nn-model/assets/113497410/1d75d25b-728c-4b8d-910b-c90d6c4121b2)



## RESULT

Thus a neural network regression model for the given dataset is executed successfully.
