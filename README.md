# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

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

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:GANANANTH H

### Register Number: 212225230070

```python

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("Exp-1.csv")
df

x=df[["Input"]].values
y=df[["Output"]].values

xt,xst,yt,yst=train_test_split(x,y,test_size=0.5,random_state=11)

scaler1=MinMaxScaler()
scaler2=MinMaxScaler()
xt=scaler1.fit_transform(xt)
xst=scaler2.fit_transform(xst)

xt1=torch.FloatTensor(xt)
xst1=torch.FloatTensor(xst)
yt1=torch.FloatTensor(yt)
yst1=torch.FloatTensor(yst)

class neuralnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(
          nn.Linear(1,20),
          nn.ReLU(),
          nn.Linear(20,10),
          nn.ReLU(),
          nn.Linear(10,1)
        )
    def forward(self,x):
        return self.network(x)

model=neuralnet()
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.1)

epochs=1000
for i in range(epochs):
    optimizer.zero_grad()
    pred=model(xt1)
    loss=criterion(pred,yt1)
    loss.backward()
    optimizer.step()
    if i%100==0:
        print(f"Epochs:{i} Loss:",loss.item())

with torch.no_grad():
    pred=model(xst1)
    loss=criterion(pred,yst1)
    print(loss.item())

input=torch.FloatTensor(scaler1.transform([[18]]))
pred=model(input)
print(pred.item())


epochs = 1000
loss_history = []  # <-- ADD THIS

for i in range(epochs):
  optimizer.zero_grad()
  pred = model(xt1)
  loss = criterion(pred, yt1)
  loss.backward()
  optimizer.step()

  loss_history.append(loss.item())  # <-- ADD THIS

  if i % 100 == 0:
    print(f"Epochs:{i} Loss:", loss.item())




plt.figure(figsize=(8, 4))
plt.plot(loss_history, label="Training Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training Loss Over Time")
plt.legend()
plt.grid(True)
plt.show()

with torch.no_grad():
  model.eval()
  preds_test = model(xst1).numpy()

plt.figure(figsize=(8, 4))
plt.scatter(xst, yst, color="black", label="Actual Test Data", alpha=0.7)

# Sort values for a clean line plot of predictions
sorted_indices = xst.flatten().argsort()
plt.plot(
    xst[sorted_indices],
    preds_test[sorted_indices],
    color="red",
    label="Model Predictions",
    linewidth=2,
)

plt.xlabel("Input")
plt.ylabel("Output")
plt.title("Actual vs Predicted Outputs")
plt.legend()
plt.grid(True)
plt.show()
```

### Dataset Information
Include screenshot of the generated data
<img width="432" height="682" alt="image" src="https://github.com/user-attachments/assets/11169819-6fb1-4284-b0ff-493673eeb592" />

### OUTPUT

### Training Loss Vs Iteration Plot
<img width="702" height="583" alt="image" src="https://github.com/user-attachments/assets/3de29c99-8a55-478a-9bcb-e0dfc3f497ad" />


### New Sample Data Prediction
Include your sample input and output here

<img width="831" height="412" alt="image" src="https://github.com/user-attachments/assets/9b810bf3-33b4-4069-977b-e6d611114028" />

<img width="1043" height="545" alt="image" src="https://github.com/user-attachments/assets/a2dfb5d9-20bb-4b18-9055-dcfda6316d71" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
