import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader as data 
from keras.models import load_model
import streamlit as st
import sklearn
import datetime

st.title('Stock Value  Prediction')

st.sidebar.header('User Input Parameters')

### Displaying the dataframe
stock_name = st.text_input("Enter the Stock name here", "TATASTEEL.NS")
st.write(f'Stock to be Predicted : {stock_name}')
stock = yf.Ticker(stock_name)

tp = st.sidebar.selectbox(
    'Time Period',
    ('1d', '1mo', '6mo','1y','3y','5y'))
# st.write('You selected:', tp)
#start_date = st.sidebar.date_input("Start date", datetime.date(2019, 1, 1))
#end_date = st.sidebar.date_input("End date", datetime.date(2021, 1, 31))


df = stock.history(period=tp)
df.reset_index(inplace=True)
df['_Date_']= pd.to_datetime(df["Date"]).dt.strftime('%d-%m-%Y').astype(str)
a = df[['_Date_','Close']].set_index('_Date_')
st.write(a)
#area graph
b = df.set_index('Date')['Close']
st.area_chart(b)


from sklearn.preprocessing import MinMaxScaler

df1 = stock.history(period='5y')
df1=df1.reset_index()['Close']
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
training_size=int(len(df1)*0.70)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


#scaler=MinMaxScaler(feature_range=(0,1))
#df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

#training_size=int(len(df1)*0.70)
#test_size=len(df1)-training_size
#train_data = pd.DataFrame(df1[0:training_size,:]).values
#test_data =pd.DataFrame(df1[training_size:len(df1),:1]).values

import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)

    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

model=load_model('ts_7030.h5')

#testing part
### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
st.subheader('prediction vs original')
fig2=plt.figure(figsize=(12,6))
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
st.pyplot(fig2)

s1 = test_data.shape[0]-time_step
x_input=test_data[s1:].reshape(1,-1)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
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
st.subheader('predicted values for the next 5 days')

predicted_values= scaler.inverse_transform(lst_output[:5])
st.write(predicted_values)

#df3=df1.tolist()
#df3.extend(lst_output)
#df3=scaler.inverse_transform(df3).tolist()

#fig3=plt.figure(figsize=(20,6))
#plt.plot(df3)
#st.pyplot(fig3)

