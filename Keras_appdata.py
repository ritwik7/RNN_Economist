import tensorflow as tf
import keras
import numpy as np
import os
# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
import seaborn as sns
import scipy.stats as sc_stats
import random

from sklearn.model_selection import train_test_split
onehot_encoder=OneHotEncoder(sparse=False)
from tensorflow.contrib import rnn

os.environ['KMP_DUPLICATE_LIB_OK']='True'


inputs = 3          # MNIST data input (image shape: 28x28)
timesteps = 29         # Timesteps
outputs= 1         # Number of classes, one class per digit
 
 
def add_releveant_features(task_df):


    task_df['PrevOutcome']=task_df['Outcome'].shift(1)
    task_df.loc[1,'PrevOutcome']= 0

    task_df['PrevChoice']=task_df['Choice'].shift(1)
    task_df.loc[1,'PrevChoice']= 0

    task_df['PrevSafe']=task_df['Safe'].shift(1)
    task_df.loc[1,'PrevSafe']= 0

    task_df['PrevBigRisky']=task_df['BigRisky'].shift(1)
    task_df.loc[1,'PrevBigRisky']= 0

    task_df['PrevSmallRisky']=task_df['SmallRisky'].shift(1)
    task_df.loc[1,'PrevSmallRisky']= 0
    
#     task_df['PrevRT']=task_df['RT'].shift(1)
#     task_df.loc[1,'PrevRT']= 0 
    return task_df





dir_list = os.listdir("/Users/ritwik7/Dropbox (Personal)/Postdoc_UCL/DATA/rlab_incomplete_rewardSWB_code/by_RN/appdata/")
dir_path ="/Users/ritwik7/Dropbox (Personal)/Postdoc_UCL/DATA/rlab_incomplete_rewardSWB_code/by_RN/appdata/"

subj_files_list =[]; ## list of subject_files fullfilling a criteria

dir_files = [i for i in os.listdir(dir_path) if i.startswith('sub')]

for subj_file_path in dir_files:

    file_path  ="/Users/ritwik7/Dropbox (Personal)/Postdoc_UCL/DATA/rlab_incomplete_rewardSWB_code/by_RN/appdata/"+ subj_file_path
    mypath =file_path
    
    play_names = [i for i in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,i)) and i.startswith('app')]   
    
    if len(play_names) >= 50: ## criteria
        subj_files_list.append(subj_file_path)
        
    





neurons=8
pretraining=False

for num, subj_file_path in enumerate([subj_files_list[0]]):

    file_path  ="/Users/ritwik7/Dropbox (Personal)/Postdoc_UCL/DATA/rlab_incomplete_rewardSWB_code/by_RN/appdata/"+ subj_file_path
                
#     file_path = file_path + "/OddEvenPlays"
    file_path = file_path + "/OddEvenPlays/RandomizedPlays1"

    train_data_df= pd.read_csv(file_path+"/train_data.csv")
    test_data_df = pd.read_csv(file_path+"/test_data.csv")
    val_data_df = pd.read_csv(file_path+"/val_data.csv")
    
    
    
    train_X = train_data_df[['Safe','BigRisky','SmallRisky']].values
    train_y = train_data_df[['Choice']].values.astype(np.int32).reshape(-1,timesteps,outputs)

    test_X = test_data_df[['Safe','BigRisky','SmallRisky']].values
    test_y = test_data_df[['Choice']].values.astype(np.int32).reshape(-1,13,outputs)

    val_X = val_data_df[['Safe','BigRisky','SmallRisky']].values
    val_y = val_data_df[['Choice']].values.astype(np.int32).reshape(-1,16,outputs)
        
    scaler = MinMaxScaler(feature_range=(0, 1))  
    train_X = scaler.fit_transform(train_X[:,]).reshape(-1,timesteps,inputs)
    test_X = scaler.fit_transform(test_X).reshape(-1,13,inputs)
    val_X = scaler.fit_transform(val_X).reshape(-1,16,inputs)


    ######################
    test_X = np.concatenate((test_X,val_X),axis=1)
    test_y = np.concatenate((test_y,val_y),axis=1)
        
    
    
    
    
#     train_X = train_X[:,None,:]
#     val_X = val_X[:,None,:]
#     test_X = test_X[:,None,:]


    # # one-hot encode the outputs

#     onehot_encoder = OneHotEncoder()
#     encode_categorical = train_y.reshape(len(train_y), 1)
#     encode_categorical_test = test_y.reshape(len(test_y), 1)
#     encode_categorical_val = val_y.reshape(len(val_y),1)


#     train_y = onehot_encoder.fit_transform(encode_categorical).toarray()
#     test_y = onehot_encoder.fit_transform(encode_categorical_test).toarray()
#     val_y = onehot_encoder.fit_transform(encode_categorical_val).toarray()


    
    
#     metric_out_df, prob_train, prob_test, prob_val = train_RNN(neurons,train_X,train_y,test_X,test_y,val_X,val_y)
#     train_RNN(neurons,train_X,train_y,test_X,test_y,val_X,val_y)





from random import random
from numpy import array
from numpy import cumsum
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed

# create a sequence classification instance
# def get_sequence(n_timesteps):
# 	# create a sequence of random numbers in [0,1]
# 	X = array([random() for _ in range(n_timesteps)])
# 	# calculate cut-off value to change class values
# 	limit = n_timesteps/4.0
# 	# determine the class outcome for each item in cumulative sequence
# 	y = array([0 if x < limit else 1 for x in cumsum(X)])
# 	# reshape input and output data to be suitable for LSTMs
# 	X = X.reshape(1, n_timesteps, 1)
# 	y = y.reshape(1, n_timesteps, 1)
# 	return X, y

# def training_data:
#     x = train_X
#     y = train_y
#     return x,y



# define problem properties
n_timesteps = 29
# define LSTM
model = Sequential()
model.add(LSTM(neurons, input_shape=(n_timesteps, inputs), return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train LSTM
for epoch in range(10):
# 	# generate new random sequence
# 	X,y = get_sequence(n_timesteps)
#     x,y = training_data(train_X,train_y)

    x,y = train_X,train_y


# 	# fit model for one epoch on this sequence
#     model.fit(x, y, epochs=1, batch_size=1, verbose=2)
    model.fit(x, y, epochs=1, batch_size=1, verbose=2, validation_split = 0.1)



# # evaluate LSTM
# X,y = get_sequence(n_timesteps)
# yhat = model.predict_classes(X, verbose=0)
# for i in range(n_timesteps):
# 	print('Expected:', y[0, i], 'Predicted', yhat[0, i])

print("done")

# yhat = model.predict_classes(test_X, verbose=0)
# print(yhat)
model.evaluate(test_X,test_y,batch_size=None, verbose=1)


