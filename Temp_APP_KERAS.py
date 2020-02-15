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

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.models import load_model
from tensorflow.python.keras.metrics import Metric

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

def data_split_train_test(train_data_df,test_data_df):

# def data_split_train_test(train_data_df,test_data_df,val_data_df):

#     train_len = 29
  
    ##----------------- UNCOMMENT BELOW
    
    if hist_flag==0: ## CURR OPTIONS ONLY
        
    
        train_X = train_data_df[['Safe','BigRisky','SmallRisky']].values
        train_y = train_data_df[['Choice']].values.astype(np.int32)

        test_X = test_data_df[['Safe','BigRisky','SmallRisky']].values
        test_y = test_data_df[['Choice']].values.astype(np.int32)

#         val_X = val_data_df[['Safe','BigRisky','SmallRisky']].values
#         val_y = val_data_df[['Choice']].values.astype(np.int32)

    elif hist_flag==1: ## CURR OPTIONS, PREV ACTIONS:
        
        train_X = train_data_df[['Safe','BigRisky','SmallRisky','PrevChoice']].values
        train_y = train_data_df[['Choice']].values.astype(np.int32)

        test_X = test_data_df[['Safe','BigRisky','SmallRisky','PrevChoice']].values
        test_y = test_data_df[['Choice']].values.astype(np.int32)

#         val_X = val_data_df[['Safe','BigRisky','SmallRisky','PrevChoice']].values
#         val_y = val_data_df[['Choice']].values.astype(np.int32)
        
        
      
    elif hist_flag==2: # CURR OPTIONS, PREV OUTCOME        
        train_X = train_data_df[['Safe','BigRisky','SmallRisky','PrevOutcome']].values
        train_y = train_data_df[['Choice']].values.astype(np.int32)

        test_X = test_data_df[['Safe','BigRisky','SmallRisky','PrevOutcome']].values
        test_y = test_data_df[['Choice']].values.astype(np.int32)

#         val_X = val_data_df[['Safe','BigRisky','SmallRisky','PrevOutcome']].values
#         val_y = val_data_df[['Choice']].values.astype(np.int32)
       
        
        
        
        
        
        
        
    elif hist_flag==3: ## CURR OPTIONS, PREV ACTIONS, PREV OUTCOME
        
        train_X = train_data_df[['Safe','BigRisky','SmallRisky','PrevChoice','PrevOutcome']].values
        train_y = train_data_df[['Choice']].values.astype(np.int32)

        test_X = test_data_df[['Safe','BigRisky','SmallRisky','PrevChoice','PrevOutcome']].values
        test_y = test_data_df[['Choice']].values.astype(np.int32)

#         val_X = val_data_df[['Safe','BigRisky','SmallRisky','PrevChoice','PrevOutcome']].values
#         val_y = val_data_df[['Choice']].values.astype(np.int32)
             
        

####### Prev O + C+ R + CurrO--------------------
    elif hist_flag==4: # CURR OPTIONS, PREV ACTIONS, PREV OUTCOME, PREV OPTIONS
        
        train_X = train_data_df[['Safe','BigRisky','SmallRisky','PrevOutcome','PrevChoice','PrevSafe','PrevBigRisky','PrevSmallRisky']].values
        train_y = train_data_df[['Choice']].values.astype(np.int32)

        test_X = test_data_df[['Safe','BigRisky','SmallRisky','PrevOutcome','PrevChoice','PrevSafe','PrevBigRisky','PrevSmallRisky']].values
        test_y = test_data_df[['Choice']].values.astype(np.int32)

#         val_X = val_data_df[['Safe','BigRisky','SmallRisky','PrevOutcome','PrevChoice','PrevSafe','PrevBigRisky','PrevSmallRisky']].values
#         val_y = val_data_df[['Choice']].values.astype(np.int32)
    
    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.fit_transform(test_X)

    
    train_X = train_X.reshape(-1,timesteps,inputs)
    test_X = test_X.reshape(-1,timesteps,inputs)
    
    ### Add half the episodes from test_X into train_X
    train_X = np.concatenate((train_X, test_X[0:int(test_X.shape[0]/2),:,:]), axis=0)
    test_X = test_X[int(test_X.shape[0]/2):,:,:]
    print(train_X.shape)
    print(test_X.shape)
    
    
    train_y = train_y.reshape(-1,timesteps,outputs)
    test_y = test_y.reshape(-1,timesteps,outputs)
    
    ### Add half the episodes from test_y into train_y
    train_y = np.concatenate((train_y, test_y[0:int(test_y.shape[0]/2),:,:]), axis=0)
    test_y = test_y[int(test_y.shape[0]/2):,:,:]
    print(train_y.shape)
    print(test_y.shape)    
    
    return train_X, train_y, test_X, test_y


def train_RNN_play_by_play(neurons,train_X,train_y,test_X,test_y):

    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=50)
    # es = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1)

    mc =keras.callbacks.ModelCheckpoint(file_path+'/best_model.h5', monitor='val_loss', mode='min', verbose=0, save_best_only=True)
    

    # define problem properties
    n_timesteps = 29
    n_epochs = 5000
    # define LSTM
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(n_timesteps, inputs), return_sequences=True))
    model.add(TimeDistributed(Dense(1, activation='sigmoid'))) ## dense + softmax activation 
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=METRICS)


    x,y = train_X,train_y
    history=  model.fit(x, y, epochs=n_epochs, batch_size=32, verbose=0, validation_split = 0.2,callbacks=[es,mc])


    # load the saved model
    saved_model = load_model(file_path+'/best_model.h5')
    # evaluate the model
    train_loss, train_acc = saved_model.evaluate(train_X, train_y, verbose=1)
    test_loss, test_acc = saved_model.evaluate(test_X, test_y, verbose=1)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    
    pseudo_R2_train = 1 + train_loss/np.log(0.5)
    pseudo_R2_test = 1 + test_loss/np.log(0.5)
    
    metric_out_df= pd.DataFrame(np.array([train_acc,train_loss,pseudo_R2_train, test_acc,test_loss,pseudo_R2_test,neurons,n_epochs]).reshape(-1,8),columns =["accuracy_train","loss_train","pseudoR2_train","accuracy_test","loss_test","pseudoR2_test","neurons","epochs"])
        
    prob_train =  model.predict_proba(train_X)
    prob_test  =  model.predict_proba(test_X)
    
    
    return metric_out_df, prob_train, prob_test





#### MAIN FUNCTION




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
        
    

inputs =4        # number of input features
timesteps = 29         # Timesteps
outputs= 1         # Number of outputs


neurons=8
pretraining=False
hist_flag=2

for num, subj_file_path in enumerate(subj_files_list):

#
# for num, subj_file_path in enumerate([subj_files_list[0]]):
    print(num)
    
    file_path  ="/Users/ritwik7/Dropbox (Personal)/Postdoc_UCL/DATA/rlab_incomplete_rewardSWB_code/by_RN/appdata/"+ subj_file_path
    Keras_file_path = file_path
                
#     file_path = file_path + "/OddEvenPlays"
    file_path = file_path + "/OddEvenPlays/RandomizedPlays1"

    train_data_df= pd.read_csv(file_path+"/train_data.csv")
    test_data_df = pd.read_csv(file_path+"/val_test_data.csv")
#     val_data_df = pd.read_csv(file_path+"/val_data.csv")
    
    Keras_file_path = Keras_file_path + "/Play_by_play"
    print(Keras_file_path)
#     os.mkdir(Keras_file_path)
    
    
    
    train_X, train_y, test_X, test_y= data_split_train_test(train_data_df,test_data_df)
    
    metric_out_df, prob_train, prob_test = train_RNN_play_by_play(neurons,train_X,train_y,test_X,test_y)

    
 
    prob_train_df = pd.DataFrame(prob_train.reshape(prob_train.shape[0],prob_train.shape[1]))
    prob_test_df = pd.DataFrame(prob_test.reshape(prob_test.shape[0],prob_test.shape[1]))

    if hist_flag==0:
        metric_out_df.to_csv(Keras_file_path+"/LSTM_updated_Crossval_currO_metricsneurons="+str(neurons)+".csv")
        prob_train_df.to_csv(Keras_file_path + "/prob_train_currO_neurons="+str(neurons)+".csv")
        prob_test_df.to_csv(Keras_file_path + "/prob_test_currO_neurons="+str(neurons)+".csv")
    
    
    elif hist_flag==1:
        metric_out_df.to_csv(Keras_file_path+"/LSTM_updated_Crossval_currOprevC_metricsneurons="+str(neurons)+".csv")
        prob_train_df.to_csv(Keras_file_path + "/prob_train_currOprevC_neurons="+str(neurons)+".csv")
        prob_test_df.to_csv(Keras_file_path + "/prob_test_currOprevC_neurons="+str(neurons)+".csv")
        
        
    elif hist_flag==2:
        metric_out_df.to_csv(Keras_file_path+"/LSTM_updated_Crossval_currOprevR_metricsneurons="+str(neurons)+".csv")
        prob_train_df.to_csv(Keras_file_path + "/prob_train_currOprevR_neurons="+str(neurons)+".csv")
        prob_test_df.to_csv(Keras_file_path + "/prob_test_currOprevR_neurons="+str(neurons)+".csv")

    elif hist_flag==3:
        metric_out_df.to_csv(Keras_file_path+"/LSTM_updated_Crossval_currOprevRC_metricsneurons="+str(neurons)+".csv")
        prob_train_df.to_csv(Keras_file_path + "/prob_train_currOprevRC_neurons="+str(neurons)+".csv")
        prob_test_df.to_csv(Keras_file_path + "/prob_test_currOprevRC_neurons="+str(neurons)+".csv")
# ################################
    elif hist_flag==4:
        metric_out_df.to_csv(Keras_file_path+"/LSTM_updated_Crossval_currprev_opts_metricsneurons="+str(neurons)+".csv")
        prob_train_df.to_csv(Keras_file_path + "/prob_train_currentprevopts_neurons="+str(neurons)+".csv")
        prob_test_df.to_csv(Keras_file_path + "/prob_test_currentprevopts_neurons="+str(neurons)+".csv")

