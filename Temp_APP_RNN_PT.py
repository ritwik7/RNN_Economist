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

os.environ['KMP_DUPLICATE_LIB_OK']='True'
#######################################




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
    
    task_df['PrevRT']=task_df['RT'].shift(1)
    task_df.loc[1,'PrevRT']= 0
    
#     task_df['RPE'] =  task_df['Outcome'].shift(1) - 0.5*(task_df['BigRisky'].shift(1) + 
    
    return task_df



def add_kback_features(task_df):

    for k in range(1,11):
        task_df[str(k)+'backOutcome']=task_df['Outcome'].shift(k)
        task_df[str(k)+'backChoice']=task_df['Choice'].shift(k)
        task_df[str(k)+'backSafe']=task_df['Safe'].shift(k)
        task_df[str(k)+'backBigRisky']=task_df['BigRisky'].shift(k)
        task_df[str(k)+'backSmallRisky']=task_df['SmallRisky'].shift(k)

    return task_df

def chunk_split_data(data,start_chunk,end_chunk):
    
    a=[k for k in range(start_chunk,end_chunk)]
    out=[]

    for d in range(0,data.shape[0],20):

        c= [c+d for c in a]
        out = out+c

    while out[-1]>=data.shape[0]-1:
        out.pop()
#     return out
    return data[out]




    
def train_RNN(neurons,train_X,train_y,test_X,test_y,val_X,val_y): 
    reset_graph()

    learning_rate = 0.001
    epochs = 50000
    batch_size = int(train_X.shape[0]/2)
    # batch_size = 100
    length = train_X.shape[0]
    display = 100
    neurons = neurons

    num_batches = 100
    seq_len = 10

    percent_above_PT = 1

    train_threshold = 1.5#PT_R2 + percent_above_PT


    save_step = 100


    best_loss_val = np.infty
    checks_since_last_progress = 0
    max_checks_without_progress = 1000


    # clear graph (if any) before running
    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, time_steps, inputs])

    y = tf.placeholder(tf.float32, [None, outputs])

    # LSTM Cell
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=neurons, activation=tf.nn.relu)
    cell_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    # pass into Dense layer
    stacked_outputs = tf.reshape(cell_outputs, [-1, neurons])
    out = tf.layers.dense(inputs=stacked_outputs, units=outputs)

    probability = tf.nn.softmax(out)

    # squared error loss or cost function for linear regression
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=out))

    # optimizer to minimize cost
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    saver = tf.train.Saver()
    
    accuracy = tf.metrics.accuracy(labels =  tf.argmax(y, 1),
                          predictions = tf.argmax(out, 1),
                          name = "accuracy")
    precision = tf.metrics.precision(labels=tf.argmax(y, 1),
                                 predictions=tf.argmax(out, 1),
                                 name="precision")
    recall = tf.metrics.recall(labels=tf.argmax(y, 1),
                           predictions=tf.argmax(out, 1),
                           name="recall")
    f1 = 2 * accuracy[1] * recall[1] / ( precision[1] + recall[1] )

    acc_up,acc_val = accuracy
    auc = tf.metrics.auc(labels=tf.argmax(y, 1),
                           predictions=tf.argmax(out, 1),
                           name="auc")
    
    valid_store = []
    
    with tf.Session() as sess:
        #######################
#         saver.restore(sess, "./checkpts/Original_RNN_LSTM_8features_v2.ckpt")
#         saver.restore(sess, "./checkpts/OriginalDATA_RNN_LSTM_8features.ckpt")
        
        if pretraining == True:

            saver.restore(sess, "./checkpts/Original_v2_DATA_RNN_LSTM_8features.ckpt")

        #######################
        
        # initialize all variables
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        # Train the model
        for steps in range(epochs):
            mini_batch = zip(range(0, length, batch_size),
                       range(batch_size, length+1, batch_size))

            # train data in mini-batches
            for (start, end) in mini_batch:
    #             print(start,end)
                sess.run(training_op, feed_dict = {X: train_X[start:end,:,:],
                                                   y: train_y[start:end,:]}) 

            ## train data in batches of length subsequence

    #         for k in range(num_batches):
    #             X_seq, y_seq = random_subsequence(train_X,train_y,seq_len)

    #             sess.run(training_op, feed_dict = {X:X_seq,y:y_seq}) 
            loss_fn = loss.eval(feed_dict = {X: train_X, y: train_y})
            loss_val = loss.eval(feed_dict={X: val_X, y: val_y})


            # print training performance 
            if (steps+1) % display == 0:
                # evaluate loss function on training set


                loss_fn = loss.eval(feed_dict = {X: train_X, y: train_y})
                print('Step: {}  \tTraining loss: {}'.format((steps+1), loss_fn))

                acc_train = acc_val.eval(feed_dict={X: train_X, y: train_y})
                print('Step: {}  \tTraining accuracy: {}'.format((steps+1), acc_train))


                acc_test = acc_val.eval(feed_dict={X: test_X, y: test_y})
    #             print('Step: {}  \tTest accuracy: {}'.format((steps+1), acc_test))

                loss_test = loss.eval(feed_dict={X: test_X, y: test_y})
    #             print('Step: {}  \tTest loss: {}'.format((steps+1), loss_test))

                accu_val = acc_val.eval(feed_dict={X: val_X, y: val_y})

                loss_val = loss.eval(feed_dict={X: val_X, y: val_y})
                print('Step: {}  \tValid loss: {}'.format((steps+1), loss_val))

                valid_store.append(loss_val)







            if (1 + loss_fn/np.log(0.5)) > train_threshold:
                    print("Threshold achieved, quit training")
                    break


            if loss_val < best_loss_val:

                        best_loss_val = loss_val
                        checks_since_last_progress = 0
            else:
                            checks_since_last_progress += 1


            # EARLY STOPPING
            if checks_since_last_progress > max_checks_without_progress:
                print("Early stopping!")
                break


            if (steps+1) % save_step ==0:
                                save_path = saver.save(sess, "./checkpts/Later_RNN_LSTM_8features.ckpt")

#                 save_path = saver.save(sess, "./checkpts/RNN_Internet_LSTM_model_5features.ckpt")





    #     evaluate model accuracy
        acc, prec, recall, f1, AUC = sess.run([accuracy, precision, recall, f1,auc],
                                         feed_dict = {X: train_X, y: train_y})
        prob_train = probability.eval(feed_dict = {X: train_X, y: train_y})
        prob_test = probability.eval(feed_dict = {X: test_X, y: test_y})
        prob_valid = probability.eval(feed_dict = {X: val_X, y: val_y})



        print('\nEvaluation  on training set')
        print('Accuracy:', acc[1])
        print('Precision:', prec[1])
        print('Recall:', recall[1])
        print('F1 score:', f1)
        print('AUC:', AUC[1])
        
      
    
    
#         save_path = saver.save(sess, "./checkpts/Original_v2_DATA_RNN_LSTM_8features.ckpt")
#         save_path = saver.save(sess, "./checkpts/Later_v2_DATA_RNN_LSTM_8features.ckpt")

        
#         save_path = saver.save(sess, "./checkpts/OriginalDATA_RNN_LSTM_8features.ckpt")
#         save_path = saver.save(sess, "./checkpts/LaterDATA_RNN_LSTM_8features.ckpt")


#         save_path = saver.save(sess, "./checkpts/Original_RNN_LSTM_8features.ckpt")

#         save_path = saver.save(sess, "./checkpts/Later_RNN_LSTM_8features.ckpt")




    ## APP DATA
#         save_path = saver.save(sess, "./checkpts/Original_v2_APPDATA_RNN_LSTM_8features.ckpt")
        save_path = saver.save(sess, "./checkpts/Later_v2_APPDATA_RNN_LSTM_8features.ckpt")


    metric_out_df= pd.DataFrame(np.array([acc[1],prec[1],recall[1],f1,AUC[1],loss_fn,accu_val,best_loss_val,acc_test,loss_test,neurons,learning_rate,epochs,steps]).reshape(-1,14),columns =["accuracy","precision","recall","f1_score","auc","loss","accuracy_val","loss_val","accuracy_test","loss_test","neurons","learning_rate","n_epochs","steps"])
    return metric_out_df, prob_train, prob_test, prob_valid
    


    
    
def random_subsequence(X,y,seq_len):
    rnd  = random.randint(0,len(X)-seq_len)
    X_seq, y_seq = X[rnd:rnd+seq_len,:], y[rnd:rnd+seq_len,:]
    return X_seq, y_seq

    print(y_seq.shape)



### WHEN RANDOMIZING FILES

def randomize_trials(df):

    locs= random.sample([a for a in range(0,df.shape[0])],df.shape[0])
    # len(locs)
    df = add_releveant_features(df.loc[locs])
    
    ### get rid of first index since it contains NaN for previous trials
    df  = df.iloc[1:]
    
    return df , pd.DataFrame(locs)

def generate_randomizations(train_data_df,test_data_df,val_data_df):
        train_data_df,train_locs = randomize_trials(train_data_df)
        test_data_df,test_locs = randomize_trials(test_data_df)
        val_data_df,val_locs = randomize_trials(val_data_df)
        
        #         os.mkdir(file_path)
        train_locs.to_csv(file_path+"/train_locs.csv")
        test_locs.to_csv(file_path+"/test_locs.csv")
        val_locs.to_csv(file_path+"/val_locs.csv")
        
        return train_locs, test_locs, val_locs   

def get_shuffled_data():
        train_locs = pd.read_csv(file_path+"/train_locs.csv")
        test_locs = pd.read_csv(file_path+"/test_locs.csv")
        val_locs = pd.read_csv(file_path+"/val_locs.csv")
        train_data_random_df  = train_data_df.iloc[train_locs.iloc[:,1]]
        test_data_random_df  = test_data_df.iloc[test_locs.iloc[:,1]]
        val_data_random_df  = val_data_df.iloc[val_locs.iloc[:,1]]
        
        train_data_random_df = add_releveant_features(train_data_random_df)
        train_data_random_df = add_kback_features(train_data_random_df)
        test_data_random_df = add_releveant_features(test_data_random_df)
        test_data_random_df = add_kback_features(test_data_random_df)
        val_data_random_df = add_releveant_features(val_data_random_df)
        val_data_random_df = add_kback_features(val_data_random_df)
        
        return train_data_random_df, test_data_random_df,val_data_random_df, train_locs, test_locs, val_locs 

def restore_input_feature(randomize_trials_flag,hist_flag):

    if randomize_trials_flag==True:
        if hist_flag==0: ## CURR OPTIONS ONLY
            print("return as is")
            
    
        if hist_flag==1: ## CURR OPTIONS, PREV ACTIONS:
            train_data_random_df.loc[train_locs.iloc[:,1],'PrevChoice'] = train_data_df.loc[train_locs.iloc[:,1],'PrevChoice']
            test_data_random_df.loc[test_locs.iloc[:,1],'PrevChoice'] = test_data_df.loc[test_locs.iloc[:,1],'PrevChoice']
            val_data_random_df.loc[val_locs.iloc[:,1],'PrevChoice'] = val_data_df.loc[val_locs.iloc[:,1],'PrevChoice']



        elif hist_flag==2: # CURR OPTIONS, PREV OUTCOME        

            train_data_random_df.loc[train_locs.iloc[:,1],'PrevOutcome'] = train_data_df.loc[train_locs.iloc[:,1],'PrevOutcome']
            test_data_random_df.loc[test_locs.iloc[:,1],'PrevOutcome'] = test_data_df.loc[test_locs.iloc[:,1],'PrevOutcome']
            val_data_random_df.loc[val_locs.iloc[:,1],'PrevOutcome'] = val_data_df.loc[val_locs.iloc[:,1],'PrevOutcome']


        elif hist_flag==3: ## CURR OPTIONS, PREV ACTIONS, PREV OUTCOME

            train_data_random_df.loc[train_locs.iloc[:,1],'PrevChoice','PrevOutcome'] = train_data_df.loc[train_locs.iloc[:,1],'PrevChoice','PrevOutcome']
            test_data_random_df.loc[test_locs.iloc[:,1],'PrevChoice','PrevOutcome'] = test_data_df.loc[test_locs.iloc[:,1],'PrevChoice','PrevOutcome']
            val_data_random_df.loc[val_locs.iloc[:,1],'PrevChoice','PrevOutcome'] = val_data_df.loc[val_locs.iloc[:,1],'PrevChoice','PrevOutcome']



    ####### Prev O + C+ R + CurrO--------------------
        elif hist_flag==4: # CURR OPTIONS, PREV ACTIONS, PREV OUTCOME, PREV OPTIONS
            print(hist_flag)
   
    return train_data_random_df, test_data_random_df,val_data_random_df 


def data_split_odd_even(train_data_df,test_data_df,val_data_df):

#     train_len = 29
#     test_len = 14
#     val_len = 15

    ##----------------- UNCOMMENT BELOW
    
    if hist_flag==0: ## CURR OPTIONS ONLY
        
    
        train_X = train_data_df[['Safe','BigRisky','SmallRisky']].values
        train_y = train_data_df[['Choice']].values.astype(np.int32)

        test_X = test_data_df[['Safe','BigRisky','SmallRisky']].values
        test_y = test_data_df[['Choice']].values.astype(np.int32)

        val_X = val_data_df[['Safe','BigRisky','SmallRisky']].values
        val_y = val_data_df[['Choice']].values.astype(np.int32)

    elif hist_flag==1: ## CURR OPTIONS, PREV ACTIONS:
        
        train_X = train_data_df[['Safe','BigRisky','SmallRisky','PrevChoice']].values
        train_y = train_data_df[['Choice']].values.astype(np.int32)

        test_X = test_data_df[['Safe','BigRisky','SmallRisky','PrevChoice']].values
        test_y = test_data_df[['Choice']].values.astype(np.int32)

        val_X = val_data_df[['Safe','BigRisky','SmallRisky','PrevChoice']].values
        val_y = val_data_df[['Choice']].values.astype(np.int32)
        
        
      
    elif hist_flag==2: # CURR OPTIONS, PREV OUTCOME        
        train_X = train_data_df[['Safe','BigRisky','SmallRisky','PrevOutcome']].values
        train_y = train_data_df[['Choice']].values.astype(np.int32)

        test_X = test_data_df[['Safe','BigRisky','SmallRisky','PrevOutcome']].values
        test_y = test_data_df[['Choice']].values.astype(np.int32)

        val_X = val_data_df[['Safe','BigRisky','SmallRisky','PrevOutcome']].values
        val_y = val_data_df[['Choice']].values.astype(np.int32)
       
        
        
        
        
        
        
        
    elif hist_flag==3: ## CURR OPTIONS, PREV ACTIONS, PREV OUTCOME
        
        train_X = train_data_df[['Safe','BigRisky','SmallRisky','PrevChoice','PrevOutcome']].values
        train_y = train_data_df[['Choice']].values.astype(np.int32)

        test_X = test_data_df[['Safe','BigRisky','SmallRisky','PrevChoice','PrevOutcome']].values
        test_y = test_data_df[['Choice']].values.astype(np.int32)

        val_X = val_data_df[['Safe','BigRisky','SmallRisky','PrevChoice','PrevOutcome']].values
        val_y = val_data_df[['Choice']].values.astype(np.int32)
             
        

####### Prev O + C+ R + CurrO--------------------
    elif hist_flag==4: # CURR OPTIONS, PREV ACTIONS, PREV OUTCOME, PREV OPTIONS
        
        train_X = train_data_df[['Safe','BigRisky','SmallRisky','PrevOutcome','PrevChoice','PrevSafe','PrevBigRisky','PrevSmallRisky']].values
        train_y = train_data_df[['Choice']].values.astype(np.int32)

        test_X = test_data_df[['Safe','BigRisky','SmallRisky','PrevOutcome','PrevChoice','PrevSafe','PrevBigRisky','PrevSmallRisky']].values
        test_y = test_data_df[['Choice']].values.astype(np.int32)

        val_X = val_data_df[['Safe','BigRisky','SmallRisky','PrevOutcome','PrevChoice','PrevSafe','PrevBigRisky','PrevSmallRisky']].values
        val_y = val_data_df[['Choice']].values.astype(np.int32)
    
    
    
    
    

    
    ######## sampling 
    
    
#### - Prev RT+C+R+O + Curr O----------------------

#     train_X = task_df.loc[task_df.TrialNum>1, ['Safe','BigRisky','SmallRisky','PrevOutcome','PrevChoice','PrevSafe','PrevBigRisky','PrevSmallRisky','PrevRT']].values
#     train_y = task_df.loc[task_df.TrialNum>1,['Choice']].values.astype(np.int32)

#     test_X = dopa_task_df.loc[dopa_task_df.TrialNum>1,['Safe','BigRisky','SmallRisky','PrevOutcome','PrevChoice','PrevSafe','PrevBigRisky','PrevSmallRisky','PrevRT']].values
#     test_y = dopa_task_df.loc[dopa_task_df.TrialNum>1,['Choice']].values.astype(np.int32)






    #### PRE TRAINING
#     stop = int(0.7*len(train_X))
#     print(stop)
#     train_X, test_X, val_X, train_y, test_y, val_y= train_X[:stop], train_X[stop:stop+int((len(train_X)-stop)/2)], train_X[stop+int((len(train_X)-stop)/2):],train_y[:stop], train_y[stop:stop+int((len(train_X)-stop)/2)], train_y[stop+int((len(train_X)-stop)/2):]
    
#     train_X, test_X, val_X, train_y, test_y, val_y = train_X, test_X, test_X, train_y, test_y, test_y
    ###################################################################


    print(train_X.shape)
    print(train_y.shape)
    print(val_X.shape)
    print(val_y.shape)
    print(test_X.shape)
    print(test_y.shape)

    # # center and scale
    scaler = MinMaxScaler(feature_range=(0, 1))    
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.fit_transform(test_X)
    val_X = scaler.fit_transform(val_X)


    train_X = train_X[:,None,:]
    val_X = val_X[:,None,:]
    test_X = test_X[:,None,:]


    # # one-hot encode the outputs

    onehot_encoder = OneHotEncoder()
    encode_categorical = train_y.reshape(len(train_y), 1)
    encode_categorical_test = test_y.reshape(len(test_y), 1)
    encode_categorical_val = val_y.reshape(len(val_y),1)


    train_y = onehot_encoder.fit_transform(encode_categorical).toarray()
    test_y = onehot_encoder.fit_transform(encode_categorical_test).toarray()
    val_y = onehot_encoder.fit_transform(encode_categorical_val).toarray()

    
    return train_X, train_y, test_X, test_y, val_X,val_y





####### MAIN FUNCTION ##########


# parameters
time_steps = 1
inputs = 4
outputs = 2



###### MAIN FUNCTION

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





neurons = 8
hist_flag=2
randomize_trials_flag=True


for num, subj_file_path in enumerate(subj_files_list):

# for subj_file_path in [subj_files_list[0]]:

    file_path  ="/Users/ritwik7/Dropbox (Personal)/Postdoc_UCL/DATA/rlab_incomplete_rewardSWB_code/by_RN/appdata/"+ subj_file_path
                
#     file_path = file_path + "/OddEvenPlays"
    file_path = file_path + "/OddEvenPlays/RandomizedPlays1"



    train_data_df= pd.read_csv(file_path+"/train_data.csv")
    test_data_df = pd.read_csv(file_path+"/test_data.csv")
    val_data_df = pd.read_csv(file_path+"/val_data.csv")
    
    
    ####### IMPORTANT ALTERATION
    train_data_df = add_kback_features(train_data_df)
    val_data_df = add_kback_features(val_data_df)
    test_data_df = pd.read_csv(file_path+"/test_data_new.csv") ## otherwise testdata df would have NaNs for 1 back

    if randomize_trials_flag==True:
        file_path = file_path + "/RandomizeTrials"

        #### FOR GENERATING RANDOMIZATIONS
#         train_locs, test_locs, val_locs = generate_randomizations(train_data_df,test_data_df,val_data_df)

        train_data_random_df, test_data_random_df,val_data_random_df, train_locs, test_locs, val_locs  = get_shuffled_data()

        train_data_random_df, test_data_random_df,val_data_random_df = restore_input_feature(randomize_trials_flag,hist_flag)
        
        train_data_df , test_data_df, val_data_df = train_data_random_df, test_data_random_df, val_data_random_df
        
        
        
        
    
    print(file_path)
    
    
    train_X, train_y, test_X, test_y,val_X,val_y = data_split_odd_even(train_data_df,test_data_df,val_data_df)

    pretraining = False; 
    metric_out_df, prob_train, prob_test, prob_val = train_RNN(neurons,train_X,train_y,test_X,test_y,val_X,val_y)
    
    print(metric_out_df)
    
   
 
    prob_train_df = pd.DataFrame(prob_train.reshape(-1,2),columns = {'action_0','action_1'})
    prob_test_df = pd.DataFrame(prob_test.reshape(-1,2),columns = {'action_0','action_1'})
    prob_val_df = pd.DataFrame(prob_val.reshape(-1,2),columns = {'action_0','action_1'})

    if hist_flag==0:
        metric_out_df.to_csv(file_path+"/LSTM_updated_Crossval_currO_metricsneurons="+str(neurons)+".csv")
        prob_train_df.to_csv(file_path + "/prob_train_currO_neurons="+str(neurons)+".csv")
        prob_test_df.to_csv(file_path + "/prob_test_currO_neurons="+str(neurons)+".csv")
        prob_val_df.to_csv(file_path + "/prob_val_currO_neurons="+str(neurons)+".csv")
    
    
    elif hist_flag==1:
        metric_out_df.to_csv(file_path+"/LSTM_updated_Crossval_currOprevC_metricsneurons="+str(neurons)+".csv")
        prob_train_df.to_csv(file_path + "/prob_train_currOprevC_neurons="+str(neurons)+".csv")
        prob_test_df.to_csv(file_path + "/prob_test_currOprevC_neurons="+str(neurons)+".csv")
        prob_val_df.to_csv(file_path + "/prob_val_currOprevC_neurons="+str(neurons)+".csv")
        
        
    elif hist_flag==2:
        metric_out_df.to_csv(file_path+"/LSTM_updated_Crossval_currOprevR_metricsneurons="+str(neurons)+".csv")
        prob_train_df.to_csv(file_path + "/prob_train_currOprevR_neurons="+str(neurons)+".csv")
        prob_test_df.to_csv(file_path + "/prob_test_currOprevR_neurons="+str(neurons)+".csv")
        prob_val_df.to_csv(file_path + "/prob_val_currOprevR_neurons="+str(neurons)+".csv")

    elif hist_flag==3:
        metric_out_df.to_csv(file_path+"/LSTM_updated_Crossval_currOprevRC_metricsneurons="+str(neurons)+".csv")
        prob_train_df.to_csv(file_path + "/prob_train_currOprevRC_neurons="+str(neurons)+".csv")
        prob_test_df.to_csv(file_path + "/prob_test_currOprevRC_neurons="+str(neurons)+".csv")
        prob_val_df.to_csv(file_path + "/prob_val_currOprevRC_neurons="+str(neurons)+".csv")
# ################################
    elif hist_flag==4:
        metric_out_df.to_csv(file_path+"/LSTM_updated_Crossval_currprev_opts_metricsneurons="+str(neurons)+".csv")
        prob_train_df.to_csv(file_path + "/prob_train_currentprevopts_neurons="+str(neurons)+".csv")
        prob_test_df.to_csv(file_path + "/prob_test_currentprevopts_neurons="+str(neurons)+".csv")
        prob_val_df.to_csv(file_path + "/prob_val_currentprevopts_neurons="+str(neurons)+".csv")
# #############################
        








