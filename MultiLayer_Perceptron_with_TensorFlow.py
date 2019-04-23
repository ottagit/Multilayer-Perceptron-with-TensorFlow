#Importing dependencies
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss,accuracy_score
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import random
import os
from tqdm import tqdm,tqdm_notebook,_tqdm_notebook
tqdm.pandas(tqdm_notebook)
#Suppressing warnings and infos of TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
SEED = 51 #Fixing seed
synthetic = False# whether to use synthetic data

def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

seed_everything(seed=SEED)

X,y = load_iris(return_X_y = True)
#Merging input and outputs
data = pd.concat([pd.DataFrame(X),pd.Series(y)],axis=1)

#Naming columns
data.columns = ['F1','F2','F3','F4','TARGET']

#Shuffling data and resetting index
data = shuffle(data)
data = data.reset_index(drop=True)
#Separating input and output
X = data.iloc[:,:4]
y = data.iloc[:,4]

if synthetic:
    X = np.random.randn(1000000,4)
    y = np.random.randn(1000000,1)

    y = np.where(y[:,0]<0.5,0,1)
    X,y = shuffle(X,y)

print(X.shape[0],y.shape[0])
#Train data
train_X = X[:int(len(X)*0.8)]
train_y = y[:int(len(y)*0.8)]
#Test data
test_X = X[int(len(X)*0.8):]
test_y = y[int(len(y)*0.8):]

#Scaling
sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.transform(test_X)

#converting y to one hot formats
train_y = pd.get_dummies(train_y).values
test_y = pd.get_dummies(test_y).values

#5 fold validation
folds = KFold(n_splits= 5, shuffle=True, random_state=SEED)

# HyperParameters
lr = 0.0003
training_epochs = 50
batch_size = 4
pred_size = int(batch_size * 2)

# Network Parameters
n_hidden_1 = 32 #number of neurons in 1st layer 
n_hidden_2 = 16 #number of neurons in 2nd layer 
n_input = 4 #4 columns
n_classes = 3 #Output classes

if synthetic:
    n_hidden_1 = 2048
    n_hidden_2 = 1024
    n_classes = 2
    batch_size = 4096
    pred_size = 8192
steps = len(train_X) # How many training data

#Resetting the graph
tf.reset_default_graph()

#Defining Placeholders
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])
hold_prob1 = tf.placeholder(tf.float32)
hold_prob2 = tf.placeholder(tf.float32)

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],\
        mean = 0.0,stddev=0.2)),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],\
        mean = 0.0,stddev = 0.2)),
    'out': tf.Variable(tf.truncated_normal([n_hidden_2, n_classes],\
        mean = 0.0,stddev = 0.2))
}
biases = {
    'b1': tf.Variable(tf.constant(0.1,shape = [n_hidden_1])),
    'b2': tf.Variable(tf.constant(0.1,shape = [n_hidden_2])),
    'out': tf.Variable(tf.constant(0.1,shape = [n_classes]))
}



# Create model
def multilayer_perceptron(x):
    # First Hidden Layer
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    #Applying RELU nonlinearity
    layer1_RELU = tf.nn.relu(layer_1)
    #Applying Dropout
    layer1_dropout = tf.nn.dropout(layer1_RELU,keep_prob=hold_prob1)
    # Second hidden layer
    layer_2 = tf.add(tf.matmul(layer1_dropout, weights['h2']), biases['b2'])
    #Applying TANH nonlinearity
    layer2_TANH = tf.nn.tanh(layer_2)
    #Applying Dropout
    layer2_dropout = tf.nn.dropout(layer2_TANH,keep_prob=hold_prob2)
    # Output layer
    out_layer = tf.matmul(layer2_dropout, weights['out']) + biases['out']
    return out_layer

# Building model
logits = multilayer_perceptron(X)

# Defining Loss Function
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y))
#Defining optimizer
optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
#Defining what to minimize
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

#Graph equations for Accuracy
matches = tf.equal(tf.argmax(logits,1),tf.argmax(Y,1))
acc = tf.reduce_mean(tf.cast(matches,tf.float32))

remaining_train = len(train_X) % batch_size
remaining_test= len(test_X) % pred_size

# Create arrays and dataframes to store results
oof_preds = np.zeros(( train_X.shape[0] , n_classes ) )
sub_preds = np.zeros(( test_X.shape[0] , n_classes ) )

#Opening our Session
with tf.Session() as sess:
    
    #k-fold validation
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_X,train_y)):
        print( "Fold {} has started".format(n_fold) )
        X_train, y_train = train_X[train_idx], train_y[train_idx]
        X_cv, y_cv = train_X[valid_idx], train_y[valid_idx]
        #Running initialization of variables
        sess.run(init)
        # Writing down the for loop for epochs
        for epoch in tqdm(range(training_epochs)):
            #For loop for batches
            for i in range(0,steps - remaining_train , batch_size):
                #Getting training data to be fed into the graphs
                batch_x, batch_y = X_train[i:i+batch_size],y_train[i:i+batch_size]
                # Training batch by batch
                _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,Y: batch_y,\
                    hold_prob1:0.7,hold_prob2:0.8})
            
            #Feeding last batch of train data
            batch_x, batch_y = X_train[-remaining_train:],y_train[-remaining_train:]
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,Y: batch_y,\
                hold_prob1:0.7,hold_prob2:0.8})

        #Calculating predictions over validation data
        remaining_cv = len(X_cv) % pred_size
        preds_on_cv = list()

        for j in range(0,len(X_cv) - remaining_cv,pred_size):
            batch_x = X_cv[j:j+pred_size]
            preds_on_cv_temp = tf.nn.softmax(logits).eval(
                feed_dict ={X: batch_x,hold_prob1:1.0, hold_prob2:1.0})
            preds_on_cv_temp = list(preds_on_cv_temp)
            preds_on_cv.extend(preds_on_cv_temp)

        #Calculating predictions over last batch of CV data
        if remaining_cv != 0:
            batch_x = X_cv[-remaining_cv:]
            preds_on_cv_temp = tf.nn.softmax(logits).eval(
                    feed_dict ={X: batch_x,hold_prob1:1.0, hold_prob2:1.0})
            preds_on_cv_temp = list(preds_on_cv_temp)
            preds_on_cv.extend(preds_on_cv_temp)

        #Calculating loss and accuracy for CV
        acc_on_cv,loss_on_cv = sess.run([acc,loss_op],feed_dict ={X: X_cv, Y: y_cv,\
            hold_prob1:1.0, hold_prob2:1.0})
        #Calculating AUC on Cross Validation data
        auc_on_cv = roc_auc_score(y_cv,preds_on_cv)
        #Printing CV statistics after each epoch
        print("Accuracy:",acc_on_cv,"Loss:",loss_on_cv,"AUC:",auc_on_cv)
        
        #Calculating predictions over test data
        preds_on_test = list()
        for j in range(0,len(test_X) - remaining_test,pred_size):
            batch_x = test_X[j:j+pred_size]
            preds_on_test_temp = tf.nn.softmax(logits).eval(
                feed_dict ={X: batch_x,hold_prob1:1.0, hold_prob2:1.0})
            preds_on_test_temp = list(preds_on_test_temp)
            preds_on_test.extend(preds_on_test_temp)
        #Calculating predictions over last batch of test data
        if remaining_test != 0:
            batch_x = test_X[-remaining_test:]
            preds_on_test_temp = sess.run(tf.nn.softmax(logits),\
                    feed_dict ={X: batch_x,hold_prob1:1.0, hold_prob2:1.0})
            preds_on_test_temp = list(preds_on_test_temp)
            preds_on_test.extend(preds_on_test_temp)
    oof_preds[valid_idx] = np.array(preds_on_cv)
    sub_preds += np.array(preds_on_test) / folds.n_splits
    print("Validation Results:")
    print("Accuracy:",acc_on_cv,"Loss:",loss_on_cv,"AUC:",auc_on_cv)
print("Training and Scoring on Validation data done")


acc_on_test = accuracy_score(test_y.argmax(axis=1),sub_preds.argmax(axis=1))
loss_on_test = log_loss(test_y,sub_preds)
auc_on_test = roc_auc_score(test_y,sub_preds)
print("Test Results are below:")
print("Accuracy:",acc_on_test,"Loss:",loss_on_test,"AUC:",auc_on_test)
