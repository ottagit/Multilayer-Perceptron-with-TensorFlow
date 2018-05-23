from sklearn.datasets import load_iris
from sklearn.utils import shuffle
import pandas as pd
from sklearn.metrics import roc_auc_score

X,y = load_iris(return_X_y = True)
#Merging input and outputs i
data = pd.concat([pd.DataFrame(X),pd.Series(y)],axis=1)

#Naming columns
data.columns = ['F1','F2','F3','F4','TARGET']

#Shuffling data and resetting index
data = shuffle(data)
data = data.reset_index(drop=True)

#Separating input and output
X = data.iloc[:,:4]
y = data.iloc[:,4]

#Train data
X_train = X[:90]
y_train = y[:90]
#Cross Validation data
X_cv = X[90:120]
y_cv = y[90:120]
#Test data
X_test = X[120:]
y_test = y[120:]

#Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_cv = sc_X.transform(X_cv)
X_test = sc_X.transform(X_test)

#converting y to one hot formats
y_train = pd.get_dummies(y_train).values
y_cv = pd.get_dummies(y_cv).values
y_test = pd.get_dummies(y_test).values

#Importing TensorFlow
import tensorflow as tf

# HyperParameters
lr = 0.001
training_epochs = 50
batch_size = 5

# Network Parameters
n_hidden_1 = 10 # number of neurons in 1st layer 
n_hidden_2 = 5 # number of neurons in 2nd layer 
n_input = 4 # 4 columns
n_classes = 3 #Output classes
steps = len(X_train) # How many training data

#Resetting the graph
tf.reset_default_graph()

#Defining Placeholders
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])
hold_prob1 = tf.placeholder(tf.float32)
hold_prob2 = tf.placeholder(tf.float32)

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],mean = 0.0,stddev=0.2)),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],mean = 0.0,stddev = 0.2)),
    'out': tf.Variable(tf.truncated_normal([n_hidden_2, n_classes],mean = 0.0,stddev = 0.2))
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
    layer2_RELU = tf.nn.tanh(layer_2)
    #Applying Dropout
    layer2_dropout = tf.nn.dropout(layer2_RELU,keep_prob=hold_prob2)
    # Output layer
    out_layer = tf.matmul(layer2_dropout, weights['out']) + biases['out']
    return out_layer

# Building model
logits = multilayer_perceptron(X)

# Defining Loss Function
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y))
#Defining optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
#Defining what to minimize
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

#Graph equations for Accuracy
matches = tf.equal(tf.argmax(logits,1),tf.argmax(Y,1))
acc = tf.reduce_mean(tf.cast(matches,tf.float32))

#Opening our Session
with tf.Session() as sess:
    #Runnig initialization of variables
    sess.run(init)
    # Writing down the for loop for epochs
    for epoch in range(training_epochs):
        #For loop for batches
        for i in range(0,steps,batch_size):
            #Getting training data to be fed into the graphs
            batch_x, batch_y = X_train[i:i+batch_size],y_train[i:i+batch_size]
            # Training batch by batch
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y,
                                                            hold_prob1:0.7,
                                                            hold_prob2:0.8})
        #Feeding CV data to te graphs
        acc_on_cv,loss_on_cv,preds_on_cv = sess.run([acc,loss_op,tf.nn.softmax(logits)],
                                                    feed_dict = 
                                                            {X: X_cv,
                                                            Y: y_cv,
                                                            hold_prob1:1.0,
                                                            hold_prob2:1.0})
        #Calculating AUC on Cross Validation data
        auc_on_cv = roc_auc_score(y_cv,preds_on_cv)
        #Printing CV statistics after each epoch
        print("Accuracy:",acc_on_cv,"Loss:",loss_on_cv,"AUC:",auc_on_cv)
    
    #Feeding test data to the graphs
    acc_on_test,loss_on_test,preds_on_test = sess.run([acc,loss_op,tf.nn.softmax(logits)],
                                                    feed_dict = 
                                                            {X: X_test,
                                                            Y: y_test,
                                                            hold_prob1:1.0,
                                                            hold_prob2:1.0})
    #Calculating AUC on CV data
    auc_on_test = roc_auc_score(y_test,preds_on_test)
    print("Test Results:")
    print("Accuracy:",acc_on_test,"Loss:",loss_on_test,"AUC:",auc_on_test)
    
    print("All done")


