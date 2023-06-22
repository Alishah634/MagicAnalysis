# Imports:
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # For scaling the data
from imblearn.over_sampling import RandomOverSampler # For oversampling the data
from sklearn.neighbors import KNeighborsClassifier # For KNN model
from sklearn.metrics import classification_report # For accuracy score
from sklearn.naive_bayes import GaussianNB # For Naive Bayes model
from sklearn.linear_model import LogisticRegression # For Logistic Regression model
from sklearn.svm import SVC # For Support Vector Machine model
import tensorflow as tf # For Neural Network model


'''
Dataset Information (from UCI Machine Learning Repository):
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. 
Irvine, CA: University of California, School of Information and Computer Science.
Donated by: P. Savicky Institute of Computer Science, AS of CR Czech Republic savicky '@' cs.cas.cz
'''

'''
The goal of this analysis is to predict whether the particle is a gamma or hadron based on the features.
Deciding which model to use based on the accuracy score of the model.
'''

# Read the dataset from the csv file as pandas dataframe:
cols = ["fLenght", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv('magic04.data', names=cols)
# Print the first 5 rows of the df(For testing purposes, WILL BE REMOVED):
print(df.head())   

# Convert hadron(h) and gamma(g) to 0 and 1 respectively for ease of use:
print(df['class'].unique()) # Print the unique values in the class column 
df['class'] = (df['class'] == 'g').astype(int) # Convert the class column to 0 and 1
print(df['class'].unique()) # Print the unique values in the class column
print(df.head()) # Print the first 5 rows of the df(For testing purposes, WILL BE REMOVED)
 
''' Q. Predict whther the particle is a gamma or hadron based on the features i.e CLASSIFICATION:'''
# Theory
# Since we know for a specific entry whether it is a gamma or hadron,
# we can use supervised learning to predict the class of a new entry; known as classification and supervised learning because,
# we are using a known class column to predict the class of a new entry based on the features or the known class column.
'''-------------------------------------------------------Supervised Learning-------------------------------------------------------:'''

# Plot the histogram for the gamma class
for label in cols[:-1]: # Iterate over the columns in the dataframe(exept the last column i.e class)
    plt.hist(df[df["class"] == 1][label],color= 'blue', label = 'Gamma', alpha = 0.7, density= True) 
    plt.hist(df[df["class"] == 0][label],color= 'red', label = 'Hadron', alpha = 0.7, density= True) 
    plt.title(label)
    plt.ylabel('Probability')
    plt.xlabel(label)
    plt.legend(loc = 'upper right')
    plt.show()

# It is noted fAsym, fLength, fAlpha seem to decide hadron or gamma particle
# Assumptions: Smaller length, asymmetry and fAlpha likely means it is a gamma particle(vice versa for hadron)

# Create Training, Validation and Test Datasets:
# Shuffle the data
# Training data set is 60% of the data
# Validation data set is 20% of the data(60-80%)
# Test data set is 20% of the data(80-100%)
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

'''Scale the dataset so that the features are all on the same scale(because the maginitudes of the features are different))'''
def ScaleData(dataframe,oversample = False ):
    # Get all the columns except the last column i.e class:
    x = dataframe[dataframe.columns[:-1]].values 
    # Get the last column i.e class:
    y = dataframe[dataframe.columns[-1]].values 
    # Create a StandardScaler object to scale the data by removing the mean and scaling to unit variance:
    scaler = StandardScaler()
    # Scale the data by fitting standard scaler to x and transforming x:
    X = scaler.fit_transform(x) 
    
    if oversample: 
        ros = RandomOverSampler()
        # Take the less oversmapled class and keep sampling such that both class have matching number of samples  
        X, y = ros.fit_resample(X,y) 
         
    # Create 2D numpy array of features and class(horizontally stacked):
    # np.reshape(y, (len(y), 1)) converts the 1D array to a 2D array, X is already a 2D array
    data = np.hstack((X, np.reshape(y, (len(y), 1)))) 
    return data, X, y

# Scale the data:
# Valid and test data are not oversampled because we want to test the model on random sample
train, X_train, y_train = ScaleData(train, oversample = True)
valid, X_valid, y_valid = ScaleData(valid, oversample = False)
test, X_test, y_test = ScaleData(test, oversample = False)

# Number of Hadrons and Gammas in the training set(To Confirm that the data is balanced):
print("Number of Gammas in the training set: ", len(train== 1))
print("Number of Hadrons in the training set: ", len(train == 0))


''' --------------------------------On to the Models we will use to predict the class of a new entry:--------------------------------'''
# ------------------------------------------------------K-Nearest Neighbors Model:------------------------------------------------------
# Uses Binary Classification
# Goal find the best value of k for the KNN model
# Need to find the Euclidean distance between the new entry and the training data

# Create a KNN model with 5 neighbors:
knn_model = KNeighborsClassifier(n_neighbors= 5)
# Fit the model to the training data:
knn_model.fit(X_train, y_train) 

# Predict the class of the Test data:
y_knn_pred = knn_model.predict(X_test) 
# Print the Classification Report of the KNN model:
print("\nClassification Report for KNN Model:\n")
print(classification_report(y_test, y_knn_pred)) # Print the classification report of the KNN model

# ------------------------------------------------------Naive Bayes Model:------------------------------------------------------
'''
 Theory:
 Assumptions : The features are independent of each other
 Uses Bayes Theorem to predict the class of a new entry(i.e given this feature, what is the probability of this class)
 Law of Total Probability: P(B) = P(B|A) * P(A) + P(B|A') * P(A')
 Bayes Theorem: P(A|B) = P(B|A) * P(A) / P(B)
 Bayes Theorem using Law of Total Probability: P(A|B) = P(B|A) * P(A) / (P(B|A) * P(A) + P(B|A') * P(A'))
 Bayes Theorem for multiple features: 
 P(A_k|x_1,x_2,...,x_n) proportional to P(x_1,x_2,...,x_n|A_k) * P(A_k)
 P(A_k|x_1,x_2,...,x_n) = P(x_1,x_2,...,x_n|A_k) * P(A_k) / P(x_1,x_2,...,x_n)
 Maximum A Posteriori(MAP) Estimation:
 Need to find some k such that P(A_k|x_1,x_2,...,x_n) is maximum
 y_pred = argmax(P(A_k|x_1,x_2,...,x_n)) = argmax(P(x_1,x_2,...,x_n|A_k) * P(A_k))
'''

# Create a Naive Bayes model:
nb_model = GaussianNB() 
# Fit the model to the training data:
nb_model = nb_model.fit(X_train, y_train) 
# Predict the class of the Test data:
y_nb_pred = nb_model.predict(X_test) 
# Print Classification report of the Naive Bayes model:
print("\nClassification Report for Naive Bayes Model:\n")
print(classification_report(y_test, y_nb_pred))

# ------------------------------------------------------Logistical Regression Model:------------------------------------------------------
'''
Theory: 
Logistic Regression is a classification algorithm
It uses the sigmoid function to predict the class of a new entry
y_pred = mx + b
Range of y_pred is (-inf, inf)
We need to convert y_pred to a probability between 0 and 1(NECESSARY FOR CLASSIFICATION and AXIOM of PROBABILITY)
probability/(1-probability) is "odds"
probability/(1-probability) = mx + b -> ln(odds) = mx + b -> odds = e^(mx + b) -> probability = e^(mx + b) / (1 + e^(mx + b))
We get the Sigmoid Function -> f(x) = 1 / (1 + e^-x)
'''

# Create a Logistical Regression model:
lg_model = LogisticRegression() 
# Fit the model to the training data:
lg_model = lg_model.fit(X_train, y_train) 
print("\nClassification Report for Logistical Regression Model:\n")
# Print the classification report of the Logistical Regression model:
print(classification_report(y_test, lg_model.predict(X_test))) 

# ------------------------------------------------------Support Vector Machine Model:------------------------------------------------------
'''
Theory:
SVM is a classification algorithm
It uses a hyperplane to separate the data into classes
In 2D, the hyperplane is a line
In 3D, the hyperplane is a plane
Find which hyperplane separates the data into classes with the maximum margin between the hyperplane and the data 
so that the model is more generalized and the data is best separated
Uses maximum margin to find the best hyperplane using the "support vectors" 
'''
# Create a Logistical Regression model:
svm_model = SVC() 
# Fit the model to the training data:
svm_model = svm_model.fit(X_train, y_train) 
print("\nClassification Report for Support Vector Machine Model:\n")
# Print the classification report of the Support Vector Machine Model:
print(classification_report(y_test, svm_model.predict(X_test)))

 
# ------------------------------------------------Neural Networks to implement ML Models:------------------------------------------------
'''
Theory:
Input Layer: Input features
Hidden Layer: Layers between the input and output layer
Output Layer: Output of the model
Neuron: A node in the neural network
Weights: The weights of the edges between the neurons
Bias: The bias of the neurons
Activation Function: A function that is applied to the output of the neurons
Activation Function of the Output Layer: Sigmoid Function
Activation Function of the Hidden Layer: ReLU Function
Loss Function: A function that measures how well the model is performing
Loss Function of the Output Layer: Binary Cross Entropy
Loss Function of the Hidden Layer: Mean Squared Error
Optimizer: A function that optimizes the weights and biases of the model
Optimizer: Stochastic Gradient Descent
Epoch: One iteration of the training data
Batch Size: Number of training examples in one iteration

'''
# Plot the loss of the model:
def plot_loss(history):
    fig, (ax1,ax2) = plt.subplots(1,2)
    
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.xlabel('Epoch')
    ax1.ylabel('Binary crossentropy')
    ax1.legend()
    ax1.grid(True)
    ax1.show()
# Plot the accuracy of the model:
def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)  
    plt.show()
    
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary crossentropy')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    plt.show()
  
def train_nn_model(X_train,y_train, num_nodes, dropout_prob,learning_rate, batch_size, epochs):
    # Create a Neural Network model:
    '''
    Choosing the number of nodes in the output layer:
    Regression: One node, Linear activation Binary Classification: One node, Sigmoid activation Multiclass Classification: One node per class, Softmax activation Multilabel Classification: One node per class, Sigmoid activation
    '''
    nn_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation = 'relu', input_shape = (10,)),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(32, activation = 'relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])
    # Compile the model:
    nn_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate), loss = 'binary_crossentropy', metrics = ['accuracy'])
    history  = nn_model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_split = 0.2, verbose = 0)
    return nn_model, history


# Dictionary using the (num_nodes, dropout_prob, learning_rate, batch_size) as the key and the validation loss as the value
# Validation lost being minimum it is the best model
# Initialize the dictionary of all the models displaying their validation loss, model, and that models history:
model_dict = {} 

# Initialize to infinity so that the any model beats this:
least_val_loss = float('inf') 
epochs = 100
for num_nodes in [16, 32, 64, 128]:
    for dropout_prob in [0,0.2]:
        for lr in [0.01,0.05,0.001]:
            for batch_size in [32,64,128]:
                print("\n\nNumber of Nodes: ", num_nodes, "\nDropout Probability: ", dropout_prob, "\nLearning Rate: ", lr, "\nBatch Size: ", batch_size)
                model, history = train_nn_model(X_train,y_train, num_nodes, dropout_prob,lr, batch_size, epochs)
                #print("\nHistories:\n")
                #plot_history(history)
                val_loss, val_acc = model.evaluate(X_valid, y_valid)
                model_dict[(num_nodes, dropout_prob, lr, batch_size)] = [val_loss, model, history]
                if val_loss < least_val_loss:
                    least_val_loss = val_loss
                    # Least Validation Loss Model:
                    best_nn_model = model
print("\nAll Models Trained\n")
print("Results of all models: ", model_dict, "\n")
print("Best Model: ", best_nn_model, "\n")

for key, value in model_dict.items():
    if value[0] == least_val_loss:
        print("Best Model Parameters: ", key, "\n")
        print("Best Model History: ", value[1], "\n")
        plot_history(value[2])

y_nn_pred = best_nn_model.predict(X_test)

# Convert the predictions to binary:
y_nn_pred = (y_nn_pred > 0.5).astype(int).reshape(-1,)

print("\nClassification Report for Neural Network Model:\n")
print(classification_report(y_test, y_nn_pred))
