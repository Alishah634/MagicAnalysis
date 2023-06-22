
# Dataset Information (from UCI Machine Learning Repository)
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. 
Irvine, CA: University of California, School of Information and Computer Science.
Donated by: P. Savicky Institute of Computer Science, AS of CR Czech Republic savicky '@' cs.cas.cz

# Project Goal
The primary aim of this analysis is to leverage supervised learning methodologies to proficiently classify particles as either gamma or hadron, relying on distinct features. The selection of an optimal model will be determined based on the accuracy score, ensuring the highest predictive accuracy and model reliability.

# Initial Analysis
An initial histogram was created to visually identify which properties are inidicators of particle type i.e gamma or hadron.
It is noted fAsym, fLength, fAlpha seem to decide hadron or gamma particle
Assumptions: Smaller length, asymmetry and fAlpha likely means it is a gamma particle(vice versa for hadron)

This project focuses on creating training, validation, and test datasets for machine learning models. The dataset is shuffled, and the split percentages for the training, validation, and test sets are defined as follows:
- Training dataset: 60% of the data
- Validation dataset: 20% of the data (60-80%)
- Test dataset: 20% of the data (80-100%)

Additionally, the dataset is scaled to ensure that all features are on the same scale using the StandardScaler from the scikit-learn library. The scaled dataset is then divided into the respective sets: train, valid, and test.

## Usage

The `ScaleData()` function is used to scale the dataset, and it takes the following parameters:
- `dataframe`: The input dataframe to be scaled
- `oversample`: A boolean flag indicating whether oversampling is required for the dataset (default is `False`)

The function returns the scaled data along with the individual features and labels.

The README.md file also includes code snippets and explanations for implementing several machine learning models:

### K-Nearest Neighbors Model
- Uses binary classification
- Finds the best value of k for the KNN model
- Predicts the class of the test data using the trained model
- Displays the classification report for the KNN model

### Naive Bayes Model
- Assumes feature independence
- Predicts the class of the test data using the trained model
- Displays the classification report for the Naive Bayes model

### Logistic Regression Model
- Uses the sigmoid function to predict the class of a new entry
- Predicts the class of the test data using the trained model
- Displays the classification report for the Logistic Regression model

### Support Vector Machine Model
- Uses a hyperplane to separate the data into classes
- Finds the best hyperplane using the support vectors
- Predicts the class of the test data using the trained model
- Displays the classification report for the Support Vector Machine model

### Neural Networks
- Implements various machine learning models using neural networks
- Uses input, hidden, and output layers with appropriate activation functions
- Trains the models using the training data and evaluates them on the validation data
- Selects the best model based on the validation loss
- Predicts the class of the test data using the best model
- Displays the classification report for the Neural Network model

## Results

The README.md file provides a summary of all the trained models, and the details of their working. It also highlights the best-performing model based on the validation loss.

## Dependencies

The following dependencies are used in this project:
- numpy
- pandas
- scikit-learn
- matplotlib
- tensorflow
