import numpy as np
import matplotlib.pyplot as plt
import time


# Importing the dataset 
dataset = np.genfromtxt("yacht_hydrodynamics.data", delimiter='') 
X = dataset[:, :-1] 
y = dataset[:, -1]  

# Splitting the dataset into the Training set and Test set (here 20% of data for test) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(X, y,random_state=0, test_size = 0.20) 

# Feature Scaling 
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
x_train = sc.fit_transform(x_train) 
x_test = sc.transform(x_test) 

# MLP Regression
from sklearn.neural_network import MLPRegressor

#Exercice 1

L_score_train = []
L_score_test = []
Time = []


# creation du reseau de neurone 2x5
mlp=  MLPRegressor(hidden_layer_sizes=(5,5),activation='relu',solver='adam',alpha=0.01,batch_size='auto',learning_rate='adaptive', learning_rate_init=0.01,max_iter=1000, tol=10**(-4), verbose=True, warm_start=False, early_stopping=True, validation_fraction=0.1, n_iter_no_change=50)  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train) 
delta1 = time.time()-start
Time.append(delta1)
# prediction des valeurs de test
y_test_predict = mlp.predict(x_test)
# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)
L_score_train.append(score_train)
L_score_test.append(score_test)
# comparaison reel/predict
plt.plot(y_test,"bo",label ="y_test")
plt.plot(y_test_predict,"rx",label ="y_test_predict")
plt.legend()
plt.xlabel("itération")
plt.title("y_test vs y_test_predict pour un réseau de de 2 fois 5 neurones")
plt.show()


# creation du reseau de neurone 2x10
mlp=  MLPRegressor(hidden_layer_sizes=(10,10),activation='relu',solver='adam',alpha=0.01,batch_size='auto',learning_rate='adaptive', learning_rate_init=0.01,max_iter=1000, tol=10**(-4), verbose=True, warm_start=False, early_stopping=True, validation_fraction=0.1, n_iter_no_change=50)  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train) 
delta2 = time.time()-start
Time.append(delta2)
# prediction des valeurs de test
y_test_predict = mlp.predict(x_test)
# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)
L_score_train.append(score_train)
L_score_test.append(score_test)
# comparaison reel/predict
plt.plot(y_test,"bo",label ="y_test")
plt.plot(y_test_predict,"rx",label ="y_test_predict")
plt.legend()
plt.xlabel("itération")
plt.title("y_test vs y_test_predict pour un réseau de de 2 fois 10 neurones")
plt.show()


# creation du reseau de neurone 2x20
mlp=  MLPRegressor(hidden_layer_sizes=(20,20),activation='relu',solver='adam',alpha=0.01,batch_size='auto',learning_rate='adaptive', learning_rate_init=0.01,max_iter=1000, tol=10**(-4), verbose=True, warm_start=False, early_stopping=True, validation_fraction=0.1, n_iter_no_change=50)  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train) 
delta3 = time.time()-start
Time.append(delta3)
# prediction des valeurs de test
y_test_predict = mlp.predict(x_test)
# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)
L_score_train.append(score_train)
L_score_test.append(score_test)

# comparaison reel/predict
plt.plot(y_test,"bo",label ="y_test")
plt.plot(y_test_predict,"rx",label ="y_test_predict")
plt.legend()
plt.xlabel("itération")
plt.title("y_test vs y_test_predict pour un réseau de de 2 fois 20 neurones")
plt.show()
