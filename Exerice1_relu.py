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

hidden = [5,10,20,50,70,100]
L_score_train = []
L_score_test = []
Time = []

# creation du reseau de neurone 5
mlp=  MLPRegressor(hidden_layer_sizes=(5),activation='relu',solver='adam',alpha=0.01,batch_size='auto',learning_rate='adaptive', learning_rate_init=0.01,max_iter=1000, tol=10**(-4), verbose=True, warm_start=False, early_stopping=True, validation_fraction=0.1, n_iter_no_change=50)  

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
plt.plot(y_test,"bo",label="y_test")
plt.plot(y_test_predict,"rx",label="y_test_predict")
plt.legend()
plt.xlabel("itération")
plt.title("y_test vs y_test_predict")
plt.show()

# Loss curve
plt.plot(mlp.loss_curve_)
plt.xlabel("itération")
plt.title("loss curve")
plt.show()


# creation du reseau de neurone 10
mlp=  MLPRegressor(hidden_layer_sizes=(10),activation='relu',solver='adam',alpha=0.01,batch_size='auto',learning_rate='adaptive', learning_rate_init=0.01,max_iter=1000, tol=10**(-4), verbose=True, warm_start=False, early_stopping=True, validation_fraction=0.1, n_iter_no_change=50)  

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

# creation du reseau de neurone 20
mlp=  MLPRegressor(hidden_layer_sizes=(20),activation='relu',solver='adam',alpha=0.01,batch_size='auto',learning_rate='adaptive', learning_rate_init=0.01,max_iter=1000, tol=10**(-4), verbose=True, warm_start=False, early_stopping=True, validation_fraction=0.1, n_iter_no_change=50)  

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

# creation du reseau de neurone 50
mlp=  MLPRegressor(hidden_layer_sizes=(50),activation='relu',solver='adam',alpha=0.01,batch_size='auto',learning_rate='adaptive', learning_rate_init=0.01,max_iter=1000, tol=10**(-4), verbose=True, warm_start=False, early_stopping=True, validation_fraction=0.1, n_iter_no_change=50)  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train) 
delta4 = time.time()-start
Time.append(delta4)
# prediction des valeurs de test
y_test_predict = mlp.predict(x_test)
# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)
L_score_train.append(score_train)
L_score_test.append(score_test)

# creation du reseau de neurone 70
mlp=  MLPRegressor(hidden_layer_sizes=(70),activation='relu',solver='adam',alpha=0.01,batch_size='auto',learning_rate='adaptive', learning_rate_init=0.01,max_iter=1000, tol=10**(-4), verbose=True, warm_start=False, early_stopping=True, validation_fraction=0.1, n_iter_no_change=50)  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train) 
delta5 = time.time()-start
Time.append(delta5)
# prediction des valeurs de test
y_test_predict = mlp.predict(x_test)
# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)
L_score_train.append(score_train)
L_score_test.append(score_test)

# creation du reseau de neurone 100
mlp=  MLPRegressor(hidden_layer_sizes=(100),activation='relu',solver='adam',alpha=0.01,batch_size='auto',learning_rate='adaptive', learning_rate_init=0.01,max_iter=1000, tol=10**(-4), verbose=True, warm_start=False, early_stopping=True, validation_fraction=0.1, n_iter_no_change=50)  

# entrainement du reseau
start = time.time()
mlp.fit(x_train, y_train) 
delta6 = time.time()-start
Time.append(delta6)
# prediction des valeurs de test
y_test_predict = mlp.predict(x_test)
# score
score_train = mlp.score(x_train,y_train)
score_test = mlp.score(x_test,y_test)
L_score_train.append(score_train)
L_score_test.append(score_test)

# Comparaison hidden vs train_score and test_score
plt.plot(hidden,L_score_train,'r',label="score_train")
plt.plot(hidden,L_score_test,'b',label="score_test")
plt.xlabel("nombre de neurones")
plt.legend()
plt.title('train_score and test_score selon le nombre de neurones du réseau')
plt.show()

# Comparaison hidden vs runtime
plt.plot(hidden,Time,'r')
plt.title('Run time en fonction du nombre de neurones')
plt.xlabel("nombre de neurones")
plt.show()
