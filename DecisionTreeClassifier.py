import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
train = pd.read_csv('iphone_purchase_records_Train.csv')
test = pd.read_csv('iphone_purchase_records_Test.csv')
X_train =train.iloc[:,[1,2]].values
X_test =test.iloc[:,[1,2]].values
Y_train = train.iloc[:,-1].values
Y_test = test.iloc[:,-1].values
SC = StandardScaler()
X_train = SC.fit_transform(X_train)
X_test = SC.transform(X_test)
from matplotlib.colors import ListedColormap
def VisualizingDataset(X_, Y_):
    X1 = X_[:, 0]
    X2 = X_[:, 1]
    for i, label in enumerate(np.unique(Y_)):
        plt.scatter(X1[Y_ == label], X2[Y_ == label],
        color = ListedColormap(("red", "green"))(i),label = label)
VisualizingDataset(X_train,Y_train)
plt.show()
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = "entropy",random_state=0)
model.fit(X_train, Y_train)
cm = confusion_matrix(Y_train, model.predict(X_train))
print(cm)
print("Ti len doan dung tren tap train: ",(cm[0,0]+cm[1,1])/300)
print("Ti len doan sai tren tap train: ",1-(cm[0,0]+cm[1,1])/300)
plot_confusion_matrix(model,X_train,Y_train)
def VisualizingResult(model, X_):
    X1 = X_[:, 0]
    X2 = X_[:, 1]
    X1_range = np.arange(start = X1.min()-1, stop = X1.max()+1, step = 0.01)
    X2_range = np.arange(start= X2.min()- 1, stop= X2.max()+1,step = 0.01)
    X1_matrix, X2_matrix = np.meshgrid(X1_range, X2_range)
    X_grid= np.array([X1_matrix.ravel(), X2_matrix.ravel()]).T
    Y_grid= model.predict(X_grid).reshape(X1_matrix.shape)
    plt.contourf(X1_matrix, X2_matrix, Y_grid, alpha = 0.5,cmap = ListedColormap(("red", "green")))
VisualizingResult(model,X_train)
VisualizingDataset(X_train,Y_train)
accuracy_score(Y_test, model.predict(X_test))
VisualizingResult(model,X_train)
VisualizingDataset(X_test,Y_test)
cm = confusion_matrix(Y_test,model.predict(X_test))
print(cm)
print("Ti len doan dung tren tap test: ",(cm[0,0]+cm[1,1])/100)
print("Ti len doan sai tren tap test: ",1-(cm[0,0]+cm[1,1])/100)
plot_confusion_matrix(model,X_test,Y_test)
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
#Use the pandas apply method to numerically encode our attrition target variable
plt.figure(figsize=(15,15))
plot_tree(model)
plt.show()