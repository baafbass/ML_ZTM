from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4)

print(X_train.shape)
print(X_test.shape)
