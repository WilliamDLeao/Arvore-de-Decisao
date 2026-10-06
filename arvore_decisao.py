from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

iris = load_iris()

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
stratify=y)


modelo = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42 )

modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print(y_pred[:30])
print(y_test[:30])

acc = accuracy_score(y_test, y_pred)
print(acc)

plt.figure(figsize=(12, 7))
plot_tree(
modelo,
feature_names=iris.feature_names,
class_names=iris.target_names,
filled=True
)
plt.show()