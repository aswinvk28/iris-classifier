from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def dataset():

    iris_data = load_iris(as_frame=True)
    iris_dataframe, target = iris_data['data'], iris_data['target']

    return iris_dataframe, target

def train(X_train, y_train, X_test, y_test):

    dtree = DecisionTreeClassifier(max_depth=None)
    dtree.fit(X_train, y_train)

    score = dtree.score(X_test, y_test)

    return dtree, score

def predict(dtree, X_test):

    y_pred = dtree.predict(X_test)

    return y_pred

def confusion_matrix_plot(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)
    cm_dataframe = pd.DataFrame(cm, columns=['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica'], index=['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica'])
    ax = sns.heatmap(cm_dataframe, annot=True)
    ax.figure.savefig("outputs/confusion_matrix_plot.png", dpi=300, bbox_inches="tight")
    return cm_dataframe, ax

def main():

    parser = argparse.ArgumentParser(description="Python Executable to Train Iris Model and print Confusion Matrix")

    parser.add_argument("-ts", "--test-size", type=float, help="Test Size", default=None)
    parser.add_argument("-rs", "--random-state", type=int, help="The random state", default=None)
    
    args = parser.parse_args()

    iris_dataframe, target = dataset()
    X_train, X_test, y_train, y_test = train_test_split(iris_dataframe, target, test_size=args.test_size, random_state=args.random_state)

    dtree, score = train(X_train, y_train, X_test, y_test)

    y_pred = predict(dtree, X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy Score = {acc*100}% for model = DecisionTreeClassifier")
    
    confusion_matrix_plot(y_test, y_pred)

if __name__ == "__main__":

    main()