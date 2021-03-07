from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    precision_recall_fscore_support, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from IPython.display import Image

pd.options.display.max_columns = 10
dataframe = pd.read_csv("titanic.csv")

# print(dataframe.keys())

# print(dataframe.head())

# print(dataframe.describe())

# column = dataframe[' Sex']
# print(column)

# small_dataframe = dataframe[["Age", "Sex"]]
# print(small_dataframe.head())

dataframe["Male"] = (dataframe["Sex"] == "male")
# print(dataframe["Male"])

# print(dataframe[" Fare"].values)

# small_array = small_dataframe.values
# print(small_array.shape)

# print(small_array[1])
# print(small_array[:, 0])

# print(small_array[small_array[:, 0] < 18])
# print((small_array[:, 0] < 18).sum())

# plt.scatter(dataframe[" Age"], dataframe[" Fare"], c=dataframe[" Pclass"])
# plt.xlabel("Age")
# plt.ylabel("Fare")
# plt.plot([0, 80], [80, 5])
# plt.show()

x = dataframe[["Pclass", "Male", "Age", "Siblings/Spouses", "Parents/Children", "Fare"]].values
y = dataframe["Survived"].values
# x_train, x_test, y_train, y_test = train_test_split(x, y)
# plt.scatter(dataframe["Fare"], dataframe["Age"], c=dataframe["Survived"])
# plt.xlabel("Fare")
# plt.ylabel("Age")
# plt.show()

# model = LogisticRegression()
# model.fit(x_train, y_train)
# kf = KFold(n_splits=5, shuffle=True)
"""for train, test in kf.split(x_train):
    print(train, test)"""
# splits = list(kf.split(x))
"""for i in range(5):
    my_split = splits[i]
    train_indices, test_indices = my_split
    x_train = x[train_indices]
    x_test = x[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    model = LogisticRegression()
    model.fit(x_train, y_train)
    print(i, "--> ", model.score(x_test, y_test))"""
# first_split = splits[0]
# print(first_split)
# train_indices, test_indices = first_split
"""print("training set indices: ", train_indices)
print("test set indices: ", test_indices)
x_train = x[train_indices]
x_test = x[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]
print("x_train: ", x_train)
print("y_train: ", y_train)
print("x_test: ", x_test)
print("y_test: ", y_test)
model.fit(x_train, y_train)
print(model.score(x_test, y_test))"""

# print(model.coef_, model.intercept_)
# print(model.predict(x[:5]))

# yPrediction = model.predict(x_test)
# print((y == yPrediction).sum() / y.shape[0])
# print(model.score(x, y))
"""print("logistic regression")
print("accuracy: ", accuracy_score(y_test, yPrediction))
print("precision: ", precision_score(y_test, yPrediction))
print("recall: ", recall_score(y_test, yPrediction))"""

"""cancer_data = load_breast_cancer()
# print(cancer_data.keys())
# print(cancer_data["feature_names"])
cancer_data_frame = pd.DataFrame(cancer_data["data"], columns=cancer_data["feature_names"])
cancer_data_frame["target"] = cancer_data["target"]
# print(cancer_data_frame.head)

cancer_model = LogisticRegression(solver="liblinear")
x = cancer_data_frame[cancer_data.feature_names].values
y = cancer_data_frame["target"].values
cancer_model.fit(x, y)

# print(cancer_model.predict([x[0]]))
# print(cancer_model.score(x, y))
y_prediction = cancer_model.predict(x)

# print("accuracy: ", accuracy_score(y, y_prediction))
# print("precision: ", precision_score(y, y_prediction))
# print("recall: ", recall_score(y, y_prediction))
# print("f1 score: ", f1_score(y, y_prediction))
# print("confusion matrix", confusion_matrix(y, y_prediction))

x_train, x_test, y_train, y_test = train_test_split(x, y)
# print("whole dataset:", x.shape, y.shape)
# print("training set:", x_train.shape, y_train.shape)
# print("test set:", x_test.shape, y_test.shape)

cancer_model_new = LogisticRegression(solver="liblinear")
cancer_model_new.fit(x_train, y_train)
y_prediction_new = cancer_model_new.predict(x_test)
# print(cancer_model_new.score(x_test, y_test))

cancer_model_new2 = LogisticRegression(solver="liblinear")
cancer_model_new2.fit(x_train[:, 0:2], y_train)
y_pred_proba2 = cancer_model_new2.predict_proba(x_test[:, 0:2])
# print("area under the roc curve 2: ", roc_auc_score(y_test, y_pred_proba2[:, 1]))

sensitivity_score = recall_score


def specificity_score(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
    return r[0]


# print("sensitivity: ", sensitivity_score(y_test, y_prediction_new))
# print("specificity: ", specificity_score(y_test, y_prediction_new))
# print("predict proba: ", cancer_model_new.predict_proba(x_test))
y_pred_proba = cancer_model_new.predict_proba(x_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
fpr2, tpr2, thresholds2 = roc_curve(y_test, y_pred_proba2[:, 1])
plt.plot(fpr, tpr)
plt.plot(fpr2, tpr2)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("1 - specificity")
plt.ylabel("sensitivity")
plt.show()

# print("area under the roc curve: ", roc_auc_score(y_test, y_pred_proba[:, 1]))

y_pred_2 = cancer_model_new.predict_proba(x_test)[:, 1] > 0.75
# print("precision: ", precision_score(y_test, y_pred_2))
# print("recall: ", recall_score(y_test, y_pred_2))"""

tree = DecisionTreeClassifier()
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)
tree.fit(x, y)
# y_prediction = tree.predict(x_test)
# print("decision tree")
# print("accuracy: ", accuracy_score(y_test, y_prediction))
# print("precision: ", precision_score(y_test, y_prediction))
# print("recall: ", recall_score(y_test, y_prediction))
"""kf = KFold(n_splits=5, shuffle=True)
for i in ["gini", "entropy"]:
    print(i)
    accuracy = []
    precision = []
    recall = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        tree = DecisionTreeClassifier(criterion=i)
        tree.fit(x_train, y_train)
        y_pred = tree.predict(x_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
    print("accuracy: ", np.mean(accuracy))
    print("precision: ", np.mean(precision))
    print("recall: ", np.mean(recall))"""

feature_names = ["Pclass", "Male", "Age", "Siblings/Spouses", "Parents/Children", "Fare"]
dot_file = export_graphviz(tree, feature_names=feature_names)
graph = graphviz.Source(dot_file)
graph.render(filename='tree', format='png', cleanup=True, view=True)



