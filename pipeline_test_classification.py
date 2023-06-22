"""
Comparing basic classification models 
@Author : Maxim Mangematin--Mathey
"""

#Import metrics
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

#import modeels
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

#Visualisation
import plotly.express as px

from sklearn.model_selection import train_test_split

X = "Features"
y = "target"

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

#Define models
models = {
    "Dummy" : DummyClassifier(),
    "KNN" : KNeighborsClassifier(),
    "SVM" : SVC(verbose=True),
    "Tree" : DecisionTreeClassifier(),
    "Random Forest" : RandomForestClassifier(verbose=True),
    "Naive Bayes" : GaussianNB(),
    "MLP" : MLPClassifier(verbose=True)

}

#Train and test all models
for name, model in models.items():

    print(name)
    print("training")
    model.fit(X_train, y_train)
    print("predict")
    prediction = model.predict(X_test)
    print("Done")

    #Calculate the metrics
    recall = recall_score(y_test, prediction, average="macro")
    accuracy = accuracy_score(y_test, prediction)
    f1 = f1_score(y_test, prediction, average="macro")

    print(
        "Model trained, test scores : \n recall = {} \n accuracy = {} \n f1 = {}".format(recall, accuracy, f1)
    )

    #plot confusion matrix
    fig = px.imshow(confusion_matrix(y_test, prediction), text_auto=True)
    fig.show()


