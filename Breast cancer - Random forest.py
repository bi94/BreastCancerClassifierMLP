import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
# Prepare dataset for training
df = pd.read_csv("data.csv", sep = ',')
a = df.replace(['M','B'], [1,0])
a = a.dropna(axis = 1)
train, test = train_test_split(a.values, test_size = 0.30)
y_train, y_test = train[:, 1], test[:, 1]
x_train, x_test = np.delete(train, [0,1], axis = 1), np.delete(test, [0,1], axis = 1)
# Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)
# Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
b = list(a.drop(['id','diagnosis'], axis=1))
feature_importance = pd.Series(clf.feature_importances_,index=b).sort_values(ascending=False)
# Creating a bar plot
sns.barplot(x=feature_importance, y=feature_importance.index)
# Add labels to the graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Important Features in Breast Cancer Classification")
plt.legend()
plt.show()