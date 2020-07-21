# Analyzed cancer from https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
#Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#SGD Stochastic Gradient Descent library
from sklearn.linear_model import SGDClassifier
#SVM Library
from sklearn.svm import SVC, NuSVC, LinearSVC
#KNN
from sklearn.neighbors import KNeighborsClassifier
#Naive Bayes
from sklearn.naive_bayes import GaussianNB
# Decision Trees
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
#GridSearch
from sklearn.model_selection import GridSearchCV

#Calculate Time
import time

#Read data
data = pd.read_csv('data.csv');

#Drop NULL column
data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

#Display info
data.info()

#Display some data
print(data.head(5))

#Summary data for breast cancer
diagnosis_all = list(data.shape)[0]
diagnosis_categories = list(data['diagnosis'].value_counts())
print("\n Summary, we have total case: ",diagnosis_all," which are malignant:",diagnosis_categories[0]
," and benign:",diagnosis_categories[1])

#Extract only mean feature
features_mean= list(data.columns[1:11])

#Plot distribution
plt.figure(figsize=(15,15))
for i, feature in enumerate(features_mean):
    rows = int(len(features_mean)/2)
    plt.subplot(rows, 2, i+1)
    sns.distplot(data[data['diagnosis']=='M'][feature], bins=12, color='red', label='M');
    sns.distplot(data[data['diagnosis']=='B'][feature], bins=12, color='blue', label='B');
    plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

#Select feature
features_selection = ['radius_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean']

#Transform M,B to 1,0 respectively
mapping = {'M':1, 'B':0}
data['diagnosis'] = data['diagnosis'].map(mapping)

#Import feature_mean to X, labels to y
x = data.loc[:,features_mean]
y = data.loc[:, 'diagnosis']

#Split data to training and test with prob 0.2
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state = 42)


#Stochastic Gradient Descent (SGD)
start_SGD = time.time()

#Build model
clf = SGDClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

end_SGD = time.time()
exc_time= end_SGD-start_SGD

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("SGD Classifier Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Execution time:",exc_time," seconds")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for SGD')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#Support Vector Machine 
start_svc = time.time()

#Build Model
clf = SVC()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

end_svc = time.time()
exc_time= end_svc-start_svc

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("SVC Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Execution time:",exc_time," seconds")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for SVC')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#NuSVC
start_nuSVC = time.time()

#Build Model
clf = NuSVC()
clf.fit(X_train, y_train)
prediciton = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

end_nuSVC = time.time()
exc_time= end_nuSVC-start_nuSVC

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("NuSVC Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Execution time:",exc_time," seconds")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for NuSVC')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#LinearSVC
start_linSVC = time.time()

#Build Model
clf = LinearSVC()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

end_linSVC = time.time()
exc_time= end_linSVC-start_linSVC

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("LinearSVC Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Execution time:",exc_time," seconds")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for LinearSVC')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#KNN
start_knn = time.time()

#Build Model
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

end_knn = time.time()
exc_time= end_knn-start_knn

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("KNN Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Execution time:",exc_time," seconds")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for KNN')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#Naive Bayes
start_naive = time.time()

clf = GaussianNB()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

end_naive = time.time()
exc_time= end_naive-start_naive

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("Naive Bayes Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Execution time:",exc_time," seconds")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for Naive Bayes')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#Random Forest
start_rdForest = time.time()

#Build Model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

end_rdForest = time.time()
exc_time= end_rdForest-start_rdForest

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("Random Forest Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Execution time:",exc_time," seconds")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for Random Forest')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#Extra Trees
start_extraTree = time.time()

clf = ExtraTreesClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

end_extraTree = time.time()
exc_time= end_extraTree-start_extraTree

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("Extra Trees Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Execution time:",exc_time," seconds")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for Extra Trees')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#Decision Trees
start_decTree = time.time()

#Build Model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

end_decTree = time.time()
exc_time= end_decTree-start_decTree

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("Decision Trees Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Execution time:",exc_time," seconds")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for Decision Trees')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()





#Select Feature
x = data.loc[:,features_selection]
y = data.loc[:, 'diagnosis']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#Stochastic Gradient Descent (SGD)
start_SGD = time.time()

#Build model
clf = SGDClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

end_SGD = time.time()
exc_time= end_SGD-start_SGD

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("SGD Classifier Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Execution time:",exc_time," seconds")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for SGD')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#Support Vector Machine 
start_svc = time.time()

#Build Model
clf = SVC()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

end_svc = time.time()
exc_time= end_svc-start_svc

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("SVC Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Execution time:",exc_time," seconds")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for SVC')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#NuSVC
start_nuSVC = time.time()

#Build Model
clf = NuSVC()
clf.fit(X_train, y_train)
prediciton = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

end_nuSVC = time.time()
exc_time= end_nuSVC-start_nuSVC

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("NuSVC Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Execution time:",exc_time," seconds")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for NuSVC')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#LinearSVC
start_linSVC = time.time()

#Build Model
clf = LinearSVC()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

end_linSVC = time.time()
exc_time= end_linSVC-start_linSVC

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("LinearSVC Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Execution time:",exc_time," seconds")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for LinearSVC')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#KNN
start_knn = time.time()

#Build Model
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

end_knn = time.time()
exc_time= end_knn-start_knn

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("KNN Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Execution time:",exc_time," seconds")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for KNN')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#Naive Bayes
start_naive = time.time()

clf = GaussianNB()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

end_naive = time.time()
exc_time= end_naive-start_naive

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("Naive Bayes Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Execution time:",exc_time," seconds")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for Naive Bayes')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#Random Forest
start_rdForest = time.time()

#Build Model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

end_rdForest = time.time()
exc_time= end_rdForest-start_rdForest

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("Random Forest Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Execution time:",exc_time," seconds")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for Random Forest')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#Extra Trees
start_extraTree = time.time()

clf = ExtraTreesClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

end_extraTree = time.time()
exc_time= end_extraTree-start_extraTree

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("Extra Trees Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Execution time:",exc_time," seconds")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for Extra Trees')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#Decision Trees
start_decTree = time.time()

#Build Model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

end_decTree = time.time()
exc_time= end_decTree-start_decTree

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("Decision Trees Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Execution time:",exc_time," seconds")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for Decision Trees')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()




#GridSearchCV Improve Model
x = data.loc[:,features_mean]
y = data.loc[:, 'diagnosis']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#Naive Bayes
start_naive = time.time()

#Set Parameters
parameters = {'priors':[[0.01, 0.99],[0.1, 0.9], [0.2, 0.8], [0.25, 0.75], [0.3, 0.7],[0.35, 0.65], [0.4, 0.6]]}

#Build Model
clf = GridSearchCV(GaussianNB(), parameters, scoring = 'average_precision', n_jobs=-1)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

end_naive = time.time()
exc_time= end_naive-start_naive

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("Naive Bayes Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Execution time:",exc_time," seconds")
print("Best parameters:",clf.best_params_)
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for Naive Bayes')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#Random Forest
start_rdForest = time.time()

#Set Parameters
parameters = {'n_estimators':list(range(1,101)), 'criterion':['gini', 'entropy']}

#Build Model
clf = GridSearchCV(RandomForestClassifier(), parameters, scoring = 'average_precision', n_jobs=-1)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

end_rdForest = time.time()
exc_time= end_rdForest-start_rdForest

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("Random Forests Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Execution time:",exc_time," seconds")
print("Best parameters:",clf.best_params_)
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for Random Forests')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#Extra Trees
start_extraTree = time.time()

#Build Model
clf = GridSearchCV(ExtraTreesClassifier(), parameters, scoring = 'average_precision', n_jobs=-1)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

end_extraTree = time.time()
exc_time= end_extraTree-start_extraTree

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("Extra Trees Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Execution time:",exc_time," seconds")
print("Best parameters:",clf.best_params_)
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for Extra Trees')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()


#Decision Trees
start_decTree = time.time()

#Set Parameter
parameters = {'criterion':['gini', 'entropy'], 'splitter':['best', 'random']}

#Build Model
clf = GridSearchCV(DecisionTreeClassifier(), parameters, scoring = 'average_precision', n_jobs=-1)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

end_decTree = time.time()
exc_time= end_decTree-start_decTree

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("Decision Trees Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Execution time:",exc_time," seconds")
print("Best parameters:",clf.best_params_)
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for Decision Trees')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()
