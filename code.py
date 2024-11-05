## importing the libraries required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA




## columns as given in the metdata 
columns = [ 'pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius','grade_of_spondylolisthesis', 'Class']

## getting the dataset into Python
df = pd.read_csv(r"C:\Users\souvi\OneDrive\Desktop\column_2C.dat", header = None, names = columns, sep = ' ')


## taking a look at the dataset briefly
df.head()


## 2 classes : AB or NO (not abnormal) which we will be used for classification 
df['Class'].unique()

## information on dataset 
df.info()

## no null values in the dataset 
df.isnull().sum()

# there are no duplicates 
df.drop_duplicates()

## rounding off the figures to 2 decimal places 
round(df.describe(),2)

## boxplot graph to detect whether any of the features have any outliers 
plt.figure(figsize=(12, 6))
sns.boxplot(data= df, orient="h", palette="Set1")
plt.title('Box plot graph for biomedical features')
plt.show()

# Exploratory data analysis 

## creating a heatmap to understand the correlation between all the variables with each other
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

## to see the overall distribution of outcome as a pie chart 
outcome_counts = df['Class'].value_counts()

# showing it graphically
plt.figure(figsize=(4, 4)) 
plt.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%')
plt.title("Distribution of Class Variable")
plt.show()

## using pairplot to find relationship between each feature variable
sns.pairplot(df, height = 2, aspect = 1)

X = df.drop('Class', axis = 1) # features 

le = LabelEncoder()
Y = le.fit_transform(df['Class']) # encode the class varaible to 0 and 1 

## splitting the data to 80/20 to understand how the model classifies on unseen/test data 
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state= 42,test_size=0.2)

## standardization of data 
scaler = StandardScaler()
features_scaled = scaler.fit_transform(X)

## Create a list to store WCSS values for different k
wcss = []

## Defining the range of k values to try and we use 10 values here 
k_values = range(1, 11)

## Calculate WCSS for each k
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    wcss.append(kmeans.inertia_)

## Plot the WCSS values against the number of clusters (k)
plt.plot(k_values, wcss, marker='o')
plt.title('Elbow Plot for ideal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.show()

## using kmeans clustering method here after finding the number of clusters 
kmeans = KMeans(n_clusters= 3, random_state=0)  # keeping 3 clusters based on WCSS method
cluster_labels = kmeans.fit_predict(features_scaled)

## applying PCA for visualisation
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

plt.figure(figsize=(7, 5))
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=cluster_labels, cmap='viridis')
plt.title('Visualising K-means using PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

## SVM  
svm_classifier = SVC(kernel='linear', C=1)
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)
y_trained = svm_classifier.predict(X_train)

## Evaluating the accuracy of SVM 
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy}")

## Calculating the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

## Accuracy
accuracy = accuracy_score(y_test, y_pred)

## Precision
precision = precision_score(y_test, y_pred)

## Recall
recall = recall_score(y_test, y_pred)

## F1 Score
f1 = f1_score(y_test, y_pred)

# Displaying the results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

## Defining the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],  # Example values
    'kernel': ['linear', 'rbf'],
    'gamma': [0.001, 0.01, 0.1, 1]  # 
}

## Initializing grid search model
grid_search = GridSearchCV(svm_classifier, param_grid, cv=2, scoring='accuracy')

## Fitting it to the data
grid_search.fit(X_train, y_train)

## Getting best parameters and scores
best_parameters = grid_search.best_params_
best_score = grid_search.best_score_

print("Optimal Parameter:", best_parameters)
print("Improved accuracy:", best_score)
