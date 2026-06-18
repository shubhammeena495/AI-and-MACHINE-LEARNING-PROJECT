import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

absolute_path = os.path.dirname(__file__)
os.chdir(absolute_path)
dataset = pd.read_csv("diabetes.csv")
dataset.isnull().sum()

plt.figure(figsize =(10,8))
sns.heatmap(dataset.corr(), annot = True, fmt= ".3f", cmap= "YlGnBu") # Fixed color map typo "Y1GnBu" to "YlGnBu"
plt.title("correlation heatmap")

#explore pregnancy and target variable 
plt.figure(figsize = (10,8))
#plotting density function of pregnancies and target variable
kde = sns.kdeplot(dataset["Pregnancies"][dataset["Outcome"]==1], color = "Red", fill = True)
kde = sns.kdeplot(dataset["Pregnancies"][dataset["Outcome"]==0], color = "Blue", fill = True)
kde.set_xlabel("Pregnancies")
kde.set_ylabel("Density")
kde.legend(["Positive","Negative"])

#exploring glucose and target variable
plt.figure(figsize = (10,8))
sns.violinplot(data=dataset, x="Outcome", y = "Glucose", split = True, linewidth=2, inner = "quart")

#explore Glucose and target variable 
plt.figure(figsize = (10,8))
#plotting density function of Glucose and target variable
kde = sns.kdeplot(dataset["Glucose"][dataset["Outcome"]==1], color = "Red", fill = True)
kde = sns.kdeplot(dataset["Glucose"][dataset["Outcome"]==0], color = "Blue", fill = True)
kde.set_xlabel("Glucose")
kde.set_ylabel("Density")
kde.legend(["Positive","Negative"])

#replace 0 values with the mean/median of the resprctive feature
#glucose 
dataset["Glucose"] = dataset["Glucose"].replace(0, dataset["Glucose"].median()) # Fixed typo "glucose" -> "Glucose"
#BloodPressure 
dataset["BloodPressure"] = dataset["BloodPressure"].replace(0, dataset["BloodPressure"].median())
#BMI 
dataset["BMI"] = dataset["BMI"].replace(0, dataset["BMI"].mean())
#SkinThickness
dataset["SkinThickness"] = dataset["SkinThickness"].replace(0, dataset["SkinThickness"].mean())
#Insulin
dataset["Insulin"] = dataset["Insulin"].replace(0, dataset["Insulin"].mean())

#splitting the dependent and independent variable
x = dataset.drop(["Outcome"], axis = 1)
y = dataset["Outcome"]

#splitting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
training_accuracy = []
test_accuracy = []

for n_neighbors in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
    #check accuracy score
    training_accuracy.append(knn.score(x_train, y_train))
    test_accuracy.append(knn.score(x_test, y_test))

# Fixed: Moved plotting out of the loop
plt.figure(figsize=(10, 8))
plt.plot(range(1, 11), training_accuracy, label = "training_accuracy") # Fixed label typo
plt.plot(range(1, 11), test_accuracy, label = "test_accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train, y_train)
print(knn.score(x_train , y_train), ":Training accuracy (KNN)")
print(knn.score(x_test , y_test), ":test accuracy (KNN)")

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state = 0)
dt.fit(x_train, y_train)
print(dt.score(x_train , y_train), ":Training accuracy (DT)")
print(dt.score(x_test , y_test), ":test accuracy (DT)")

dt1 = DecisionTreeClassifier(random_state=0, max_depth = 3)
dt1.fit(x_train, y_train)
print(dt1.score(x_train , y_train), ":Training accuracy (DT max_depth=3)")
print(dt1.score(x_test , y_test), ":test accuracy (DT max_depth=3)")

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=42)
mlp.fit(x_train, y_train)
print(mlp.score(x_train , y_train), ":Training accuracy (MLP)")
print(mlp.score(x_test , y_test), ":test accuracy (MLP)")

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test) 
mlp1 = MLPClassifier(random_state=0)
mlp1.fit(x_train_scaled, y_train)
print(mlp1.score(x_train_scaled , y_train), ":Training accuracy (Scaled MLP)")
print(mlp1.score(x_test_scaled, y_test), ":test accuracy (Scaled MLP)")
 