import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay



dataset=pd.read_csv('Iris.csv') # load iris dataset
print(dataset.head())
print(dataset.info()) 
X = dataset.drop(columns=['Id', 'Species'])
Y = dataset['Species']
print(X.head())
print ("Shape:", X.shape)
print("\nClass_Distribution:",Y.value_counts())
le = LabelEncoder()
Y = le.fit_transform(dataset['Species'])
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm',hue="Species", data=dataset,edgecolor='black',palette='Set1')
plt.title('Sepal Length Vs Sepal Width',fontsize=14)
plt.xlabel('Sepal Length in cm')
plt.ylabel('Sepal Width in cm')
plt.tight_layout()
plt.show() 
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_train,X_test,Y_train,Y_test=train_test_split(X_scaled,Y,test_size=0.2,random_state=42,stratify=Y)
print("Train_set",len(X_test))
print("Test_set",len(X_test))
print("\n=== Train-Test Split ===")
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))
#To find optimal k 
error_rates = []
k_range     = range(1, 10)
 
for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, Y_train)
    pred_vl = model.predict(X_test)
    error_rates.append(1 - accuracy_score(Y_test, pred_vl))
 
best_k = k_range[np.argmin(error_rates)]
print(f"\nElbow method -> Optimal K = {best_k}")
 
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(Y_test, y_pred)
f1  = f1_score(Y_test, y_pred, average='weighted')
cm  = confusion_matrix(Y_test, y_pred)
c = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

 
print("\n=== Model Evaluation ===")
print("Accuracy :", round(acc * 100, 2), "%")
print("F1 Score :", round(f1, 3))
print("\nConfusion Matrix:")
c.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

print("\nClassification Report:")
print(classification_report(Y_test, y_pred, target_names=le.classes_))

 
