import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = pd.read_csv('C:/Users/pjyim/Documents/프로그래밍 과제/시험 과제/traffic_accidents.csv')


print(data.info())
print(data.describe())


data = data.dropna()


data = pd.get_dummies(data, drop_first=True)


X = data.drop('accident_severity', axis=1)  # 'accident_severity'가 목표 변수라고 가정
y = data['accident_severity']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))


feature_importances = pd.Series(model.feature_importances_, index=data.drop('accident_severity', axis=1).columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Feature Importances")
plt.show()