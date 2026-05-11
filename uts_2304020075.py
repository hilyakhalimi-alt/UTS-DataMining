import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from google.colab import drive
drive.mount('/content/drive')

train = pd.read_csv('/data_training.csv')
test = pd.read_csv('/data_testing.csv')

train.head()
train.info()
train.describe()

#cek missing value
train.isnull().sum()
test.isnull().sum()

train.fillna(train.mean(), inplace=True)

import matplotlib.pyplot as plt
import seaborn as sns

#visualisasi jumlah tiap kategori
plt.figure(figsize=(8,5))
sns.countplot(x='quality', data=train)

plt.title('Distribusi Quality Wine')
plt.xlabel('Quality')
plt.ylabel('Jumlah Data')

plt.show()

train.hist(figsize=(15,12))
plt.show()

plt.show()

#hubungan ID dan Quality
plt.figure(figsize=(8,5))

sns.scatterplot(
    x='Id',
    y='quality',
    data=train
)

plt.title('Id vs Quality')

#cek duplikat
train.duplicated().sum()

#exploratory data
train['quality'].value_counts()

#menentukan fitur dan target
X = train.drop('quality', axis=1)
y = train['quality']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#feature scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(test)

#split data training
X_train, X_valid, y_train, y_valid = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

#membuat model
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

#evaluasi model
y_pred = model.predict(X_valid)

accuracy_score(y_valid, y_pred)

confusion_matrix(y_valid, y_pred)

classification_report(y_valid, y_pred)

#prediksi data testing
hasil_prediksi = model.predict(X_test_scaled)

submission = pd.DataFrame({
    'id': test['Id'],
    'quality': hasil_prediksi
})

submission.to_csv('hasilprediksi.csv', index=False)

from google.colab import files

files.download('hasilprediksi.csv')

#cek
submission.head()
