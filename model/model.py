import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

sns.set_theme()

Data_URL = 'data/Crop_recommendation.csv'

df = pd.read_csv(Data_URL)
print(df.head())
print(df.describe())
df['label'].value_counts().plot(kind='barh')
# plt.tight_layout()
# plt.show()

# No. of classes
num_classes = df['label'].unique().shape[0]
classes = df['label'].unique()

print("Number of classes:", num_classes)
print("Classes:", classes)

grouped = df.groupby("label")

grouped.mean()["N"].plot(kind='barh')
# plt.tight_layout()
# plt.show()


feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

features = df[feature_columns]
labels = df[["label"]]
print(features.head())
print(labels.head())

train_X, test_X, train_Y, test_Y = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)


def evaluate(model):
    predictions = model.predict(test_X)
    acc = accuracy_score(predictions, test_Y)
    return round(acc * 100, 3)


model = LogisticRegression()
model.fit(train_X, train_Y)
acc = evaluate(model)
print(f'LogisticRegression (accuracy): {acc}%')

model = DecisionTreeClassifier(criterion='gini', max_depth=12, random_state=42)
model.fit(train_X, train_Y)
acc = evaluate(model)
print(f'DecisionTreeClassifier with gini (accuracy): {acc}%')

model = DecisionTreeClassifier(criterion='entropy', max_depth=12, random_state=42)
model.fit(train_X, train_Y)
acc = evaluate(model)
print(f'DecisionTreeClassifier with entropy (accuracy): {acc}%')

model = RandomForestClassifier(n_estimators=100)
model.fit(train_X, train_Y)
acc = evaluate(model)
filename = 'model/RandomForestClassifier.sav'
pickle.dump(model, open(filename, 'wb'))
print(f'RandomForestClassifier (accuracy): {acc}%')

model = KNeighborsClassifier()
model.fit(train_X, train_Y)
acc = evaluate(model)
print(f'KNeighborsClassifier (accuracy): {acc}%')

model = AdaBoostClassifier(n_estimators=100)
model.fit(train_X, train_Y)
acc = evaluate(model)
print(f'AdaBoostClassifier (accuracy): {acc}%')

model = GradientBoostingClassifier(n_estimators=100)
model.fit(train_X, train_Y)
acc = evaluate(model)
print(f'GradientBoostingClassifier (accuracy): {acc}%')

model = GaussianNB()
model.fit(train_X, train_Y)
acc = evaluate(model)
print(f'GaussianNB (accuracy): {acc}%')

model = svm.SVC(kernel='rbf')
model.fit(train_X, train_Y)
acc = evaluate(model)
print(f'SVC (accuracy): {acc}%')
