import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('patient_data.csv')

# data cleaning 

df.rename(columns={'C': 'Gender'}, inplace=True)
df.columns=df.columns.str.strip().str.lower()
print(df.columns.tolist())
df['takemedication']= df['takemedication'].replace({'Yes ': 'Yes'})
df['nosebleeding']= df['nosebleeding'].replace({'No ': 'No'})
df['systolic']= df['systolic'].replace({'121- 130': '121 - 130', '130+':'130 - 140', '100+': '100 - 110'})
df['stages'] = df['stages'].replace({'HYPERTENSIVE CRISI': 'HYPERTENSIVE CRISIS' , 'HYPERTENSION (Stage-2).' : 'HYPERTENSION (Stage-2)'})
print(df.drop_duplicates())

# nominal (binary / unordered categories) yes/no,0/1
nominal_features = [
    'gender', 'history', 'patient',
    'takemedication', 'breathshortness',
    'visualchanges', 'nosebleeding', 'controlleddiet'
]

# ordinal features (ordered categories)
ordinal_features = [col for col in df.columns if col not in nominal_features]

# target column removed from ordinal list
ordinal_features.remove('stages')

print(nominal_features)
print(ordinal_features)

# -----------------------------
# 2. Encode Binary Features
# -----------------------------

for col in nominal_features:

    # Yes/No columns
    if set(df[col].dropna().unique()) == {'Yes', 'No'}:
        df[col] = df[col].map({'No': 0, 'Yes': 1})

    # Gender column
    elif col == 'gender':
        df[col] = df[col].map({'Male': 0, 'Female': 1})


# -----------------------------
# 3. Encode Age Groups
# -----------------------------

df['age'] = df['age'].map({
    '18-34': 1,
    '35-50': 2,
    '51-64': 3,
    '65+': 4
})

# -----------------------------
# 4. Encode Severity
# -----------------------------

df['severity'] = df['severity'].map({
    'Mild': 0,
    'Moderate': 1,
    'Severe': 2
})

# -----------------------------
# 5. Encode Diagnosis Duration
# -----------------------------

df['whendiagnoused'] = df['whendiagnoused'].map({
    '<1 Year': 1,
    '1 - 5 Years': 2,
    '>5 Years': 3
})

# -----------------------------
# 6. Encode Blood Pressure (Ordinal)
# -----------------------------

df['systolic'] = df['systolic'].map({
    '100 - 110': 0,
    '111 - 120': 1,
    '121 - 130': 2,
    '130+': 3
})

df['diastolic'] = df['diastolic'].map({
    '70 - 80': 0,
    '81 - 90': 1,
    '91 - 100': 2,
    '100+': 3
})

# -----------------------------
# 7. Encode Target Variable
# -----------------------------

df['stages'] = df['stages'].map({
    'NORMAL': 0,
    'HYPERTENSION (Stage-1)': 1,
    'HYPERTENSION (Stage-2)': 2,
    'HYPERTENSIVE CRISIS': 3
})

# visualization (must be before encoding)

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="gender", palette="Set2")
plt.title("Gender Distribution")
plt.show()

df["gender"].value_counts().plot.pie(
    autopct="%1.1f%%",
    figsize=(5,5)
)
plt.title("Gender Distribution (Pie Chart)")
plt.ylabel("")
plt.show()

##hypertension stages visualization 
plt.figure(figsize=(7,5))
sns.countplot(data=df, x="stages", palette="coolwarm")
plt.title("Hypertension Stages Distribution")
plt.xticks(rotation=30)
plt.show()

##take medications vs severity 
plt.figure(figsize=(7,5))
sns.countplot(
    data=df,
    x="takemedication",
    hue="severity",
    palette="Set1"
)
plt.title("TakeMedication vs Severity")
plt.show()

#age vs hypertension

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="age", hue="stages",palette="husl")
plt.title("AGE VS HYPERTENSION")
plt.show()

#systolic vs diastolic 

sns.pairplot(df[["systolic","diastolic", "stages"]], hue="stages", diag_kind="kde", palette="husl")
plt.suptitle("pairplot : systolic vs diastolic")
plt.show()

# model building 

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
X = df.drop('stages', axis=1)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
y = df['stages']

# Check class distribution
print("\n" + "="*50)
print("CLASS DISTRIBUTION ANALYSIS")
print("="*50)
print("\nStages distribution:")
print(y.value_counts().sort_index())
print("\nClass proportions (%):")
print((y.value_counts(normalize=True).sort_index() * 100).round(2))
print("="*50 + "\n")

# Use stratified train-test split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# model testing

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = {}

#for logistic regression

logreg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("Logistic Regression (with balanced class weights)")
print("accuracy :", accuracy_score(y_test,y_pred))
print("classification report:\n", classification_report(y_test,y_pred))
print("confusion matrix:\n" , confusion_matrix(y_test,y_pred))
# accuracy = 0.956

# for decisiontree 

decisionTree= DecisionTreeClassifier()
decisionTree.fit(X_train, y_train)
y_pred = decisionTree.predict(X_test)
print("Decision Tree")
print("accuracy  :",accuracy_score(y_test, y_pred))
# accuracy = 0.923

# for random forest 

randomForest = RandomForestClassifier()
randomForest.fit(X_train, y_train)
y_pred = randomForest.predict(X_test)
print("random forest ")
print(" accuracy is : ", accuracy_score(y_test, y_pred))
# accuracy = 0.923

# for svm 

svm = SVC()
svm.fit(X_train,y_train)
y_pred = svm.predict(X_test)
print("SVM")
print("accuracy is ", accuracy_score(y_test, y_pred))
# accuracy is 0.953

# KNN 

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(" KNN ")
print("accuracy is ", accuracy_score(y_test, y_pred))
# accuracy is 0.953

# RIDGE CLASSIFIER 

rc = RidgeClassifier()
rc.fit(X_train,y_train)
y_pred = rc.predict(X_test)
print("ridge classifier")
print("accuracy is ", accuracy_score(y_test,y_pred))
# accuracy is 0.953

#gaussian 

naive_bayes = GaussianNB()
naive_bayes.fit(X_train,y_train)
y_pred = naive_bayes.predict(X_test)
print("gaussian naive bayes")
print("accuracy score is ", accuracy_score(y_test,y_pred))
#accuracy is 0.890

# now compare all the models and select best one, if it is 100% accuracy then its overfitting so choose wisely 
# in the above the case we selected logistic regression and its accuracy is 95.6

# model deployment 

import joblib

joblib.dump(knn, "knn_model.pkl")
print("model is saved as knn_model.pkl")