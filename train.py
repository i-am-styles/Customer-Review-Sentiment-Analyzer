import pandas as pd
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,accuracy_score

df = pd.read_csv("data.csv")

# print(df.isna().sum()) no null data found

X = df["review"]
y = df["sentiment"]

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,stratify=y)

model = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("model", MultinomialNB())
])

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

# print("Actual")
# print(y_test.head())
#
# print("predicted")
# print(y_pred[:5])

print(classification_report(y_test,y_pred))
accuracy = accuracy_score(y_test,y_pred)

joblib.dump(model,filename="model.joblib")
joblib.dump(accuracy,filename="accuracy.joblib")

model = joblib.load("model.joblib")

print(type(model))
print(model)
