from re import M
from fastapi import FastAPI
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


"""df = pd.read_csv('clean_data.csv')

X = df['final_text']
y = df['Emotion']

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 0.8, random_state = 42)

cv = CountVectorizer(max_features=1000)
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

model = LogisticRegression(solver = 'sag', max_iter = 200)

model.fit(X_train, y_train)
y = le.inverse_transform(y)"""


app = FastAPI()



@app.get("/{text}")
async def prediction(text :str):
    tfidf ,model = pickle.load(open('lr_combined_tfidf.bin', 'rb'))
    y_pred = model.predict(tfidf.transform([text]))
    return {"text": text, "emotion" : y_pred[0]}