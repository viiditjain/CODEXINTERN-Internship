import streamlit as st
<<<<<<< HEAD
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def trasform_text(text):
  text = text.lower()  #lowertext
  text = nltk.word_tokenize(text)  #tokenize

  y =[]
  for i in text:
    if i.isalnum():    #removing special characters
        y.append(i)

  text = y[:] #copy list
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Spam Mail Detector")

input_sns = st.text_area("Enter the message

if st.button('Predict'):
    # 1. preprocessing
    trasformed_sms = trasform_text(input_sns)
    # 2. vectorize
    vector_input = tfidf.transform([trasformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("No Spam")    
=======
import numpy as np
import pickle

with open("iris_dataset.pkl", 'rb') as f:
    model = pickle.load(f)

st.title("Iris Flower Prediction")   

speal_length = st.slider("speal length(cm)", 0.0, 8.0)
speal_width = st.slider("speal width(cm)", 0.0, 8.0)
petal_length = st.slider("petal length(cm)", 0.0, 8.0)
petal_width = st.slider("petal width(cm)", 0.0, 8.0)

if st.button("Prediction"):
    input_data = np.array([[speal_length, speal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    species = ['Setosa','Versicolor','Virginica']
    st.success(f"Predicted Iris Species:{species[prediction[0]]}")
>>>>>>> c3b3895f5234b475c0cdac98c979c6346ad29f0d
