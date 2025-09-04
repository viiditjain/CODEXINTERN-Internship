import streamlit as st
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