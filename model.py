import streamlit as st
import flask
from flask import request, jsonify
# Import the needed libraries
import json
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
import re
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')



app = flask.Flask(__name__)

# Load the data
with open("/Users/ricardo/Desktop/project_data_final_final.json") as file:
  data = json.load(file)

"""## Data Pre-processing"""

# Create lists to store the patterns and tags after tokenization
words =[]
patterns = []
tags_of_patterns = []
classes = []

# Tokenize the data (break the text into individual words or tokens) by looping on the file
for intent in data['intents']:
  for pattern in intent['patterns']:
    tokens = word_tokenize(pattern)
    words.extend(tokens)
    patterns.append(pattern)
    tags_of_patterns.append(intent['tag'])

    # Also create a list of unique tags
    if intent['tag'] not in classes: 
      classes.append(intent["tag"])

classes
#words

#patterns[0:5]

#tags_of_patterns
# Load the stop words file 
with open("/Users/ricardo/Desktop/stop-words.txt") as stop_words:
  stop_words = stop_words.readlines()
  stop_words = ",".join(stop_words)
  stop_words = re.sub('\n',"",stop_words).split(",")

stop_words

#classes

out_empty = [0]*len(classes) # list of zeros whose length is the same as number of tags (classes)
training = []

# Lemmetization
# Initialize the lemmetizer
lemmatizer = WordNetLemmatizer()
patterns

patterns_lower = []

for idx,doc in enumerate(patterns):
    new = doc.lower().translate(str.maketrans('', '', string.punctuation))
    patterns_lower.append(new)

patterns_lower                      

# Vectorization using Bag of Words
for idx,doc in enumerate(patterns_lower):
  bow = []
  if doc not in stop_words:
      text = lemmatizer.lemmatize(doc)
  
  for word in words:
      bow.append(1) if word in text else bow.append(0)
      output_row = list(out_empty)
      output_row[classes.index(tags_of_patterns[idx])] = 1

      training.append([bow,output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

# Separate the features from the target (tags)
patterns = np.array(list(training[:,0]))
tags = np.array(list(training[:,1]))



## Model Evaluation

# Split the data into training dataset (80%) and test dataset (20%) 
X_train, X_test, y_train, y_test = train_test_split(patterns, tags, test_size=0.2, random_state=42)



# Model 1: Artificial Neural Network
classifier = Sequential()
classifier.add(Dense(units = 64, activation = 'relu', input_dim = 2720))
classifier.add(Dropout(rate = 0.6))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(rate = 0.6))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 36, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.summary()
model1 = classifier.fit(X_train, y_train, epochs = 2, shuffle = False)


words_input =[]
# Machine learning model
def predict(input_user):
    # Tokenize the user input
    input_tokens = word_tokenize(input_user)
    words_input.extend(input_tokens)
    print(words_input)



    # Lemmetization
    # Initialize the lemmetizer for input_user
    lemmatizer_user = WordNetLemmatizer()


    training1 = []
    words_input
    out_empty2 = [0]*len(classes)

    patterns_lower_input = []

    for word in words_input:
        new = word.lower().translate(str.maketrans('', '', string.punctuation))
        patterns_lower_input.append(new)

    #patterns_lower_input

    # Vectorization using Bag of Words
    for word in patterns_lower_input:
          bow2 = []
          if word not in stop_words:
              text = lemmatizer_user.lemmatize(word)
              for word in words:
                  bow2.append(1) if word in text else bow2.append(0)
                  output_row2 = list(out_empty2)
                  training1.append([bow2,output_row2])



    training1 = np.array(training1, dtype=object)
    patterns1 = np.array(list(training1[:,0]))


    y_pred = classifier.predict(patterns1)
    y_pred_sum = np.sum(y_pred, axis=0)

    y_pred_sum
    max_indices = np.argmax(y_pred_sum, axis=0)
    tag = classes[max_indices]
    tag

    for intent in data['intents']: 
        if intent["tag"] == tag:
          result = random.choice(intent["responses"])
          print(result)
          
    return result

@app.route('/predict', methods=['POST'])
def predict_api():
    input_user = request.json['input_user']
    result = predict(input_user)
    response = {'result': result}
    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000, debug=True)

    
    
    
    
    
    
    