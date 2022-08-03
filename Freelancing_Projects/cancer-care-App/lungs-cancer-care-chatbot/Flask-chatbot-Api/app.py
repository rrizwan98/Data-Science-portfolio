from flask import Flask, jsonify
from flask_cors import CORS, cross_origin
import random
import json
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
#from keras.models import load_model
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

model = load_model('cancer_care_model1.1.h5')

intents = json.loads(open('cancerChatbot.json', encoding="utf8").read())
words = pickle.load(open('cancer_care_words1.1.pkl', 'rb'))
classes = pickle.load(open('cancer_care_classes1.1.pkl', 'rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return(np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['context'] == tag):
            result = random.choice(i['answer'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


''' Flask code '''


app = Flask(__name__)
CORS(app)


@app.route("/", methods=['GET', 'POST'])
def hello():
    return jsonify({"key": "home page value"})

# function to replace '+' character with ' ' spaces


def decrypt(msg):

    string = msg

    # converting back '+' character back into ' ' spaces
    # new_string is the normal message with spaces that was sent by the user
    new_string = string.replace("+", " ")

    return new_string

# here we will send a string from the client and the server will return another
# string with som modification
# creating a url dynamically


@app.route('/home/<name>')
def hello_name(name):

    # dec_msg is the real question asked by the user
    dec_msg = decrypt(name)

    # get the response from the ML model & dec_msg as the argument
    response = chatbot_response(dec_msg)

    # creating a json object
    json_obj = jsonify({"top": {"res": response}})

    return json_obj


if __name__ == '__main__':
    app.run(debug=True)
