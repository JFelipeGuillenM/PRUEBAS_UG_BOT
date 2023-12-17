import random
import json
import pickle
import numpy as np
from tensorflow import keras
import nltk
from nltk.stem import WordNetLemmatizer
import h5py
from config import TELEGRAM_TOKEN
import telebot

from keras.models import load_model

bot = telebot.TeleBot(TELEGRAM_TOKEN)

lematizer = WordNetLemmatizer()
intents = json.loads(open('training.json', encoding='utf-8').read())

words = pickle.load(open('words.pkl', 'rb'), encoding='latin1')
classes = pickle.load(open('classes.pkl', 'rb'), encoding='latin1')
with h5py.File('modelo.h5', 'r') as file:
    model = keras.models.load_model(file)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lematizer.lemmatize(word) for word in  sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESOLD]

    results.sort(key = lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, training_json):
    tag = intents_list[0]['intent']
    list_of_intents = training_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

"""print("GO Bot is running!")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)"""

#Comando start
@bot.message_handler(commands=['start'])
def cmd_start(message):
    bot.send_message(message.chat.id, "Hola", parse_mode = "html")


#Mensajes Telegram
@bot.message_handler(content_types = ['text'])
def bot_mensajes_texto(message):
    ints = predict_class(message.text)
    res = get_response(ints, intents)
    bot.send_message(message.chat.id, res, parse_mode = 'html')

#MAIN----------------------------------------------------------------------------------
if __name__ == '__main__':
    print('Iniciando el bot')
    bot.infinity_polling()
    print('Fin')
    




