import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import random #pour importer des réponses aléatoires à la fin 
import json 
import pickle #pour la sérialisation 
import numpy as np 

import nltk
from nltk.stem import WordNetLemmatizer #permet de ramener un mot à sa forme "racine" exemple   : work, works, working, worked = les mêmes mots 


from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import SGD


lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = [] #va contenir tous les mots individuels (tokens) de tous les patterns
classes = [] #va contenir la liste des tags uniques (greetings, goodbye, etc.)
documents = [] #va contenir des paires (liste de mots du pattern, tag associé) — utile pour garder le lien entre chaque pattern et son tag au moment de construire les données d'entraînement
ignore_letters = ['?','!','.',','] #caractères que l'on souhaite pas considérer 

############PREPARATION DES DONNEES################

for intent in intents['intents'] :
    for pattern in intent['patterns'] :
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append(((word_list), intent['tag']))
        if intent['tag'] not in classes :
            classes.append(intent['tag'])
            
       
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]


words = sorted(set(words)) #set enlève les doublons et sorted remet l'ensemble sous forme de liste 
classes = sorted(classes)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
#permet de récupérer si besoin les listes words et classes sans repasser par toutes les étapes car tout le contenu sera sauvegarder sur disque

training = []
output_empty = [0] * len(classes)

for document in documents : 
    bag = []
    word_patterns  = document[0] #le mot associé 
    word_patterns= [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words :
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1 #document[1] = tag
    training.append([bag, output_row])

random.shuffle(training)

train_x = np.array([row[0] for row in training]) #récupère tous les bags (cela sera utilisé pour challenger le modèle) 
train_y = np.array([row[1] for row in training]) #récupère tous les output_row (cela sera utilisé comme réponse aux productions du modèle)

############PREPARATION DES DONNEES############


############CONSTRUCTION DU RESEAU DE NEURONNES############

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

hist = model.fit(train_x,train_y, epochs=200, batch_size = 5, verbose=1)
model.save('chatbot_model.keras')
print("Done")

############CONSTRUCTION DU RESEAU DE NEURONNES############