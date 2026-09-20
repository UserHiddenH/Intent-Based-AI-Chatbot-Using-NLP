import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import random #pour importer des réponses aléatoires à la fin 
import json 
import pickle #pour la sérialisation 
import numpy as np 

import nltk
from nltk.stem import WordNetLemmatizer #permet de ramener un mot à sa forme "racine" exemple   : work, works, working, worked = les mêmes mots 


from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = [] #va contenir tous les mots individuels (tokens) de tous les patterns
classes = [] #va contenir la liste des tags uniques (greetings, goodbye, etc.)
documents = [] #va contenir des paires (liste de mots du pattern, tag associé) — utile pour garder le lien entre chaque pattern et son tag au moment de construire les données d'entraînement
ignore_letters = ['?','!','.',','] #caractères que l'on souhaite pas considérer 

for intent in intents['intents'] :
    for pattern in intent['patterns'] :
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append(((word_list), intent['tag']))
        if intent['tag'] not in classes :
            classes.append(intent['tag'])
            
       
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]


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
training = np.array(training)


train_x = list(training[:,0]) #récupère tous les bags (cela sera utilisé pour challenger le modèle) 
train_y = list(training[:,1]) #récupère tous les output_row (cela sera utilisé comme réponse aux productions du modèle)