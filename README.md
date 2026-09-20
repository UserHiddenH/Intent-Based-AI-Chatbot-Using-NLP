# Intent-Based AI Chatbot (NLTK + Neural Network)
(in progress...)
### (with a custom JSON intents dataset)

## Overview

This project implements a **retrieval-based chatbot** that classifies user messages into predefined **intents** and responds with a matching pre-written reply.

The goal of this project is to understand how a simple NLP pipeline works end-to-end — from raw text to a trained neural network classifier — using **NLTK** for language preprocessing and **TensorFlow/Keras** to build and train the model.

---

## Features

* Custom intents dataset (JSON): tags, patterns, and responses
* Text preprocessing with NLTK (tokenization, lemmatization)
* Vocabulary construction from all training patterns
* Bag-of-words text representation
* One-hot encoding of intent classes
* Fully connected neural network (Keras Sequential model)
* Dropout layers to reduce overfitting
* Gradient descent optimization (SGD)
* Data serialization with Pickle (vocabulary + classes)
* Random response selection for more natural-sounding replies
* Real-time intent prediction from user input

---

## Network Architecture

```
Input Layer (bag-of-words, size = vocabulary length)
        │
        ▼
Dense Layer (128 neurons)
      ReLU
        │
     Dropout
        │
        ▼
Dense Layer (64 neurons)
      ReLU
        │
     Dropout
        │
        ▼
Output Layer (size = number of intents)
      Softmax
```

Each output neuron represents one intent tag (e.g. `greetings`, `goodbye`, `thanks`...).

---

## Dataset

The project uses a **custom-built `intents.json` dataset**, structured as follows:

* **Tags:** intent categories (e.g. greetings, goodbye, thanks, help...)
* **Patterns:** example phrases a user might type for each intent (formal and informal)
* **Responses:** possible bot replies for each intent, picked at random at runtime

Before training, all patterns are:

* Tokenized into individual words
* Lemmatized to their base form (e.g. "running" → "run")
* Converted into bag-of-words vectors representing the full training vocabulary

---

## Technologies

* Python
* NLTK
* TensorFlow / Keras
* NumPy
* Pickle
* JSON

---

## Training

The network is trained on bag-of-words vectors (`X`) mapped to one-hot encoded intent labels (`y`).

During training, the following operations are performed:

1. Load and preprocess `intents.json`
2. Build vocabulary and class list
3. Convert each pattern into a bag-of-words vector
4. One-hot encode the corresponding intent
5. Shuffle the training data
6. Train the Keras model using the SGD optimizer
7. Save the trained model and serialized vocabulary/classes for reuse

---

## Results

The trained model can:

* Predict the intent behind a new, unseen user message
* Select a fitting response from the matching intent's response list
* Hold a simple real-time text conversation in the terminal

---

## Learning Objectives

This project was developed to gain a deeper understanding of:

* Natural Language Processing (NLP) fundamentals
* Tokenization and lemmatization
* Bag-of-words text representation
* Multi-class classification with one-hot encoding
* Building and training a neural network with Keras
* Avoiding overfitting with Dropout
* Data serialization with Pickle

---

## Possible Improvements

Future improvements include:

* Part-of-speech tagging for more accurate lemmatization
* Multi-intent detection (handling messages with more than one intent)
* Confidence threshold with a fallback response for uncertain predictions
* Larger and more balanced intents dataset
* Simple web or GUI interface (e.g. Flask)
* Model evaluation metrics (accuracy, confusion matrix)

---
