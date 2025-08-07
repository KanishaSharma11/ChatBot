import random
import json
import pickle
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt') #Pre-trained model for splitting text into words (tokenization).
nltk.download('wordnet') #A lexical database used by the lemmatizer.

lemmatizer = WordNetLemmatizer()

# Load intents
with open(r'D:\Projects\ChatBot\FinTech\finance_intents.json') as file:
    intents = json.load(file)

# Preprocessing
words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and clean words
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save vocabularies
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    # Create bag of words
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # Create output row
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([np.array(bag), np.array(output_row)])

# Shuffle and separate features and labels
random.shuffle(training)

# Convert to separate arrays
train_x = np.array([x[0] for x in training])
train_y = np.array([x[1] for x in training])

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
])

model.compile(loss='categorical_crossentropy', 
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

# Train model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('finance_chatbot_model.h5')

print("Model trained and saved successfully!")
# Save the model architecture and weights
model_json = model.to_json()    
