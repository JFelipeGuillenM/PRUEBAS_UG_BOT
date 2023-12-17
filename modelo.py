import random
import json
import pickle
import numpy as np
import tensorflow as tf

# Importando NLTK y el lematizador de palabras
import nltk
from nltk.stem import WordNetLemmatizer

# Inicializando el lematizador
lemmatizer = WordNetLemmatizer()

# Cargando los datos de entrenamiento desde 'training.json' en la variable 'intents'
intents = json.loads(open('training.json').read())

# Inicializando listas y variables para procesar las palabras y clases
words = [] # Almacena todas las palabras 
classes = [] # Almacena todas las clases 
documents = [] # Almacena las palabras con su respectivo identificador o clase
ignoreLetters = ['¿','?','¡','!','.',',',';'] # Caracteres a ignorar

# Recorriendo los datos del json para rellenar las listas
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokeniza el patrón en palabras
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList) # Agrega las palabras a la lista 'words'
        documents.append((wordList, intent['tag'])) # Agrega las palabras y su clase a 'documents'
        if intent['tag'] not in classes: 
            classes.append(intent['tag'])

# Lematizando y filtrando las palabras, para ordenarlas y eliminar duplicados
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))
classes = sorted(set(classes))

# Guardando las listas de palabras y clases en archivos binarios
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Preparando los datos de entrenamiento
training = []
outputEmpty = [0] * len(classes)

# Para cada documento, se crea un bag de palabras binarias y la etiqueta correspondiente
for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    # Crea un bag de palabras binarias
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)
    # Se crea una lista de salida categórica
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    # Se combina el bag de palabras con la lista de salida y se agregan a la lista 'training'
    training.append(bag + outputRow)

# Mezclando aleatoriamente los datos de entrenamiento
random.shuffle(training)
# Convierte 'training' a una matriz NumPy
training = np.array(training)

# Dividiendo 'training' en matrices de entrada ('trainX') y salida ('trainY')
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Modelo de red neuronal con 128 neuronas 
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# Configura el optimizador y compila el modelo
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrena el modelo
model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)
# Guarda el modelo entrenado en un archivo
model.save('modelo.h5')

print('Done')