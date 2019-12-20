import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from keras.layers.convolutional import Conv1D
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from keras.layers.recurrent import LSTM

from numpy import array
from numpy import asarray
from numpy import zeros
import seaborn as sns
'''La base de datos "requerimientos2" contiene la información
de la base de datos PROMISE la cual cuenta con 625 requisitos'''

requeriments = pd.read_csv("requerimientos2.csv")
#Verificamos si el conjunto de datos contiene algun valor nulo
requeriments.isnull().values.any()
requeriments.shape
#Imprimimos parte de la data prara verificarla
print (requeriments.head())

requeriments["Requeriments"][3]
#Mostramos la distribución de requsitos Funcionales y No Funcionles
sns.countplot(x='Type', data=requeriments)

'''Definimos una funcion preprocess que toma una cadena de texto 
como parametro y luego eliminar caracteres eseciales o aquellos 
que no son pructiferos'''
def preprocess_text(sen):
   
    sentence = remove_tags(sen)
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)
'''Almacenaremos los datos de la fase anterior en el vector X,
'''
X = []
sentences = list(requeriments['Requeriments'])
for sen in sentences:
    X.append(preprocess_text(sen))
#X[3]
y = requeriments['Type']

'''Convertimos nuestras etiquetas en dígitos, en este caso seran dos
Funcionales y No Funcionales. Funcionles representara a 1 y No funcionales a 0''' 
y = np.array(list(map(lambda x: 1 if x=="funcional" else 0, y)))

'''Para poder entrenar y evaluar nuestro conjunto de datos, usamos el
metodo train_test_split, dividimos un 80% para train y 20% para test'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#Procesos embedding
'''En primer lugar utilizamos la clases Tokenier para crear un diccionario
de palabra a indice. En el diccionario de palabra a indice, cada palabra en el corpus
se usa como una clave, mientras que un indice se usa como valor de la clave'''
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

'''La variable X_train contiene listas donde cada lista contiene enteros
. Cada lista corresponde a cada oracion en el conunto de entrenamiento'''
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1
'''El tamaño máximo de cada lista puede ser 100 de ser inferiores a 1000 se 
rellenaran con 0'''
maxlen = 20

#Ubicamos el tamaño de vocabulario y realizamos el relleno tanto como 
#en train y test
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

'''Utilizamos GloVe embeddings para crear nuestra matriz de caracteristicas
, entonces cargammos embeddings de palabras GloVe y creamos un diccionario
que contendrá palabras como claves y su lista embeddings correspondiente como valores '''
embeddings_dictionary = dict()
glove_file = open('glove.twitter.27B.100d.txt')#, encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

'''Creamos una matriz embeddings donde cada número de filas correpsondera
al indice de la palabra ene l corpus. La matriz tendrá 100 columnas donde 
cada columna contendrá las incrustaciones de palabras GloVe para las palabras en nuestro corpus.'''
embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

''' Creamos nuestra capa embedding, que tendra una longitud de 100, y el vector
de salida tambien sera 100.'''

model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())

history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])


plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

model = Sequential()

embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)

model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())

history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()

instance = X[34]
print(instance)
'''

model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)
model.add(LSTM(128))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())

history = model.fit(X_train, y_train, batch_size=128, epochs=50, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

'''
instance = tokenizer.texts_to_sequences(instance)

flat_list = []
for sublist in instance:
    for item in sublist:
        flat_list.append(item)

flat_list = [flat_list]

instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)

model.predict(instance)

#instance = X[57]
#print(instance)
