
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from nltk.corpus import stopwords
from sklearn.utils import shuffle
import re

STOPWORDS = set(stopwords.words("english"))
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 2000000
EMBEDDING_DIM = 300

#path = '/Users/manzhu/Desktop/PycharmTensorflow/DisasterTweets/'
#EMBEDDING_FILE = path+'GoogleNews-vectors-negative300.bin'
EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin'


df_inform = pd.read_csv('training_data_set_unduplicate.csv',sep=',')
data = shuffle(df_inform)
X_inform = data.Content
y_inform = data.Symbol
r1 = '[’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'

for text in X_inform:
    text = re.sub(r1, "",text)
    text = filter(lambda word: word not in STOPWORDS, text)
category = to_categorical(y_inform)

tokenizer = Tokenizer(num_words=2000000)
tokenizer.fit_on_texts(X_inform)

class_sequences = tokenizer.texts_to_sequences(X_inform)
class_data = pad_sequences(class_sequences, maxlen=MAX_SEQUENCE_LENGTH)

VALIDATION_SPLIT = 0.2

nb_validation_samples = int(VALIDATION_SPLIT * class_data.shape[0])
print(category.shape, class_data.shape, nb_validation_samples)
x_train = class_data[:-nb_validation_samples]
y_train = category[:-nb_validation_samples]
x_val = class_data[-nb_validation_samples:]
y_val = category[-nb_validation_samples:]



#######################################################

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

from keras.layers import Embedding
word_index = tokenizer.word_index
nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

embedding_layer = Embedding(embedding_matrix.shape[0], # or len(word_index) + 1
                            embedding_matrix.shape[1], # or EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten
from keras.layers import Dense, Input, LSTM, GRU, Embedding, Dropout, Activation, Bidirectional
from keras.optimizers import RMSprop
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(250,3,padding='valid',activation='relu',strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(250))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(4))
    model.add(Activation('sigmoid'))
    model.compile(loss='categorical_crossentropy',optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-6), metrics=['categorical_accuracy'])
    model.summary()

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=70)
#score = model.evaluate(x_val, y_val, verbose=0)
#########scores
y_pred = model.predict(x_val,batch_size=100,verbose=1)
Y_pred = np.argmax(y_pred,axis=1)
Y_val = np.argmax(y_val,axis = 1)
print(metrics.classification_report(Y_val, Y_pred))
#########

import matplotlib.pyplot as plt
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

