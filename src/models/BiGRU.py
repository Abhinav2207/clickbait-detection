from keras.models import Sequential
from keras.layers import Convolution1D,Dropout, GRU, Bidirectional, Dense, Embedding
from keras.regularizers import l2

def BiGRU(vocabulary_size, embedding_dimension, input_length, embedding_weights=None):
    
    model = Sequential()
    if embedding_weights is None:
        model.add(Embedding(vocabulary_size, embedding_dimension, input_length=input_length, trainable=False))
    else:
        model.add(Embedding(vocabulary_size, embedding_dimension, input_length=input_length, weights=[embedding_weights], trainable=False))

    model.add(Dense(100, activation='relu'))  
    model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(128)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    return model
