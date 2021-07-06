# -*- coding: utf-8 -*-
import re
import string
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.layers import Embedding, GRU, Dense
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt

def remove_punctuation(text):
        no_punc = [words for words in text if words not in string.punctuation]
        word_wo_punc = "".join(no_punc)
        return word_wo_punc
def remove_numeric(corpus):
        output = "".join(words for words in corpus if not words.isdigit())
        return output
def preprocess(input, tokenizer, input2 = None ):
        
        input_tokens =  tokenizer.texts_to_sequences(input)
        num_tokens = [len(tokens) for tokens in input_tokens]
        if input2 is not None:
            input2_tokens =  tokenizer.texts_to_sequences(input2)
            num_tokens = [len(tokens) for tokens in input_tokens + input2_tokens]
        num_tokens = np.array(num_tokens)
        num_tokens
        np.mean(num_tokens)
        np.max(num_tokens)
        max_tokens = np.mean(num_tokens) + (2*np.std(num_tokens))
        max_tokens = int(max_tokens)
        np.sum(num_tokens < max_tokens) / len(num_tokens)
        input_pad = pad_sequences(input_tokens, maxlen = max_tokens) 
        if input2 is not None:
            input2_pad = pad_sequences(input2_tokens, maxlen = max_tokens)
            return input_pad, input2_pad, max_tokens
        return input_pad
def tokens_to_string(tokens, idx):
        inverse_map = dict(zip(idx.values(), idx.keys()))
        words = [inverse_map[token] for token in tokens if token != 0]
        text = " ".join(words) # Kelimeler aralarında boşluk bırakılarak ard arda yazılacaktır.
        return text
def create_model(input_dim, output_dim, input_length, name):
        model = Sequential()
        model.add(Embedding(input_dim=input_dim, output_dim=output_dim , input_length=input_length, name=name))
        model.add(GRU(units = 16, return_sequences = True))
        model.add(GRU(units = 8, return_sequences = True))
        model.add(GRU(units = 4))
        model.add(Dense(1, activation = "sigmoid"))
        return model
def data_loader():
    df = pd.read_csv("turkish_movie_sentiment_dataset.csv")
    comments = lambda x : x[23:-24]
    df["comment"] = df["comment"].apply(comments)
    floatize = lambda x : float(x[0:-2])
    df["point"] = df["point"].apply(floatize)
    df["point"].value_counts()
    #df.drop(df[df["point"] == 3].index, inplace = True)
    df["point"] = df["point"].replace(1, 0)
    df["point"] = df["point"].replace(2, 0)
    df["point"] = df["point"].replace(3, 0.5)
    df["point"] = df["point"].replace(4, 1)
    df["point"] = df["point"].replace(5, 1)
    df["point"].value_counts()
    df.reset_index(inplace = True)
    df.drop("index", axis = 1, inplace = True)
    df.head()
    df["comment"] = df["comment"].apply(lambda x: x.lower())
    df.head()
    

    df["comment"] = df["comment"].apply(lambda x: remove_punctuation(x))
    df["comment"] = df["comment"].apply(lambda x: x.replace("\r", " "))
    df["comment"] = df["comment"].apply(lambda x: x.replace("\n", " "))

    df.head()
    

    df["comment"] = df["comment"].apply(lambda x: remove_numeric(x)) 
    df.head()
    target = df["point"].values.tolist()
    data = df["comment"].values.tolist()
    return data, target

def main():
    
    data, target = data_loader()

    cutoff = int(len(data)*0.80)

    X_train, X_test = data[:cutoff], data[cutoff:]
    y_train, y_test = target[:cutoff], target[cutoff:]

    num_words = 10000
    tokenizer = Tokenizer(num_words = num_words)
    tokenizer.fit_on_texts(data)
    # tokenizer.word_index
    X_train_tokens = tokenizer.texts_to_sequences(X_train)
    X_test_tokens = tokenizer.texts_to_sequences(X_test)

    
    #print([X_train[1000]])
    #print(X_train_tokens[1000])
    """
    num_tokens = [len(tokens) for tokens in X_train_tokens + X_test_tokens]
    num_tokens = np.array(num_tokens)
    num_tokens
    np.mean(num_tokens)
    np.max(num_tokens)
    max_tokens = np.mean(num_tokens) + (2*np.std(num_tokens))
    max_tokens = int(max_tokens)
    max_tokens
    np.sum(num_tokens < max_tokens) / len(num_tokens)
    X_train_pad = pad_sequences(X_train_tokens, maxlen = max_tokens) 
    X_test_pad = pad_sequences(X_test_tokens, maxlen = max_tokens)"""

    X_train_pad, X_test_pad, max_tokens = preprocess(X_train,tokenizer, X_test)

    print(X_train_pad.shape)
    print(X_test_pad.shape)
    np.array(X_train_tokens[800])
    X_train_pad[2000]
    idx = tokenizer.word_index

    
    tokens_to_string(X_train_tokens[350], idx)
    embedding_size = 50

    X_train_pad = np.array(X_train_pad)
    y_train = np.array(y_train)

    X_test_pad = np.array(X_test_pad)
    y_test = np.array(y_test)

    

    #create model
    optimizer = Adam(lr = 1e-3)
    model = create_model(num_words,embedding_size, max_tokens, "embedding_layer")
    model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
    model.summary()

    #hyperparameters
    batch_size=256
    epochs=20
    """train and save model"""
    history = model.fit(X_train_pad, y_train, batch_size = batch_size, epochs = epochs, validation_data=(X_test_pad,y_test))
    #model.save("./Models")
    """"""
    #y_pred = model.predict(X_test_pad[0:1000])
    #y_pred = y_pred.T[0]
    #print(y_pred)
    # # y_pred
    # cls_pred = np.array([1.0 if p > 0.5 else 0.0 for p in y_pred])
    # cls_true = np.array(y_test[0:1000])
    # incorrect = np.where(cls_pred != cls_true)
    # incorrect = incorrect[0]
    # # incorrect
    # len(incorrect)
    # idx = incorrect[10]
    # X_test[idx]
    # y_pred[idx]
    # cls_true[idx]

    # Plot training & validation accuracy values
    """plt.figure(figsize=(14,3))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()"""

if __name__ == "__main__":
    main()



