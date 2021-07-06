from bitirme import preprocess, data_loader
from tensorflow import keras
from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np
import sys

def predict(input):
  data, target = data_loader()
  num_words = 10000
  tokenizer = Tokenizer(num_words = num_words)
  tokenizer.fit_on_texts(data)
  loaded_model = keras.models.load_model("./Models")

  #prediction = ['dünyanın en iyi filmi harika bayıldım']
  prediction = [input]
  prediction_pad = preprocess(prediction, tokenizer)
  prediction_array = np.array(prediction_pad)
  p = loaded_model.predict(prediction_array)

  print(p[0][0])
  
if __name__ == "__main__":
    input = sys.argv[1]
    predict(input)