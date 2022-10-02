# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime,timedelta
import warnings
warnings.filterwarnings('ignore') # to ignore deprecated functions
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle
from nltk.stem import WordNetLemmatizer
from nlp_processing import bag_of_words

# path
TRAIN_DIR = '../flask_deploying/train_set.p'
TEST_DIR = '../flask_deploying/test_set.p'
MODEL_DIR = '../flask_deploying/model.pkl'

def create_model(train_X,train_Y):
  model = Sequential()
  model.add(Dense(128, input_shape= (train_X.shape[1],), activation = "relu"))
  model.add(Dropout(0.5))

  model.add(Dense(64,activation="relu"))
  model.add(Dropout(0.5))

  model.add(Dense(train_Y.shape[1],activation = "softmax"))
  adam = Adam(learning_rate=0.01,decay = 1e-6)
  model.compile(loss='categorical_crossentropy',optimizer = adam, metrics= ["accuracy"])

  return model

def main():
    train_sm = pickle.load(open(TRAIN_DIR, 'rb'))
    test_sm = pickle.load(open(TEST_DIR, 'rb'))

    train_X, test_X, train_Y, test_y = train_test_split(train_sm, test_sm, test_size=0.2, random_state=42)

    model = create_model(train_X,train_Y)
    model.fit(train_X, train_Y, validation_data=(test_X, test_y), batch_size=64, epochs=50, verbose=1)

    pickle.dump(model, open(MODEL_DIR, 'wb'))


if __name__ == '__main__':
    main()
