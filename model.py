# Importing the libraries
import warnings
warnings.filterwarnings('ignore') # to ignore deprecated functions
from keras.models import Sequential
from keras.layers.core import Dense
from tensorflow.keras.optimizers import Adam
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
import pickle


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
