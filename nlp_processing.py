import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
nltk.download("punkt")
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import string
import pickle
import warnings
warnings.filterwarnings("ignore")
from imblearn.over_sampling import SMOTE # SMOTE


# path
TRAIN_DIR = '../flask_deploying/train_set.p'
TEST_DIR = '../flask_deploying/test_set.p'

df = pd.read_excel("data1_new_2.xlsx")

lemmatizer = WordNetLemmatizer()

data = df['Question']
data_X = df['Question'].values.tolist()
y = df['cluster']
data_y = df['cluster'].values.tolist()
classes= df['cluster'].drop_duplicates().values.tolist()
classes = sorted(set(classes))

def get_word(df):
  lemmatizer = WordNetLemmatizer()
  words = []
  data = df['Question']
  for i in data:
    tokens = nltk.word_tokenize(i)
    words.extend(tokens)
  words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
  words = sorted(set(words))

  return words

def get_train_test(data_X, words):
  training = []
  out_empty = [0]*len(classes)
  for idx, doc in enumerate(data_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
      bow.append(1) if word in text else bow.append(0)

    output_row = list(out_empty)
    output_row[classes.index(data_y[idx])] = 1
    training.append([bow, output_row])


  training = np.array(training, dtype= object)
  train = np.array(list(training[:,0]))
  test = np.array(list(training[:,1]))

  return train, test

def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def bag_of_words(text, vocab):
  tokens = clean_text(text)
  bow = [0] * len(vocab)
  for w in tokens:
    for idx, word in enumerate(vocab):
      if word == w:
        bow[idx] = 1
  return np.array(bow)


def main():
  df = pd.read_excel("data1_new_2.xlsx")

  lemmatizer = WordNetLemmatizer()

  data = df['Question']
  data_X = df['Question'].values.tolist()
  y = df['cluster']
  data_y = df['cluster'].values.tolist()
  classes = df['cluster'].drop_duplicates().values.tolist()
  classes = sorted(set(classes))

  words = get_word(df)
  train, test = get_train_test(data_X, words)

  sm = SMOTE(random_state=777, k_neighbors=3)
  train_sm, test_sm = sm.fit_resample(train, test)

  pickle.dump(train_sm, open(TRAIN_DIR, 'wb'))
  pickle.dump(test_sm, open(TEST_DIR, 'wb'))



if __name__ == '__main__':
    main()
