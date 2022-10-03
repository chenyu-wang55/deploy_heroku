from flask import Flask, request,render_template
import pickle
import pandas as pd
import numpy as np
from nlp_processing import get_word
from nlp_processing import bag_of_words

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])


def predict():
    '''
    For rendering results on HTML GUI
    '''

    message = [str(x) for x in request.form.values()][0]

    df = pd.read_excel("data1_new_2.xlsx")
    words = get_word(df)
    classes = df['cluster'].drop_duplicates().values.tolist()
    classes = sorted(set(classes))
    model = pickle.load(open('model.pkl', 'rb'))

    bow = bag_of_words(message, words)
    result = model.predict(np.array([bow]))[0]
    thresh = 0.2
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]

    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for r in y_pred:
        return_list.append(classes[r[0]])

    tag = return_list[0]

    if tag not in df['cluster']:
        result = "Sorry I can't find the answer"
    else:
        sample = df[df['cluster'] == tag]
        test = list(sample['Answer'])
        result = test[0]

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)