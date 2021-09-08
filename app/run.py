import sys

from flask import Flask, request, render_template
from joblib import load
import pandas as pd

sys.path.append('../util')
from custom_transformer import NumericalFeatures, CategoricalFeatures

app = Flask(__name__)

model = load('../model/loan_data.pkl')


@app.route('/')
@app.route('/index.html')
def display():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    test_data = list(request.form.values())
    test_data = [[convert_to_float(s) for s in test_data]]
    test_data = pd.DataFrame(test_data)
    result = model.predict(pd.DataFrame(test_data))
    # return render_template("index.html", tables=[test_data.to_html(classes='data')], titles=test_data.columns.values)
    return render_template("index.html", result=result)


def convert_to_float(s):
    try:
        return float(s)
    except ValueError:
        return s


def main():
    app.run(port=3201, debug=True)


if __name__ == '__main__':
    main()
