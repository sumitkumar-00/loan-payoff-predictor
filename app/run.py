import sys
import os
from pathlib import Path

from flask import Flask, request, render_template
from joblib import load
import pandas as pd

# Add util directory to path
curr_dir = sys.path[0]
parent_dir = Path(curr_dir).parents[0]
dir = os.path.join(parent_dir, 'util')
sys.path.append(dir)

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
    if result == 1:
        string = "Loan is not expected to be paid in full"
    else:
        string = "Loan is expected to be paid off"

    # return render_template("index.html", tables=[test_data.to_html(classes='data')], titles=test_data.columns.values)
    return render_template("index.html", result= "Prediction: " + string)


def convert_to_float(s):
    try:
        return float(s)
    except ValueError:
        return s


def main():
    app.run(port=3201, debug=True)


if __name__ == '__main__':
    main()
