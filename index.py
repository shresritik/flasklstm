from sklearn.preprocessing import MinMaxScaler
from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import tensorflow
import ast
app = Flask(__name__)
new_model = tensorflow.keras.models.load_model('model.h5')


@app.route('/api/response', methods=['POST'])
def post_response():
    test_res = request.json['response']
    nested_list = ast.literal_eval(test_res)
    test = np.array(nested_list)

    predicted = new_model.predict(test).flatten()

    print(predicted[0])
    return jsonify({"message": str(predicted[0])})


if (__name__ == "__main__"):
    app.run()
