from flask import Flask, render_template, request
import numpy as np
import pickle
# import joblib
app = Flask(__name__)
filename = 'file_breastCancer.pkl'
model = pickle.load(open(filename, 'rb'))    # load the model


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])  # The user input is processed here
def predict():
    uniformity_cell_size = request.form['uniformity_cell_size']
    uniformity_cell_shape = request.form['uniformity_cell_shape']
    bare_nuclei = request.form['bare_nuclei']
    bland_chromatin = request.form['bland_chromatin']
    pred = model.predict(
        np.array([[uniformity_cell_size, uniformity_cell_shape, bare_nuclei, bland_chromatin,]]))
    # print(pred)
    return render_template('index.html', predict=str(pred))


if __name__ == '__main__':
    app.run(debug=True)
