from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the pickled model
with open('knn_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


# Define a route for making predictions
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        open = float(request.form['open'])
        close = float(request.form['close'])

        high = float(request.form['high'])
        low = float(request.form['low'])

        open_close_diff = open - close
        high_close_diff = high - low

        input_data = np.array([open_close_diff, high_close_diff]).reshape(1, -1)

        prediction = model.predict(input_data)

        return render_template('index.html', prediction=prediction)

    # Return the prediction
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
