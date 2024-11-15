from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl",'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('page.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('Location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    print(f"Location: {location}, BHK: {bhk}, Bathrooms: {bath}, Sqft: {sqft}")

    if location is None or bhk is None or bath is None or sqft is None:
        return "Please fill in all fields"
    bhk = int(bhk)
    bath = int(bath)
    sqft = float(sqft)
    input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    input_data['location'] = input_data['location'].fillna('Unknown')
    prediction = pipe.predict(input_data)[0] * 1e5
    updated_prediction = round(prediction, 2)
    formatted_prediction = "{:,.2f}".format(updated_prediction)
    return f": {formatted_prediction}"


if __name__ == "__main__":
    app.run(debug=True, port=5001)