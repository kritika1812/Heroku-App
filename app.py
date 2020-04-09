import numpy as np
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.cluster import KMeans
import pickle

app = Flask(__name__)

# Load the model
kmeans = pickle.load(open('model.pkl','rb'))

@app.route('/<latitude>,<longitude>')
def predict(latitude,longitude):
    tempdf = pd.read_csv('results_file.csv')
    cluster = kmeans.predict(np.array([longitude,latitude]).reshape(1,-1))[0]
    output = tempdf[tempdf['cluster']==cluster].iloc[0:5][["property_name","address", "hotel_star_rating" , "hotel_category"]]
    jsout = output.to_json(orient='values')
    return jsout

if __name__ == '__main__':
    app.run(port=5000, debug=True)
