#app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from models import dbMgr

# Flask Class 객체를 선언한다 . 
app = Flask(__name__)

# Main Page
@app.route('/') # Local Host 
def index():
        return render_template("index.html")

# Data Model Page
@app.route('/data_model_form')
def data_model():
        return render_template('data_model_form.html')

# Data Predict & Insert Data Base
@app.route('/modeling',methods=['POST'])
def predict_data():
        data1 = request.form['data1']
        data2 = request.form['data2']
        data3 = request.form['data3']
        data4 = request.form['data4']
        data5 = request.form['data5']
        data6 = request.form['data6']        
        # Input Data 
        data = [[data1,data2,data3,data4,data5,data6]]

        # Predict Y 
        result = dbMgr.modeling()
        data = pd.DataFrame(data=data)
        predict = result.predict(data)

        return render_template('data_model_form.html', predict=predict.round(3))

# Finished Code 
if __name__ == '__main__' :
    app. debug = True
    app.run()