from flask import Flask,render_template, request
from models import dbMgr
import pandas as pd


app=Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data1')
def sub_page():
    return render_template('sub_page.html')

@app.route('/modeling_page')
def model_page():
    return render_template('modeling_page.html')

@app.route('/modeling',methods=['POST'])
def modeling():
    a = request.form['a']
    b = request.form['b']
    c = request.form['c']
    e = request.form['e']
    f = request.form['f']
    g = request.form['g']
    h = request.form['h']
    
    data1 = [[a,b,c,e,f,g,h]]

    estimator = dbMgr.modeling_RF()
    df3 = pd.DataFrame(data=data1)
    predict = estimator.predict(df3)
    print(predict)
    return render_template('modeling_page.html',predict=predict)

if __name__=='__main__':
    app.debug=True
    app.run()