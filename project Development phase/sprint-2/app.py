import os
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import pickle
#import inputScript
import numpy as np
app = Flask(__name__)
model = pickle.load(open('model_pkl_rf' , 'rb'))
ASSET=os.path.join('static','assets')
@app.route("/")
def index():
    filename=os.path.join(ASSET, 'work2.jpg')
    return render_template('index.html', filename=filename)

@app.route("/predict")
def predict():
    return render_template('final.html')

@app.route('/y_predict', methods=['POST'])
def y_predict():
    url = request.form['URL']

    checkprediction = [[-1,1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,0,1,1,1,1,-1,-1,-1,-1,1,1,-1,1]]
    #inputScript.main(url)
    prediction = model.predict(checkprediction)
    print("prediction", prediction)
    output = prediction[0]
    if output==1:
        pred ="You are safe! This is a legitimate Website."
    else:
        pred ="You are on the wrong site. Be cautious!"
    return render_template('final.html', prediction_text ='{}'.format(pred), url=url)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)