# importing the necessary dependencies
from flask import Flask, render_template, request
from flask_cors import cross_origin
import pickle
import sklearn
import pymongo
import pandas as pd
import logging

## Creating logging config

logging.basicConfig(filename='forest_fire_log.log',
                    filemode='a',
                    level = logging.INFO,
                    format='%(asctime)s %(levelname)s-%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )

## Creating Logger Object
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

from bulk_prediction import Bulk_Predictor

app = Flask(__name__) # initializing a flask app

models = ['classification_model_saved.sav', 'regression_model_saved.sav']
classification_model = pickle.load(open(models[0], 'rb')) # loading the model file from the storage
regression_model = pickle.load(open(models[1], 'rb')) # loading the model file from the storage

@app.route('/',methods=['GET', 'POST'])  # route to display the home page
@cross_origin()
def homePage():
    logger.info('Rendering Homepage')
    return render_template("home.html")

@app.route('/prediction_choice',methods=['GET', 'POST'])  # route to display the home page
@cross_origin()
def prediction_choice():
    try:
        if request.method == 'POST':
            choice = request.form['choice']
            if choice == 'single':
                return render_template('single_prediction.html', title = 'Single Prediction')
            else:
                return render_template('bulk_prediction.html', title = 'Bulk Prediction')
        logger.info('Rendering prediction choice page')
    except:
        logger.error('Error while rendering prediction page')

@app.route('/single_prediction',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def single_prediction():
    if request.method == 'POST':
        try:
            day=int(request.form['day'])
            month=int(request.form['month'])
            year=int(request.form['year'])
            RH=float(request.form['RH'])
            Ws = float(request.form['Ws'])
            Rain = float(request.form['Rain'])
            FFMC = float(request.form['FFMC'])
            DMC = float(request.form['DMC'])
            DC = float(request.form['DC'])
            ISI = float(request.form['ISI'])
            BUI = float(request.form['BUI'])
            FWI = float(request.form['FWI'])
            logger.info('Fetching data from web')
            prediction_temp=regression_model.predict([[RH, Ws, Rain, FFMC, DMC, DC, ISI]])
            prediction_classes=classification_model.predict([[RH, Ws, Rain, FFMC, DMC, DC, ISI]])
            logger.info('Prediction Done!')
            if prediction_classes[0] == 0:
                prediction_classes = 'Not Fire'
            else:
                prediction_classes = 'Fire'

            results = [[day, month, year, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI, FWI, prediction_temp[0], prediction_classes]]
    
            return render_template('results.html', results=results)
        except Exception as e:
            logger.error('Something went wrong during single prediction')
            return 'something is wrong'
            
    else:
        return render_template('home.html')

@app.route('/bulk_prediction',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def bulk_prediction():
    if request.method == 'POST':
        try:
            client_url = request.form['client url']
            db = request.form['database name']
            collection = request.form['collection name']
            logger.info('Fetching mongodb connection data')

            bulk_predictor = Bulk_Predictor(client_url, db, collection)
            logger.info('Connection with mongodb established')
            df = bulk_predictor.predictAndFetchRecord()
            logger.info('Prediction for bulk test done')

            results = []
            for i in range(len(df)):
                results.append(list(df.iloc[i]))

            return render_template('results.html', results=results)

        except Exception as e:
            logger.error('Something went wrong during bulk prediction')
            return 'something is wrong'
    else:
        return render_template('home.html')


if __name__ == "__main__":
	app.run(debug=True) # running the app