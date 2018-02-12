#import pickle
import pandas as pd
import os
from time import gmtime, strftime

from bottle import route, run, request, static_file, post, get
from sklearn import datasets, model_selection, naive_bayes, metrics

#from main import main

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

@route('/driver')
def driver():
    return static_file('./driver.html', '.')

@route('/')
def home():
    return static_file('./static/index.html', '.')

@route('/<filename:path>')
def send_static(filename):
    return static_file(filename, root='static')

ModelParameters = {'model': ''}

basePath = os.path.abspath(os.path.dirname(__file__))
datasetName = "./static/data/all.csv"

dataset = pd.read_csv(os.path.join(basePath, datasetName))
dataset['select_test'] = False

# dataset['_row_id'] = dataset['PATIENT_ID'].tolist()
# dataset.drop(columns=['PATIENT_ID'], inplace = True)

dataset.fillna('-', inplace = True)

columns = dataset.columns.tolist()
predictedColumn = ['INT_EVT']
Tests = []

features = {}
for c in columns:
    features[c] = True
features[predictedColumn[0]] = False
features['PATIENT_ID'] = False
features['select_test'] = False

last_performance = {}
last_test_result = {}
performances = []

def defineFeatures(cols):
    global features
    for c in cols:
        features[c['name']] = c['isActive']
    return

@post('/api/data/reset')
def resetData():
    global dataset, columns, selectedColumns, Tests, predictedColumn
    dataset = pd.read_csv(os.path.join(basePath, datasetName))
    dataset['select_test'] = False
    # dataset['PATIENT_ID'] = range(dataset['select_test'].size)
    columns = dataset.columns.tolist()
    features = {}
    last_performance = {}
    performances = []
    for c in columns:
        features[c] = True
    features['PATIENT_ID'] = False
    features['select_test'] = False
    predictedColumn = []
    Tests = []
    return {'action': 'data/reset', 'msg': 'ok'}

@get('/api/data/list')
def listData():
    return {'action': 'data/list', 'msg': 'ok', 'result': dataset.to_dict('record')}

@post('/api/data/column')
def columnData():
    # params: {name: string}
    name = request.json['name']
    column = dataset.loc[:, name]
    return {'action': 'data/list', 'msg': 'ok', 'result': {'name': name, 'column': column.tolist()}}

@post('/api/data/row')
def rowData():
    # expects {row_index: string}
    # outputs a df, first row is column names, second row is one patient's record.

    row_ix = request.json['row_index']
    rows = dataset.iloc[[row_ix]] # slice by index

    # print '=============\n' + str(row_ix)
    # print dataset.iloc[[row_ix]].to_json(orient = 'records')

    return {'action': 'data/row', 'msg': 'ok', 'result': rows.to_json(orient = 'records')}

# @post('/api/class/set')
# def setClass():
#     # expects {name:string}
#     global predictedColumn
#     # put existing class back to features
#     if len(predictedColumn) > 0:
#         defineFeatures([{'name': predictedColumn[0], 'isActive': True}])
#     # set new class column
#     name = request.json['name']
#     predictedColumn = [name]
#     # remove it from features
#     defineFeatures([{'name': name, 'isActive': False}])

#     logging.debug("@setClass")
#     logging.debug(features)
#     logging.debug(predictedColumn)

#     return {'action': 'class/set', 'msg': 'ok', 'result': predictedColumn}

# @get('/api/class/list')
# def listClass():
#     global features
#     return {'action': 'features/list', 'msg': 'ok', 'result': predictedColumn}

@post('/api/features/set')
def setFeatures():
    # expects *array* of {name:string, isActive:boolean}
    defineFeatures(request.json)
    return {'action': 'features/set', 'msg': 'ok', 'result': filter(lambda c: features[c], columns)}


@get('/api/features/list')
def listFeatures():
    global features
    return {'action': 'features/list', 'msg': 'ok', 'result': filter(lambda c: features[c], columns)}


@post('/api/tests/set')
def setTests():
    # expects a row object {index:number}
    # flip its select_test status
    global Tests
    row_index = request.json['index']
    status = dataset.loc[row_index,'select_test']
    new_status = not status 
    dataset.loc[row_index,'select_test'] = new_status
    return {'action': 'tests/set', 'msg': 'ok', 'result': {'index': row_index, 'select_test': new_status}}

@post('/api/tests/lookup')
def lookupTests():
    pass


@post('/api/tests/filter')
def filterTests():
    # expects {value: "True" | "False"}
    value = request.json['value'] == "True" # convert str to bool
    data = dataset.loc[dataset['select_test'] == value]
    return {'action': 'tests/list', 'msg': 'ok', 'result': data.to_dict('record')}

# incomplete: this function stores the parameters for a model (for later use)


@post('/api/model/set')
def setModel():
    # expects object with model parameters
    global ModelParameters
    ModelParameters = request.json['parameters']
    return {'action': 'model/set', 'msg': 'ok', 'result': ModelParameters}

# inclomplete: this returns the current parameters


@get('/api/model/get')
def getModel():
    return {'action': 'model/get', 'msg': 'ok', 'result': ModelParameters}

# incomplete: this function "call main.py" with the model parameters and using the data set


@get('/api/model/train')
def trainModel():
    global last_performance
    global last_test_result
    global performances

    logging.debug("starts training for %s" % predictedColumn[0])

    active_features = [x[0] for x in list(features.items()) if x[1] == True]
    testResult, coef, mse, var_score = main(dataset, active_features,predictedColumn[0], ModelParameters)

    last_performance = {'time': strftime("%Y-%m-%d %H:%M:%S", gmtime()),
                        "prediction": predictedColumn[0],
                        "active features": active_features,
                        "algorithm": ModelParameters.get('model'),
                        "parameters": ModelParameters.get('parameters'),
                        'coefficients': zip(active_features, coef),
                        'mean squared error': mse,
                        'variance score': var_score}
    last_test_result = testResult
    # stack prev model to performances
    performances.append(last_performance)

    return {'action': 'model/train', 'msg': 'ok', 'result': last_performance}


@get('/api/model/log')
def getLog():
    return {'action': 'model/get', 'msg': 'ok', 'result': performances}

run(host='0.0.0.0', port=8090, quiet=False, debug=True)
