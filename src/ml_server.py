import pandas as pd


import os
import pickle

from collections import namedtuple
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for, session
from flask import render_template, redirect, current_app, send_file

from wtforms.validators import DataRequired
from wtforms import StringField, SubmitField, FileField


import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar


class RandomForestMSE:
    def __init__(self, n_estimators, max_depth=None, feature_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        
        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        if self.max_depth == -1:
            self.max_depth = None
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
            
        y_val : numpy ndarray
            Array of size n_val_objects           
        """
        if self.feature_subsample_size is None or self.feature_subsample_size == -1:
            self.feature_subsample_size = int(X.shape[1] / 3) + 1 #Так как задача регрессии
        
        self.models = []
        
        loss = []
        def RMSE(y_true, y_pred):
            return np.sqrt(((y_true - y_pred) ** 2).mean())
        
        class Loss:
            loss = 0
            it = 0
        
        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.feature_subsample_size, **self.trees_parameters)
            index = np.random.randint(len(X), size=len(X))
            tree.fit(X[index], y[index])
            self.models.append(tree)
            loss_now = Loss()
            loss_now.loss = RMSE(y, self.predict_loss(X, i+1))
            loss_now.it = i+1
            loss.append(loss_now)
        return loss
        
    def predict_loss(self, X, n):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        answer = np.zeros(len(X))
        for i in range(n):
            answer += self.models[i].predict(X)
        return answer / self.n_estimators
        
    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        answer = np.zeros(len(X))
        for i in range(self.n_estimators):
            answer += self.models[i].predict(X)
        return answer / self.n_estimators


class GradientBoostingMSE:
    def __init__(self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        
        learning_rate : float
            Use learning_rate * gamma instead of gamma
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        
        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        if self.max_depth == -1:
            self.max_depth = None
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects
        """
        if self.feature_subsample_size is None or self.feature_subsample_size == -1:
            self.feature_subsample_size = int(X.shape[1] / 3) + 1#Так как задача регрессии
            
            
        self.models = []
        self.c = np.zeros(self.n_estimators)
        answer = np.zeros(len(X))

        
        def mse(a, b):
            return ((a - b) ** 2).mean()
        
        loss = []
        def RMSE(y_true, y_pred):
            return np.sqrt(((y_true - y_pred) ** 2).mean())
        
        class Loss:
            loss = 0
            it = 0
        
        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.feature_subsample_size, **self.trees_parameters)
            tree.fit(X, 2 * (answer - y))
            self.c[i] = minimize_scalar(lambda x: mse(y, answer + x*tree.predict(X))).x
            self.models.append(tree)
            answer += self.learning_rate * self.c[i] * tree.predict(X)
            loss_now = Loss()
            loss_now.loss = RMSE(y, self.predict_loss(X, i+1))
            loss_now.it = i+1
            loss.append(loss_now)
        return loss
    
    def predict_loss(self, X, n):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        answer = np.zeros(len(X))
        for i in range(n):
            answer += self.learning_rate * self.c[i] * self.models[i].predict(X)
        return answer

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        answer = np.zeros(len(X))
        for i in range(self.n_estimators):
            answer += self.learning_rate * self.c[i] * self.models[i].predict(X)
        return answer


app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
data_path = './../data'
Bootstrap(app)
messages = []
UPLOAD_FOLDER = ''
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

class Loss:
    loss = 0
    it = 0
    


class MyForm(FlaskForm):
    file = FileField()
    


@app.route('/', methods=['GET', 'POST'])
def my():
    form = MyForm()
    
    if form.validate_on_submit():
        global train
        global target
        global name
        train = pd.read_csv(form.file.data)
        target = train['target']
        train.drop(columns=['target'], inplace=True)
        name = form.file.data.filename
        
        return redirect(url_for('set_param'))
        #return render_template('param.html')
    
    return render_template('begin.html', form=form)


class ParamForm(FlaskForm):
    text = StringField('Type of alghoritm', validators=[DataRequired()])
    text2 = StringField('n_estimators', validators=[DataRequired()])
    text3 = StringField('learning_rate', validators=[DataRequired()])
    text4 = StringField('max_depth', validators=[DataRequired()])
    text5 = StringField('feature_subsample_size', validators=[DataRequired()])
    submit = SubmitField('Go (долгая загрузка == обучение)')


@app.route('/set_param', methods=['GET', 'POST'])
def set_param():
    try:
        param_form = ParamForm()
        if param_form.validate_on_submit():
            global type_alg 
            global n_estimators
            global learning_rate
            global max_depth
            global feature_subsample_size
            type_alg = int(param_form.text.data)
            n_estimators = int(param_form.text2.data)
            learning_rate = float(param_form.text3.data)
            max_depth = int(param_form.text4.data)
            feature_subsample_size = int(param_form.text5.data)
            return redirect(url_for('success_learn'))
        return render_template('from_form.html', form=param_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))



@app.route('/success_learn', methods=['GET', 'POST'])
def success_learn():
    global model 
    if type_alg:
        model = GradientBoostingMSE(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, feature_subsample_size=feature_subsample_size)
    else:
        model = RandomForestMSE(n_estimators=n_estimators, max_depth=max_depth, feature_subsample_size=feature_subsample_size)
    global loss
    loss = model.fit(train.values, target.values)
    app.logger.info(loss)
    return render_template('param.html')


@app.route('/get_param', methods=['GET', 'POST'])
def get_param():
    type_ = 'random forest'
    if type_alg:
        type_ = 'boosting'
        return render_template('get_param.html', messages={'loss': loss, 'lr':learning_rate, 'ne':n_estimators, 'md':max_depth, 
                                                           'fss':feature_subsample_size, 'type':type_, 'name':name})
    return render_template('get_param.html', messages={'loss': loss, 'lr':'-', 'ne':n_estimators, 'md':max_depth, 
                                                       'fss':feature_subsample_size, 'type':type_, 'name':name})


@app.route('/success_learn2', methods=['GET', 'POST'])
def success_learn2():
    return render_template('param.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = MyForm()
    
    if form.validate_on_submit():
        global test
        test = pd.read_csv(form.file.data)
        
        return redirect(url_for('prediction'))
    
    return render_template('predict.html', form=form)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    pred = model.predict(test)
    pred = pd.DataFrame(pred)
    pred.to_csv('answer.csv')
    return send_file('answer.csv', as_attachment=True)
    
