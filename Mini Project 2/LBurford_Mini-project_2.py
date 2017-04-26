from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from collections import deque
import numpy as np
 

def initialize(context): 
        
    #choose stock    
    context.security = sid(698)

    #select parameters for logistic regression & neural network
    logistic = linear_model.LogisticRegression(C = 800)  
    rbm = BernoulliRBM(learning_rate = 0.001, n_iter = 15, n_components = 250)
    
    #use pipeline to add nn & logistic regression to classifier variable
    context.classifier = Pipeline(steps=[('rbm', rbm), 
                                         ('logistic', logistic)])

    #initiialize variables
    context.X = deque(maxlen=1000) 
    context.Y = deque(maxlen=1000) 
    context.prediction = 0 
    context.lookback = 5
    context.history = 180
    
    #schedule functions
    schedule_function(model, date_rules.month_end(), 
                      time_rules.market_close(minutes = 10))
    schedule_function(rebalance, date_rules.week_start(), 
                      time_rules.market_open(minutes = 1))
    schedule_function(record_vars, date_rules.every_day(),
                      time_rules.market_close())
  

def model(context, data):
    
    #get pricing and volume history for previous 6 months
    recent_prices = data.history(context.security, 'price', context.history, '1d').values 
    recent_volumes = data.history(context.security, 'volume', context.history, '1d').values
    
    #get changes in pricing and volume
    price_changes = np.diff(recent_prices).tolist() 
    volume_changes = np.diff(recent_volumes).tolist()

    #assign values to X & Y for model fitting
    for i in range(context.history - context.lookback - 1):
        context.X.append(price_changes[i:i+context.lookback] + volume_changes[i:i+context.lookback]) 
        context.Y.append(price_changes[i+context.lookback]) 
    
    #fit model if enough data is collected
    if len(context.Y) > 120:
        context.classifier.fit(context.X, context.Y)     
    
    
def rebalance(context, data):
    
    #check to see if model was built
    if context.classifier:
        
        #get pricing and volume history for previous 5 days
        recent_prices = data.history(context.security, 'price', context.lookback+1, '1d').values 
        recent_volumes = data.history(context.security, 'volume', context.lookback+1, '1d').values
        
        #get changes in pricing and volume
        price_changes = np.diff(recent_prices).tolist()
        volume_changes = np.diff(recent_volumes).tolist()
        
        #predict values using the model if enough data is collected
        if len(context.Y) > 120:
            context.prediction = context.classifier.predict(price_changes + volume_changes) 
                   
            #if prediction is greater than 0 order stock and if not sell
            if context.prediction > 0:
                order_target_percent(context.security, 1.0)
            else:
                order_target_percent(context.security, -1.0) 
            
def record_vars(context, data):
    
    #record prediction value
    record(prediction=int(context.prediction))  


