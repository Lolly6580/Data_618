import numpy as np
from pykalman import KalmanFilter

 
#initialize variables and schedule functions
def initialize(context): 
    
    context.ewa = sid(14516)
    context.ewc = sid(14517)
    
    set_slippage(slippage.FixedSlippage(spread=0))
    set_commission(commission.PerShare(cost=0))
    
    schedule_function(rebalance, date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(minutes = 15))

           
def rebalance(context, data):     
    
    x = np.asarray([data.current(context.ewa, 'price'), 1.0])
    y = data.current(context.ewc, 'price')
    
    delta = 0.0001
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.expand_dims(np.vstack([[x], [np.ones(x.shape[0])]]).T, axis=1)
    
    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, 
                  initial_state_mean=[0,0],  
                  initial_state_covariance=np.ones((2, 2)),  
                  transition_matrices=np.eye(2),  
                  observation_matrices=obs_mat,  
                  observation_covariance=2,  
                  transition_covariance=trans_cov)
    
    state_means = kf.filter(y)[0]
    
    curx = data.current(context.ewa, 'price')
    cury = data.current(context.ewc, 'price')
    
    sm = state_means[0,-1]
    est = (cury - (sm * curx))
    threshold = est - sm

    if threshold <= 0.095:  
        order_target_percent(context.ewc, est)
        order_target_percent(context.ewa, -1 * est)
    else:
        order_target_percent(context.ewa, est)
        order_target_percent(context.ewc, -1 * est)
        