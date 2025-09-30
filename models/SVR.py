from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import shap


def SV_Regression(train, valid, test, params):
    
    ker = params[0]
    eps = params[1]
    con = params[2]
    deg = params[3]
    gam = params[4]
    shapley_num = params[5]
    get_effect = params[6]
    
    #Split the data sets into x and y here as specified in the original code
    train_x, train_y = train.iloc[:,:-1], train.iloc[:,-1]
    if valid.shape[0] != 0:
        valid_x, valid_y = valid.iloc[:,:-1], valid.iloc[:,-1]
    test_x, test_y = test.iloc[:,:-1], test.iloc[:,-1]
    
    #Develop & evaluate a model here as specified in the original code
    svr = SVR(kernel=ker, epsilon=eps, C=con, degree=deg, gamma=gam)
    svr.fit(train_x, train_y)
    
    predicted = svr.predict(test_x)
    if valid.shape[0] != 0:
        predicted_valid = svr.predict(valid_x)
    else:
        predicted_valid = []
    predicted_train = svr.predict(train_x)

    ## Calculate the metrics
    actual_test = test_y.values.tolist()
    mse = mean_squared_error(actual_test, predicted)
    r = pearsonr(actual_test, predicted)[0]
    
    if get_effect == True:
        explainer = shap.KernelExplainer(svr.predict,shap.sample(train_x, shapley_num))
        effect = abs(explainer.shap_values(shap.sample(test_x,shapley_num))).sum(axis=0)
    else:
        effect = pd.DataFrame()
    
    return r, mse, pd.DataFrame(effect).T, predicted, predicted_valid, predicted_train
