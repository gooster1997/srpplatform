import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from imblearn.over_sampling import SMOTE

if __name__ == '__main__':
    
    # Load data
    df = pd.read_csv('dataset.csv')

    a = df[['M', 'CTA', 'PC', 'Mw']]
    b = df['CTA_species']
   
    # Over sampling
    smo = SMOTE(random_state=42)
    a_temp, b_temp = smo.fit_sample(a, b)
    df = pd.DataFrame({'M' : np.transpose(a_temp)[0],
                       'CTA' : np.transpose(a_temp)[1],
                       'PC' : np.transpose(a_temp)[2],
                       'Mw' : np.transpose(a_temp)[3],
                       'CTA_species' : b_temp})
	
    X = df[['M', 'CTA', 'PC', 'CTA_species']]
    Y = df['Mw']
    
    # One-hot encoding
    X = pd.get_dummies(X)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=33)

	# Select the best model parameters through GridSearchCV
    parameters = {
		'n_estimators':[100,200,300,400,500,600,700,800],
		'max_depth':[1,2,3,4,5,6],
        'max_features':['sqrt'],
        'min_samples_split':[2]
    }
    rfr = rf()
    gsearch = GridSearchCV(	estimator = rfr, 
							param_grid = parameters, 
							scoring='neg_mean_squared_error', 
							n_jobs=4, iid=False, cv=5)

    gsearch.fit(X_train,y_train)

    print("Best parameters selected: %s"%gsearch.best_params_)

	# Print the R2 and RMSE values of the predicative model 
    preds = gsearch.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, preds))
    print("Training data RMSE: %f" % (rmse))
    r2 = r2_score(y_train, preds)
    print("Training data R2: %f" % (r2))

    preds = gsearch.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("Test data RMSE: %f" % (rmse))
    r2 = r2_score(y_test, preds)
    print("Test data R2: %f" % (r2))
    
    # Generate the combinatorial condition pool
    conditions = [[M, CTA, PC, 0, 1] for M in np.arange(0.5, 1.0, step = 0.05)
                                     for CTA in np.arange(1, 40, step = 1)
                                     for PC in np.arange(0, 4, step = 0.1)]
    
    M = np.transpose(conditions)[0]
    CTA = np.transpose(conditions)[1]
    PC = np.transpose(conditions)[2]
    
    # Assign the target Mw
    Mw = 2000000
    
    # Search for optimal conditions
    preds = gsearch.predict(conditions)
    M_opt = []
    CTA_opt = []
    PC_opt = []
    for index, value in enumerate(preds):
        if (Mw - 50000 < value < Mw + 50000):
            M_opt.append(M[index])
            CTA_opt.append(CTA[index])
            PC_opt.append(PC[index])
            
    # Export the optimal conditions into excel files
    dataframe = pd.DataFrame({'M': M_opt, 'CTA': CTA_opt, 'PC': PC_opt})
    dataframe.to_csv('optimal conditions.csv', index=False, sep=',')