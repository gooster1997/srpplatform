import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

if __name__ =='__main__':
    
    # Load data
    df1 = pd.read_csv('experiment.csv')
    df2 = pd.read_csv('literature.csv')
    columns = ['M', 'M/CTA', 'PC/M', 'time', 'Mn', 'MWD', 'others']
    experiment = df1[columns]
    literature = df2[columns]
    
    # Over sampling
    dataset_oversampling = pd.DataFrame()
    
    for name, group in literature.groupby('others'):
        temp = pd.concat([experiment, group])
        temp_dummies = pd.get_dummies(temp)
        smo = SMOTE()
        X_smo, y_smo = smo.fit_sample(temp_dummies, temp_dummies['others_A'])
        X_smo = pd.DataFrame(X_smo, columns=temp_dummies.columns)
        dataset_oversampling = pd.concat([dataset_oversampling, X_smo])
        
    dataset_oversampling = dataset_oversampling.fillna(0)
    dataset_oversampling = dataset_oversampling.drop_duplicates()
    dataset_oversampling.to_csv('dataset oversampling.csv', index=False, sep=',')
    
    # Read the modified dataset
    y = dataset_oversampling['Mn']
    X = dataset_oversampling.drop(['Mn', 'MWD'], axis=1)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)
    
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.fit_transform(X_train)
    
    # Select the best model parameters through GridSearchCV
    parameters = {
            'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,0.1,1, 5,10,20,100]
            }
    las = Lasso()
    gsearch = GridSearchCV(	estimator = las, 
							param_grid = parameters, 
							scoring='neg_mean_squared_error', 
							n_jobs=4, iid=False, cv=5)
    
    gsearch.fit(X_train,y_train)
    
    print("Best parameters selected: %s"%gsearch.best_params_)
    
    # Print the R2 and RMSE values of the predicative model & generate the experimental vs. predicted values into excel files
    preds = gsearch.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, preds))
    print("Training data RMSE: %f" % (rmse))
    r2 = r2_score(y_train, preds)
    print("Training data R2: %f" % (r2))
    dataframe = pd.DataFrame({'test':y_train, 'predict':preds})
    dataframe.to_csv('lasso-train.csv', index = False, sep = ',')
    preds = gsearch.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("Test data RMSE: %f" % (rmse))
    r2 = r2_score(y_test, preds)
    print("Test data R2: %f" % (r2))
    dataframe = pd.DataFrame({'test':y_test, 'predict':preds})
    dataframe.to_csv('lasso-test.csv', index = False, sep = ',')