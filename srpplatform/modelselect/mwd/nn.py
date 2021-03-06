import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import neural_network
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':

	# Load data
	df = pd.read_csv('dataset.csv')

	X = df[['M', 'CTA', 'PC']]
	Y = df['MWD']
	
	# Split the data into training and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=33)

	min_max_scaler = MinMaxScaler()
	X_train = min_max_scaler.fit_transform(X_train)
	X_test = min_max_scaler.fit_transform(X_train)

	# Select the best model parameters through GridSearchCV
	parameters = {
        'activation':['relu'],
		'hidden_layer_sizes':[100,200],
        'alpha':[1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1,2,10,100],
        'max_iter':[100000],
	}
	nn = neural_network.MLPRegressor()
	gsearch = GridSearchCV(	estimator = nn, 
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
	dataframe.to_csv('nn-train.csv', index = False, sep = ',')

	preds = gsearch.predict(X_test)
	rmse = np.sqrt(mean_squared_error(y_test, preds))
	print("Test data RMSE: %f" % (rmse))
	r2 = r2_score(y_test, preds)
	print("Test data R2: %f" % (r2))
	dataframe = pd.DataFrame({'test':y_test, 'predict':preds})
	dataframe.to_csv('nn-test.csv', index = False, sep = ',')