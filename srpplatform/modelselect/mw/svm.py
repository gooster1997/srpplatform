import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':

	# Load data
	df = pd.read_csv('dataset.csv')

	X = df[['M', 'CTA', 'PC']]
	Y = df['Mw']
	
	# Split the data into training and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=33)

	min_max_scaler = MinMaxScaler()
	X_train = min_max_scaler.fit_transform(X_train)
	X_test = min_max_scaler.fit_transform(X_train)

	# Select the best model parameters through GridSearchCV
	parameters = [
            {'kernel':['linear'], 'C':[1,10,100,1000]},
            {'kernel':['poly'], 'C':[1], 'degree':[2,3]},
            {'kernel':['rbf'], 'C':[1,10,100,1000], 'gamma':[1,0.1,0.01,0.001]}
            ]
	clf = svm.SVR()
	gsearch = GridSearchCV(	estimator = clf, 
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
	dataframe.to_csv('svm-train.csv', index = False, sep = ',')

	preds = gsearch.predict(X_test)
	rmse = np.sqrt(mean_squared_error(y_test, preds))
	print("Test data RMSE: %f" % (rmse))
	r2 = r2_score(y_test, preds)
	print("Test data R2: %f" % (r2))
	dataframe = pd.DataFrame({'test':y_test, 'predict':preds})
	dataframe.to_csv('svm-test.csv', index = False, sep = ',')