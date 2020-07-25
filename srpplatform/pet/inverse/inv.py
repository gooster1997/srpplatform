import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors

if __name__ == '__main__':
    
    # Load data
    df = pd.read_csv('dataset oversampling.csv')
    
    X = df.drop(['Mn', 'MWD'], axis=1)
    Y1 = df['Mn']
    Y2 = df['MWD']
    
    # Train the dataset with optimized models
    knn = neighbors.KNeighborsRegressor(n_neighbors=1, weights='uniform')
    rfr = rf(max_depth=6, max_features='sqrt', min_samples_split=2, n_estimators=200)
    
    min_max_scaler = MinMaxScaler()
    X_nor = min_max_scaler.fit_transform(X)
    knn.fit(X_nor, Y1)
    rfr.fit(X, Y2)
    
    # Generate the combinatorial condition pool
    conditions = [[M, M_CTA_1 * M_CTA_2, PC_M_1 * PC_M_2, 1, 0, 0, 0, 0, 0, 0, time] for M in np.arange(0.1, 8, step = 0.1)
                                                              for M_CTA_1 in np.arange(1, 9, step = 1)
                                                              for M_CTA_2 in np.logspace(1, 5, num = 5)
                                                              for PC_M_1 in np.arange(1, 9, step = 1)
                                                              for PC_M_2 in np.logspace(-1, 3, num = 5)
                                                              for time in np.arange(0.1, 4, step = 0.1)]
    
    M = np.transpose(conditions)[0]
    M_CTA = np.transpose(conditions)[1]
    PC_M = np.transpose(conditions)[2]
    time = np.transpose(conditions)[10]
    
    # Assign the target Mn
    Mn = 40000
    MWD = 1.20
    
    # Search for optimal conditions
    conditions_nor = min_max_scaler.fit_transform(X)
    Mn_preds = knn.predict(conditions_nor)
    MWD_preds = rfr.predict(conditions)
    M_opt = []
    M_CTA_opt = []
    PC_M_opt = []
    time_opt = []
    for index in range(len(conditions)):
        if (Mn - 500 < Mn_preds < Mn + 500) and (MWD - 0.05 < MWD_preds < MWD + 0.05):
            M_opt.append(M[index])
            M_CTA_opt.append(M_CTA[index])
            PC_M_opt.append(PC_M[index])
            time_opt.append(time[index])
            
    # Export the optimal conditions into excel files
    dataframe = pd.DataFrame({'M': M_opt, 'M/CTA': M_CTA_opt, 'PC/M': PC_M_opt, 'time': time_opt})
    dataframe.to_csv('optimal conditions.csv', index=False, sep=',')