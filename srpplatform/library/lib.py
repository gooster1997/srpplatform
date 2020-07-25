import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor as rf

if __name__ == '__main__':
    
    # Load data
    df = pd.read_csv('dataset.csv')
    
    X = df[['M', 'CTA', 'PC']]
    Y = df['Mw']
    
    # Train the random forest model
    rfr = rf(max_depth=6, max_features='sqrt', min_samples_split=2, n_estimators=300)
    rfr.fit(X, Y)

    # Generate the combinatorial condition pool
    conditions = [[M, CTA, PC] for M in np.arange(0.5, 1.0, step = 0.05)
                               for CTA in np.arange(1, 40, step = 1)
                               for PC in np.arange(0, 4, step = 0.1)]
    
    M = np.transpose(conditions)[0]
    CTA = np.transpose(conditions)[1]
    PC = np.transpose(conditions)[2]
    
    # Predict the corresponding values of the condition pool
    preds = rfr.predict(conditions)
    
    # Generate the conditions vs. predicted results into excel files
    dataframe = pd.DataFrame({'M': M, 'CTA': CTA, 'PC': PC, 'Mw': preds})
    dataframe.to_csv('library.csv', index = False, sep = ',')