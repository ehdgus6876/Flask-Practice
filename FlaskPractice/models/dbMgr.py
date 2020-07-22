import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
def modeling_RF():
    result = None
    try:
        df1 = pd.read_csv('model_data.csv',encoding="CP949")
        print(df1.shape)

        df2 = df1.dropna()
        X = df2[['AGE','WEIGHT','RUNTIME','RUNPULSE','RSTPULSE','MAXPULSE']]
        Y = df2['OXY']

        model = Pipeline([('scaler',RobustScaler()),
                ('model',RandomForestRegressor())])
        param = {'model__max_depth':[2,3,4,5]}
        
        model_grid = GridSearchCV(model, param_grid=param,cv=3)
        model_grid.fit(X,Y)
        estimator = model_grid.best_estimator_
    except Exception as e:
        print(e)
    finally:
        pass
    return estimator