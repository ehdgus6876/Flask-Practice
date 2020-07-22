import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
def modeling_RF():
    result = None
    try:
        # 오류(SettingWithCopyError 발생)
        pd.set_option('mode.chained_assignment', 'raise') # SettingWithCopyError

        # 경고(SettingWithCopyWarning 발생, 기본 값입니다)
        pd.set_option('mode.chained_assignment', 'warn') # SettingWithCopyWarning

        # 무시
        pd.set_option('mode.chained_assignment',  None) # <==== 경고를 끈다
        df1=pd.read_excel("data.xlsx",sheet_name='data')
        df2=df1.iloc[0:100]
        cond1=df2['식전혈당(공복혈당)']>=126
        df2.loc[cond1,"target"]=1
        df2.loc[~cond1,"target"]=0
        X=df2[['트리글리세라이드','연령대코드(5세단위)','수축기혈압','이완기혈압','체중(5Kg단위)','허리둘레','요단백']]
        Y=df2['target']
        
        model = Pipeline([('scaler',RobustScaler()),
                ('model',DecisionTreeClassifier())])
        param = {'model__max_depth':[2,3,4,5]}
        
        model_grid = GridSearchCV(model, param_grid=param,cv=1)
        model_grid.fit(X,Y)
        estimator = model_grid.best_estimator_
    except Exception as e:
        print(e)
    finally:
        pass
    return estimator