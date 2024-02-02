import pandas as pd 
import numpy as np 
import os,sys
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVC


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,roc_auc_score
from sklearn.model_selection import train_test_split

import argparse

def get_data():
    try:
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        #reading data
        df=pd.read_csv(url,sep=';')
        return df
    except Exception as e:
        print(e)


def eval(y_true,y_pred,pred_prob):
    print(f' mae : {mean_absolute_error(y_true,y_pred)}')
    print(f' mse : {mean_squared_error(y_true,y_pred)}')
    print(f' rmse : {np.sqrt(mean_squared_error(y_true,y_pred))}')
    print(f' r2 : {(r2_score(y_true,y_pred))}')


def mainfun(n_estimators,max_depth):
    try:
        df=get_data()
        print('==========================================================')
        print()
        print(df.head())
        print('==========================================================')
        print()
        print(df.info())
        print('==========================================================')
        train,test=train_test_split(df,random_state=42)
        x_train=train.drop(columns='quality',axis=1)
        x_test=test.drop(columns='quality',axis=1)
        y_train=train[['quality']]
        y_test=test[['quality']]

        print(x_train)
        print('==========================================================')
        print()
        print(y_train)

        with mlflow.start_run():
            #linear regression trainning
            # lr=ElasticNet()
            # lr.fit(x_train,y_train)
            # y_pred=lr.predict(x_test)
            # eval(y_test,y_pred)



            #Support vector classification trainning
            # svc=SVC(kernel='rbf')
            # svc.fit(x_train,y_train)
            # y_pred_svc=svc.predict(x_test)
            # accuracy=accuracy_score(y_test, y_pred_svc)
            # roc_auc_sc=roc_auc_score(y_test, y_pred_svc)
            # print(f'accuracy score in svc is : {accuracy_score(y_test, y_pred_svc)}')
            # print(f'roc score in svc is : {roc_auc_score(y_test, y_pred_svc)}')
            # mlflow.log_param('acc_score_svc',accuracy)


            print('================================================================================================')

            #random forest trainning
            rf=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
            rf.fit(x_train,y_train)
            y_pred_rf=rf.predict(x_test)
            print(f'accuracy score in rf is : {accuracy_score(y_test, y_pred_rf)}')
            accuracy=accuracy_score(y_test, y_pred_rf)

            pred_prob=rf.predict_proba(x_test)
            roc_auc_sc=roc_auc_score(y_test,pred_prob,multi_class='ovr')
            
            mlflow.log_param('max_depth',max_depth)
            mlflow.log_param('n_estimators',n_estimators)
            mlflow.log_metric('acc_score_rf',accuracy)
            mlflow.log_metric('roc_score_rf',roc_auc_sc)

            #mlflow model logging
            mlflow.sklearn.log_model(rf,"randomforestmodel")

        if __name__=='__main__':
  
            args=argparse.ArgumentParser()
            args.add_argument("--n_est","-n",default=50,type=int)
            args.add_argument("--max_depth","-md",default=5,type=int)
            parsing=args.parse_args()
            
        
        # rf=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
        # rf.fit(x_train,y_train)
        # y_pred_rf=rf.predict(x_test)
        # print(f'accuracy score in rf is : {accuracy_score(y_test, y_pred_rf)}')


        #evalute the model
        

    except Exception as e:
        print(e)








if __name__=='__main__':
    try:
        
        args=argparse.ArgumentParser()
        args.add_argument("--n_est","-n",default=50,type=int)
        args.add_argument("--max_depth","-md",default=5,type=int)
        parsing=args.parse_args()
        mainfun(n_estimators=parsing.n_est,max_depth=parsing.max_depth)
            
    except Exception as e:
        print(e)

        