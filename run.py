import os

n_estimators=[10,20,30,40,50]
max_depth=[10,12,15,20]

for n in n_estimators:
    for md in max_depth:
        os.system(f'python basic_ml_model.py -n {n} -md {md}')