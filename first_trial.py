import mlflow


def cal_sum(x,y):
    return (x*y)


if __name__=='__main__':
    #starting ml flow server
    with mlflow.start_run():
        x,y=10,75
        z=cal_sum(x,y)
        #track the experiments with mlflow
        mlflow.log_param('x',x)
        mlflow.log_param('y',y)
        mlflow.log_metric('z',z)
    # print(f'the sum of value is: {z}')