import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error


class blackboxprobe:
    def __init__(self,model_path):
        self.model=joblib.load(model_path)
        self.call_count=0
        self.max_calls=100

    def predict(self,X):
        if self.call_count>=self.max_calls:
            raise Exception(f"api limit hit --your api count {self.call_count}/{self.max_calls}")
        self.call_count+=1
        return self.model.predict(X)#NOTE: EXPLAIN THIS LINE OF CODE
    
def rec_intercept(probe):
    x_zero=np.zeros((1,3))
    intercept=probe.predict(x_zero)[0]
    print(f"intercept is {intercept}")
    return intercept

def rec_coefficent(probe,intercept):
    a=np.array([1,0,0])
    coef1=probe.predict(a)[0]
    fi_coef1=coeff1-intercept
    b=np.array([0,1,0])
    coef2=probe.predict(b)[0]
    fi_coef2=coeff2-intercept
    c=np.array([0,0,1])
    coef3=probe.predict(c)[0]
    fi_coef3=coeff3-intercept
    print(f"coeff a,b,c {fi_coef1},{fi_coef2},{fi_coef3}")
    return fi_coef1,fi_coef2,fi_coef3

# def estimate_the_error(probe,dlink):


if __name__=="__main__":
    probe=blackboxprobe("concrete_strength_20260901_164159_model.joblib")
    intercept_model=rec_intercept(probe)
    coefficent_model=rec_coefficent(probe)
    print(intercept_model)
    print(coefficent_model)