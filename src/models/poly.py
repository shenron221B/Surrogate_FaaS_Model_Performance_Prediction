from sklearn import linear_model
from fit_eval.utils import XtoRho
from sklearn.preprocessing import PolynomialFeatures

def fit_poly (model, X, Y, deg=3):
    N=len(model.serv_times)

    X2 = XtoRho(model, X)
    poly = PolynomialFeatures(degree=deg, include_bias=True)     
    X2 = poly.fit_transform(X2)
    
    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(X2, Y)

    return lambda x: regr.predict(poly.transform(XtoRho(model, x)))