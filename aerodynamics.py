import numpy as np
from components.py import *

class aerodynamics:
    def __init__(self,Aref):
        self.Aref = Aref

    def C_na(self, body, alpha):
        # function to find normal force coefficient
        if alpha != 0:
            c_na = 2*(body.area_l-body.area_0)*np.sin(alpha)/(alpha*self.Aref)
        else:
            c_na = 2*(body.area_l-body.area_0)/self.Aref
        return c_na
    
    def C_ma(self, body, alpha):
        # function to find moment coeffecient for a von karman nose cone
        if alpha != 0:
            c_ma = (body.length*body.area_l-body.volume)*np.sin(alpha)/(alpha*self.Aref*body.radius)
        else:
            c_ma = (body.length*body.area_l-body.volume)/(self.Aref*body.radius)
        return c_ma

    def C_lift(self, body, alpha):
        #  function to find lift coeffecient
        K = 1.1
        c_lift = K*2*body.radius*body.length*(np.sin(alpha)**2)/self.Aref
        return c_lift

    def X_lift(self, body):
        # function to find lift location
        if body.cl is not None:
            return body.cl
        else:
            x = np.linspace(0,body.length)
            y = x*2*body.r(x)
            integral = np.trapezoid(y,x)
            return integral/(body.length*2*body.radius)

    def X_n(self, body):
        # function to find centre of pressure
        X_n = (body.length*body.area_l-body.volume)/(body.area_l-body.area_0)
        return X_n