import numpy as np
from components import *

class aerodynamics:
    def __init__(self,Aref, mach, gamma = 1.4):
        self.mach = mach
        self.gamma = gamma
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
        if body.area_l != body.area_0:
            X_n = (body.length*body.area_l-body.volume)/(body.area_l-body.area_0)
        else:
            X_n = 0
        return X_n

    def C_na_fins(self, fin, tube, alpha):
        # function to find normal force coefficient of fins
        beta = np.sqrt(np.abs(1-self.mach**2))
        if self.mach <1:
            c_na_single = (2*np.pi*(fin.span**2)/self.Aref)/(1+np.sqrt(1+(beta*(fin.span**2)/(fin.area*np.cos(fin.Lc)))**2))
        else:
            k1 = 2/beta
            k2 = ((self.gamma+1)*self.mach**4-4*beta**2)/(4*beta**4)
            k3 = ((self.gamma+1)*self.mach**8+(2*self.gamma**2-7*self.gamma-5)*self.mach**6+10*(self.gamma+1)*self.mach**4+8)/(6*beta**7)
            c_na_single = (fin.area/self.Aref)*(k1+k2*alpha+k3*alpha**2)
        #for 4 fins
        c_na = 2*c_na_single
        # fin body interference
        kt = 1 + tube.radius/(tube.span+tube.radius)
        c_na = kt*c_na
        return c_na

    

    