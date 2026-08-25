import numpy as np
from components import *

class aerodynamics:
    def __init__(self,Aref, gamma = 1.4):
        self.gamma = gamma
        self.Aref = Aref

    # ----  NORMAL FORCE    ----

    def C_na(self, body, alpha):
        # function to find normal force coefficient
        if alpha != 0:
            c_na = 2*(body.area_l-body.area_0)*np.sin(alpha)/(alpha*self.Aref)
        else:
            c_na = 2*(body.area_l-body.area_0)/self.Aref
        return c_na

    def C_n_lift(self, body, alpha):
        #  function to find lift coeffecient
        K = 1.1
        c_n = K*2*body.radius*body.length*(np.sin(alpha)**2)/self.Aref
        return c_n

    def C_na_fins(self, fin, tube, mach, alpha):
        # function to find normal force coefficient of fins
        beta = np.sqrt(np.abs(1-mach**2))
        if mach <1:
            c_na_single = (2*np.pi*(fin.span**2)/self.Aref)/(1+np.sqrt(1+(beta*(fin.span**2)/(fin.area*np.cos(fin.Lc)))**2))
        else:
            k1 = 2/beta
            k2 = ((self.gamma+1)*mach**4-4*beta**2)/(4*beta**4)
            k3 = ((self.gamma+1)*mach**8+(2*self.gamma**2-7*self.gamma-5)*mach**6+10*(self.gamma+1)*mach**4+8)/(6*beta**7)
            c_na_single = (fin.area/self.Aref)*(k1+k2*alpha+k3*alpha**2)
        #for 4 fins
        c_na = 2*c_na_single
        # fin body interference
        kt = 1 + tube.radius/(tube.span+tube.radius)
        c_na = kt*c_na
        return c_na

    def C_n_total(self, rocket, mach, alpha):
        #function to find total coefficient of normal force
        cn_total = 0
        for part in rocket.parts:
            cn_total += self.C_na(part,alpha)*alpha + self.C_n_lift(part, alpha)
        #account for fins
        cn_total += self.C_na_fins(rocket.fins, rocket.fin_tube, mach, alpha)*alpha
        return cn_total

    # ----  MOMENTS ----

    def C_ma(self, body, alpha):
        # function to find moment coeffecient for a von karman nose cone
        if alpha != 0:
            c_ma = (body.length*body.area_l-body.volume)*np.sin(alpha)/(alpha*self.Aref*body.radius)
        else:
            c_ma = (body.length*body.area_l-body.volume)/(self.Aref*body.radius)
        return c_ma    

    # only necessary for takeoff or transonic oscillation loading
    def pitch_damp():
        return None

    def C_m_total(self, rocket, alpha):
        #function to find total coefficient of normal force
        c_ma_total = 0
        for part in rocket.parts:
            c_ma_total += self.C_ma(part,alpha)
        return c_ma_total*alpha

    # ---- ROLL DYNAMICS ----
    # only necessary for take off loading

    def equil_roll():
        return None
    
    def roll_damp():
        return None

    # ----  CENTRE OF PRESSURE  ----
    # not currently needed

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

    def cp(self,rocket, mach, alpha):
        # finds cp distance from nose cone
        x = 0
        sum_top = 0
        for part in rocket.parts:
            sum_top += (x+self.X_n(part,alpha))*self.C_na(part,alpha)*alpha + (x+self.X_lift(part,alpha))*self.C_n_lift(part,alpha)
            x += part.length
        # account for fins
        sum_top += (rocket.fins.location + rocket.fins.cp(mach)[0])*self.C_na_fins(rocket.fins, rocket.fin_tube, alpha)*alpha
        sum_bottom = self.C_n_total(rocket, alpha)
        cp = sum_top/sum_bottom
        return cp

    # ---- DRAG FORCES ----

    def skin_friction(self, rocket, roughness, velocity, kv, mach):
        # kv is kinematic viscosity
        Re = velocity*rocket.length/kv
        Re_crit = 51*(roughness/rocket.length)**-1.039
        if Re < 10**-4:
            cf = 1.48*10**-2
        elif Re < Re_crit:
            cf = 1/((1.5*np.log(Re)-5.6)**2)
        else:
            cf = 0.032*(roughness/rocket.length)**0.2

        # compressibility correction
        if mach < 1:
            cf = cf*(1-0.1*mach**2)
        elif Re < Re_crit:
            cf = cf/(1+0.15*mach**2)**0.58
        else:
            cf = cf/(1+0.18*mach**2)

        #now find the cd from cf
        cd_skin = (cf/self.Aref)*((1+1/(2*rocket.fineness_ratio))*rocket.Awet_body
                             +(1+2*rocket.fins.thickness/rocket.fins.c_hat)*8*rocket.fins.area)
        return cd_skin

    