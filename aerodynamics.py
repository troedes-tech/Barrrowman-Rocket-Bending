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

    def cd_blunt(self, mach):
        # drag coefficient of a blunt cylinder
        if mach < 1:
            cd = 0.85*(1+(mach**2)/4+(mach**4)/40)
        else:
            cd = 0.85*(1.84-0.76/(mach**2)+0.166/(mach**4)+0.035/(mach**6))
        return cd

    def cd_vk(self, mach, nosecone):
        # function for finding the axial drag coefficient of a von karman nosecone
        c0 = self.cd_blunt(mach)
        fn = 0.5*nosecone.length/nosecone.radius # fineness
        exp = np.log(fn+1)/np.log(4)
        data_min = nosecone.cd_data_f3[0][0]
        data_max = nosecone.cd_data_f3[0][-1]

        # transonic/supersonic region
        if mach > data_min and mach < data_max:
            c3 = np.interp(mach,nosecone.cd_data_f3[0], nosecone.cd_data_f3[1])
            cd = c0*np.pow(c3/c0,exp)

        # subsonic
        elif mach < data_min:
            cd = 0

        # supersonic extrapolation
        else:
            # issue with this but assume constant for Mach >3
            cd = c0*np.pow(nosecone.cd_data_f3[1][-1]/c0,exp)

        return cd

    def cd_ogive(self, mach, nosecone):
        # function for finding the axial drag coefficient of an ogive nosecone
        
        if mach > 1:
            cd = (0.72*(nosecone.shape_factor-0.5)**2+0.82)*self.cd_cone(mach, nosecone)
        else:
            cd_m0 = 0.8*(np.sin(nosecone.shoulder_angle)**2)
            cd_m1 = (0.72*(nosecone.shape_factor-0.5)**2+0.82)*np.sin(nosecone.half_angle)
            cd_m1_slope = 4*(1-0.5*np.sin(nosecone.half_angle))/(self.gamma+1)

            b = cd_m1_slope/(cd_m1 - cd_m0)
            a = cd_m1-cd_m0

            cd = a*np.pow(mach,b)+cd_m0

        return cd

    def cd_cone(self, mach, nosecone):
        # needs to be updated for subsonic case if being used for conical nosecone
        # function is currently only used in ogive calculations
        if mach >= 1.3:
            cd = 2.1*(np.sin(nosecone.half_angle)**2)+0.5*np.sin(nosecone.half_angle)/np.sqrt((mach**2)-1)

        else: # 1 < mach <1.3
            cd1 = np.sin(nosecone.half_angle)  # mach 1
            cd2 = 2.1*(np.sin(nosecone.half_angle)**2)+0.5*np.sin(nosecone.half_angle)/np.sqrt((1.3**2)-1)  # mach 1.3
            cd1_slope = 4*(1-0.5*np.sin(nosecone.half_angle))/(self.gamma+1)

            # quadratic interpolation
            a = (-10/9)*(10*cd1-10*cd2+3*cd1_slope)
            b = (200/9)*(cd1-cd2)+(23*cd1_slope/3)
            c = (1/9)*(-91*cd1+100*cd2-39*cd1_slope)

            cd = a*mach**2 + b*mach +c

        return cd

    







        
         



        