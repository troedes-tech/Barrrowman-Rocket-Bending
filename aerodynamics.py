import numpy as np
from components import *

class aerodynamics:
    def __init__(self,Aref, mach, alpha, kv, temp, gamma = 1.4, R_const = 287):
        self.gamma = gamma
        self.Aref = Aref
        self.mach = mach
        self.alpha = alpha
        self.kv = kv # kinematic viscosity
        self.R_const = R_const # specific gas constant
        self.temp = temp

    # ----  NORMAL FORCE    ----

    def C_na(self, body):
        # function to find normal force coefficient
        if self.alpha != 0:
            c_na = 2*(body.area_l-body.area_0)*np.sin(self.alpha)/(self.alpha*self.Aref)
        else:
            c_na = 2*(body.area_l-body.area_0)/self.Aref
        return c_na

    def C_n_lift(self, body):
        #  function to find lift coeffecient
        K = 1.1
        c_n = K*2*body.radius*body.length*(np.sin(self.alpha)**2)/self.Aref
        return c_n

    def C_na_fins(self, fin, tube):
        # function to find normal force coefficient of fins
        beta = np.sqrt(np.abs(1-self.mach**2))
        if self.mach <1:
            c_na_single = (2*np.pi*(fin.span**2)/self.Aref)/(1+np.sqrt(1+(beta*(fin.span**2)/(fin.area*np.cos(fin.Lc)))**2))
        else:
            k1 = 2/beta
            k2 = ((self.gamma+1)*self.mach**4-4*beta**2)/(4*beta**4)
            k3 = ((self.gamma+1)*self.mach**8+(2*self.gamma**2-7*self.gamma-5)*self.mach**6+10*(self.gamma+1)*self.mach**4+8)/(6*beta**7)
            c_na_single = (fin.area/self.Aref)*(k1+k2*self.alpha+k3*self.alpha**2)
        #for 4 fins
        c_na = 2*c_na_single
        # fin body interference
        kt = 1 + tube.radius/(tube.span+tube.radius)
        c_na = kt*c_na
        return c_na

    def C_n_total(self, rocket):
        #function to find total coefficient of normal force
        cn_total = 0
        for part in rocket.parts:
            cn_total += self.C_na(part)*self.alpha + self.C_n_lift(part)
        #account for fins
        cn_total += self.C_na_fins(rocket.fins, rocket.fin_tube)*self.alpha
        return cn_total

    # ----  MOMENTS ----

    def C_ma(self, body):
        # function to find moment coeffecient for a von karman nose cone
        if self.alpha != 0:
            c_ma = (body.length*body.area_l-body.volume)*np.sin(self.alpha)/(self.alpha*self.Aref*body.radius)
        else:
            c_ma = (body.length*body.area_l-body.volume)/(self.Aref*body.radius)
        return c_ma    

    # only necessary for takeoff or transonic oscillation loading
    def pitch_damp():
        return None

    def C_m_total(self, rocket):
        #function to find total coefficient of normal force
        c_ma_total = 0
        for part in rocket.parts:
            c_ma_total += self.C_ma(part)
        return c_ma_total*self.alpha

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

    def cp_fins(self,fins):
            # function to find centre of pressure of a fin
            y = (fins.span/3)*(fins.root_chord+2*fins.tip_chord)/(fins.root_chord+fins.tip_chord)
            if self.mach < 2:
                x = (fins.sweep_length/3)*(fins.root_chord+2*fins.tip_chord)/(fins.root_chord+fins.tip_chord)+(fins.root_chord**2+fins.root_chord*fins.tip_chord+3*fins.tip_chord**2)/(6*(fins.root_chord+fins.tip_chord))
            else:
                beta = np.sqrt(np.abs(1-self.mach**2))
                x = (fins.ar*beta-0.67)/(2*fins.ar*beta-1)*fins.c_hat
            return [x,y]

    def cp(self,rocket):
        # finds cp distance from nose cone
        x = 0
        sum_top = 0
        for part in rocket.parts:
            sum_top += (x+self.X_n(part))*self.C_na(part)*self.alpha + (x+self.X_lift(part))*self.C_n_lift(part)
            x += part.length
        # account for fins
        sum_top += (rocket.fins.location + self.cp_fins(rocket.fins)[0])*self.C_na_fins(rocket.fins, rocket.fin_tube)*self.alpha
        sum_bottom = self.C_n_total(rocket)
        cp = sum_top/sum_bottom
        return cp

    # ---- DRAG FORCES ----

    def skin_friction(self, rocket):
        # kv is kinematic viscosity
        velocity = self.mach*np.sqrt(self.gamma*self.R_const*self.temp)
        Re = velocity*rocket.length/self.kv
        Re_crit = 51*(rocket.roughness/rocket.length)**-1.039
        if Re < 10**-4:
            cf = 1.48*10**-2
        elif Re < Re_crit:
            cf = 1/((1.5*np.log(Re)-5.6)**2)
        else:
            cf = 0.032*(rocket.roughness/rocket.length)**0.2

        # compressibility correction
        if self.mach < 1:
            cf = cf*(1-0.1*self.mach**2)
        elif Re < Re_crit:
            cf = cf/(1+0.15*self.mach**2)**0.58
        else:
            cf = cf/(1+0.18*self.mach**2)

        #now find the cd from cf
        cd_skin = (cf/self.Aref)*((1+1/(2*rocket.fineness_ratio))*rocket.Awet_body
                             +(1+2*rocket.fins.thickness/rocket.fins.c_hat)*8*rocket.fins.area)
        return cd_skin

    def cd_blunt(self):
        # drag coefficient of a blunt cylinder
        if self.mach < 1:
            cd = 0.85*(1+(self.mach**2)/4+(self.mach**4)/40)
        else:
            cd = 0.85*(1.84-0.76/(self.mach**2)+0.166/(self.mach**4)+0.035/(self.mach**6))
        return cd

    def cd_vk(self, nosecone):
        # function for finding the axial drag coefficient of a von karman nosecone
        c0 = self.cd_blunt()
        fn = 0.5*nosecone.length/nosecone.radius # fineness
        exp = np.log(fn+1)/np.log(4)
        data_min = nosecone.cd_data_f3[0][0]
        data_max = nosecone.cd_data_f3[0][-1]

        # transonic/supersonic region
        if self.mach > data_min and self.mach < data_max:
            c3 = np.interp(self.mach,nosecone.cd_data_f3[0], nosecone.cd_data_f3[1])
            cd = c0*np.pow(c3/c0,exp)

        # subsonic
        elif self.mach < data_min:
            cd = 0

        # supersonic extrapolation
        else:
            # issue with this but assume constant for Mach >3
            cd = c0*np.pow(nosecone.cd_data_f3[1][-1]/c0,exp)

        return cd

    def cd_ogive(self, nosecone):
        # function for finding the axial drag coefficient of an ogive nosecone
        
        if self.mach > 1:
            cd = (0.72*(nosecone.shape_factor-0.5)**2+0.82)*self.cd_cone(nosecone)
        else:
            cd_m0 = 0.8*(np.sin(nosecone.shoulder_angle)**2)
            cd_m1 = (0.72*(nosecone.shape_factor-0.5)**2+0.82)*np.sin(nosecone.half_angle)
            cd_m1_slope = 4*(1-0.5*np.sin(nosecone.half_angle))/(self.gamma+1)

            b = cd_m1_slope/(cd_m1 - cd_m0)
            a = cd_m1-cd_m0

            cd = a*np.pow(self.mach,b)+cd_m0

        return cd

    def cd_cone(self, nosecone):
        # needs to be updated for subsonic case if being used for conical nosecone
        # function is currently only used in ogive calculations
        if self.mach >= 1.3:
            cd = 2.1*(np.sin(nosecone.half_angle)**2)+0.5*np.sin(nosecone.half_angle)/np.sqrt((self.mach**2)-1)

        else: # 1 < mach <1.3
            cd1 = np.sin(nosecone.half_angle)  # mach 1
            cd2 = 2.1*(np.sin(nosecone.half_angle)**2)+0.5*np.sin(nosecone.half_angle)/np.sqrt((1.3**2)-1)  # mach 1.3
            cd1_slope = 4*(1-0.5*np.sin(nosecone.half_angle))/(self.gamma+1)

            # quadratic interpolation
            a = (-10/9)*(10*cd1-10*cd2+3*cd1_slope)
            b = (200/9)*(cd1-cd2)+(23*cd1_slope/3)
            c = (1/9)*(-91*cd1+100*cd2-39*cd1_slope)

            cd = a*self.mach**2 + b*self.mach +c

        return cd

    def cd_boattail(self, boattail):
        # function to find the drag coeffcieint of a boattail
        lh_ratio = 0.5*boattail.length/(boattail.radius_0-boattail.radius_l)

        if lh_ratio < 1:
            coeff = 1
        elif lh_ratio < 3:
            coeff = (3-lh_ratio)/2
        else:
            coeff = 0

        # TODO double check these areas are correct ones
        cd = coeff*(boattail.area_0/boattail.area_l)*self.cd_base()

        return cd

    def cd_base(self):
        # dont add this on top of boattail drag I believe #TODO double check
        if self.mach < 1:
            cd = 0.12+10.13*self.mach**2
        else:
            cd = 0.25/self.mach
        return cd

    def cd_total(self, rocket):
        # finds the total axial drag coefficient
        cd_tot = 0
        # nosecone
        nosecone_name = rocket.parts[0].__class__.__name__ 
        if nosecone_name == 'nosecone_vk':
            cd_tot += self.cd_vk(rocket.parts[0])*(rocket.parts[0].area_l/self.Aref)
        elif nosecone_name == 'nosecone_ogive':
            cd_tot += self.cd_ogive(rocket.parts[0])*(rocket.parts[0].area_l/self.Aref)

        # transition
        for part in rocket.parts:
            if part.__class__.__name__ == 'transition_ogive':
                cd_tot += self.cd_ogive(part)*((part.area_l-part.area_0)/self.Aref)

        # boattail/base
        if rocket.parts[-1].__class__.__name__ == 'boattail':
            cd_tot += self.cd_boattail(rocket.parts[-1])*(rocket.parts[-1].area_l/self.Aref)
        else:
            cd_tot += self.cd_base()*(rocket.parts[-1].area_l/self.Aref)

        cd_tot += self.skin_friction(rocket)

        # scaling function
        a = self.alpha*180/np.pi # a in degrees
        if a <= 17:
            scale = -(3/24565)*a**3 + (9/2890)*a**2 + 1
        else:
            scale = (6.68351*10**-6)*a**3 + (-0.0010727)*a**2 + (0.0306773)*a + 1.05566

        return scale*cd_tot
            









        
         



        