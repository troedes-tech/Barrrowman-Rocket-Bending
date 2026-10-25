import numpy as np
from components import *


class aerodynamics:
    def __init__(self, Aref, mach, alpha, kv, temp, gamma=1.4, R_const=287):
        self.gamma = gamma
        self.Aref = Aref
        self.mach = mach
        self.alpha = alpha
        self.kv = kv
        self.R_const = R_const
        self.temp = temp

    # ---- NORMAL FORCE ----

    def C_na(self, body):
        if getattr(body, 'area_l', None) is None or getattr(body, 'area_0', None) is None:
            return 0.0

        delta_area = float(body.area_l - body.area_0)
        if np.isclose(delta_area, 0.0):
            return 0.0

        if np.isclose(self.alpha, 0.0):
            return 2.0 * delta_area / self.Aref
        return 2.0 * delta_area * np.sin(self.alpha) / (self.alpha * self.Aref)

    def C_n_lift(self, body, K=1.1):
        if getattr(body, 'radius', None) is None or getattr(body, 'length', None) is None:
            return 0.0
        if body.length <= 0 or body.radius <= 0:
            return 0.0

        return K * 2.0 * body.radius * body.length * (np.sin(self.alpha) ** 2) / self.Aref

    def C_na_fins(self, fin, tube):
        if getattr(fin, 'span', None) is None or getattr(fin, 'area', None) is None:
            return 0.0

        beta = np.sqrt(np.abs(1.0 - self.mach ** 2))
        if self.mach < 1.0:
            denom = 1.0 + np.sqrt(1.0 + (beta * (fin.span ** 2) / (fin.area * np.cos(fin.Lc))) ** 2)
            c_na_single = (2.0 * np.pi * (fin.span ** 2) / self.Aref) / denom
        else:
            k1 = 2.0 / beta
            k2 = ((self.gamma + 1.0) * self.mach ** 4 - 4.0 * beta ** 2) / (4.0 * beta ** 4)
            k3 = ((self.gamma + 1.0) * self.mach ** 8 + (2.0 * self.gamma ** 2 - 7.0 * self.gamma - 5.0) * self.mach ** 6 + 10.0 * (self.gamma + 1.0) * self.mach ** 4 + 8.0) / (6.0 * beta ** 7)
            c_na_single = (fin.area / self.Aref) * (k1 + k2 * self.alpha + k3 * self.alpha ** 2)

        c_na = 2.0 * c_na_single
        if getattr(tube, 'radius', None) is not None and getattr(tube, 'span', None) is not None:
            kt = 1.0 + tube.radius / (tube.span + tube.radius)
            c_na *= kt
        return c_na

    def C_n_total(self, rocket):
        cn_total = 0.0
        for part in getattr(rocket, 'parts', []):
            area_change = getattr(part, 'area_l', None) is not None and getattr(part, 'area_0', None) is not None
            if area_change and not np.isclose(part.area_l, part.area_0):
                cn_total += self.C_na(part) * self.alpha

            if getattr(part, 'radius', None) is not None and getattr(part, 'length', None) is not None:
                if part.length > 0 and part.radius > 0:
                    k_lift = 1.1
                    if area_change and np.isclose(part.area_l, part.area_0):
                        k_lift = 0.6 #TODO where did AI get this
                    cn_total += self.C_n_lift(part, K=k_lift)

        if getattr(rocket, 'fins', None) is not None and getattr(rocket, 'fin_tube', None) is not None:
            cn_total += self.C_na_fins(rocket.fins, rocket.fin_tube) * self.alpha
        return cn_total

    # ---- MOMENTS ----

    def C_ma(self, body):
        if getattr(body, 'area_l', None) is None or getattr(body, 'area_0', None) is None:
            return 0.0

        delta_area = float(body.area_l - body.area_0)
        if np.isclose(delta_area, 0.0):
            return 0.0

        numerator = body.length * body.area_l - body.volume
        if np.isclose(self.alpha, 0.0):
            return numerator / (self.Aref * body.radius)
        return numerator * np.sin(self.alpha) / (self.alpha * self.Aref * body.radius)

    def pitch_damp(self, rocket=None):
        return 0.0

    def C_m_total(self, rocket):
        c_ma_total = 0.0
        x = 0.0

        for part in getattr(rocket, 'parts', []):
            arm = x
            if getattr(part, 'area_l', None) is not None and getattr(part, 'area_0', None) is not None:
                if not np.isclose(part.area_l, part.area_0):
                    c_ma_total += self.C_ma(part) * (arm + self.X_n(part))

            if getattr(part, 'radius', None) is not None and getattr(part, 'length', None) is not None:
                if part.length > 0 and part.radius > 0:
                    c_ma_total += self.C_n_lift(part) * (arm + self.X_lift(part))

            x += getattr(part, 'length', 0.0)

        if getattr(rocket, 'fins', None) is not None and getattr(rocket, 'fin_tube', None) is not None:
            fin_arm = rocket.fins.location + self.cp_fins(rocket.fins)[0]
            c_ma_total += self.C_na_fins(rocket.fins, rocket.fin_tube) * self.alpha * fin_arm

        return c_ma_total / self.Aref

    # ---- ROLL DYNAMICS ----

    def equil_roll(self, rocket=None):
        return 0.0

    def roll_damp(self, rocket=None):
        return 0.0

    # ---- CENTRE OF PRESSURE ----

    def X_lift(self, body):
        if getattr(body, 'cl', None) is not None:
            return body.cl
        else:
            x = np.linspace(0,body.length)
            y = x*2*body.r(x)
            integral = np.trapezoid(y,x)
            return integral/(body.length*2*body.radius)

    def X_n(self, body):
        if getattr(body, 'area_l', None) is None or getattr(body, 'area_0', None) is None:
            return 0.0
        if np.isclose(body.area_l, body.area_0):
            return 0.0
        return (body.length * body.area_l - body.volume) / (body.area_l - body.area_0)

    def cp_fins(self, fins):
        y = (fins.span / 3.0) * (fins.root_chord + 2.0 * fins.tip_chord) / (fins.root_chord + fins.tip_chord)
        if self.mach < 1.0:
            x = (fins.sweep_length / 3.0) * (fins.root_chord + 2.0 * fins.tip_chord) / (fins.root_chord + fins.tip_chord) + (fins.root_chord ** 2 + fins.root_chord * fins.tip_chord + 3.0 * fins.tip_chord ** 2) / (6.0 * (fins.root_chord + fins.tip_chord))
        elif self.mach < 1.3:
            x_sub = (fins.sweep_length / 3.0) * (fins.root_chord + 2.0 * fins.tip_chord) / (fins.root_chord + fins.tip_chord) + (fins.root_chord ** 2 + fins.root_chord * fins.tip_chord + 3.0 * fins.tip_chord ** 2) / (6.0 * (fins.root_chord + fins.tip_chord))
            beta = np.sqrt(np.abs(1.0 - self.mach ** 2))
            x_sup = (fins.ar * beta - 0.67) / (2.0 * fins.ar * beta - 1.0) * fins.c_hat
            t = (self.mach - 1.0) / 0.3
            x = x_sub + t * (x_sup - x_sub)
        else:
            beta = np.sqrt(np.abs(1.0 - self.mach ** 2))
            x = (fins.ar * beta - 0.67) / (2.0 * fins.ar * beta - 1.0) * fins.c_hat
        return [x, y]

    def cp(self, rocket): #TODO check this
        x = 0.0
        sum_top = 0.0

        for part in getattr(rocket, 'parts', []):
            if getattr(part, 'area_l', None) is not None and getattr(part, 'area_0', None) is not None:
                if not np.isclose(part.area_l, part.area_0):
                    sum_top += (x + self.X_n(part)) * self.C_na(part) * self.alpha

            if getattr(part, 'radius', None) is not None and getattr(part, 'length', None) is not None:
                if part.length > 0 and part.radius > 0:
                    sum_top += (x + self.X_lift(part)) * self.C_n_lift(part)

            x += getattr(part, 'length', 0.0)

        if getattr(rocket, 'fins', None) is not None and getattr(rocket, 'fin_tube', None) is not None:
            fin_cp = self.cp_fins(rocket.fins)
            sum_top += (rocket.fins.location + fin_cp[0]) * self.C_na_fins(rocket.fins, rocket.fin_tube) * self.alpha

        sum_bottom = self.C_n_total(rocket)
        if np.isclose(sum_bottom, 0.0):
            return 0.0
        return sum_top / sum_bottom

    # ---- DRAG FORCES ----

    def skin_friction(self, rocket):
        velocity = self.mach * np.sqrt(self.gamma * self.R_const * self.temp)
        cd_skin = 0.0

        for part in getattr(rocket, 'parts', []):
            if getattr(part, 'length', None) is None:
                continue
            L = max(float(part.length), 1e-9)
            Re = velocity * L / self.kv
            roughness_ratio = getattr(rocket, 'roughness', 0.0) / L if L > 0 else 0.0
            Re_crit = 51.0 * (roughness_ratio) ** -1.039 if roughness_ratio > 0 else 1e9

            if Re < 1e-4:
                cf = 1.48e-2
            elif Re < Re_crit:
                cf = 1.0 / ((1.5 * np.log(Re) - 5.6) ** 2)
            else:
                cf = 0.032 * (roughness_ratio ** 0.2)

            if self.mach < 1.0:
                cf *= (1.0 - 0.1 * self.mach ** 2)
            elif Re < Re_crit:
                cf /= (1.0 + 0.15 * self.mach ** 2) ** 0.58
            else:
                cf /= (1.0 + 0.18 * self.mach ** 2)

            area = getattr(part, 'Awet', None)
            if area is None and getattr(part, 'radius', None) is not None:
                area = 2.0 * np.pi * part.radius * L
            if area is not None:
                cd_skin += (cf * area / self.Aref)

        if getattr(rocket, 'fins', None) is not None:
            fin = rocket.fins
            span = max(float(fin.span), 1e-9)
            fin_Re = velocity * span / self.kv
            roughness_ratio = getattr(rocket, 'roughness', 0.0) / span if span > 0 else 0.0
            Re_crit_fin = 51.0 * (roughness_ratio) ** -1.039 if roughness_ratio > 0 else 1e9
            if fin_Re < 1e-4:
                fin_cf = 1.48e-2
            elif fin_Re < Re_crit_fin:
                fin_cf = 1.0 / ((1.5 * np.log(fin_Re) - 5.6) ** 2)
            else:
                fin_cf = 0.032 * (roughness_ratio ** 0.2)
            if self.mach < 1.0:
                fin_cf *= (1.0 - 0.1 * self.mach ** 2)
            cd_skin += (fin_cf / self.Aref) * (1.0 + 2.0 * fin.thickness / fin.c_hat) * 8.0 * fin.area

        return cd_skin

    def cd_blunt(self):
        if self.mach < 1.0:
            return 0.85 * (1.0 + (self.mach ** 2) / 4.0 + (self.mach ** 4) / 40.0)
        return 0.85 * (1.84 - 0.76 / (self.mach ** 2) + 0.166 / (self.mach ** 4) + 0.035 / (self.mach ** 6))

    def cd_vk(self, nosecone):
        c0 = self.cd_blunt()
        fn = 0.5 * nosecone.length / nosecone.radius
        exp = np.log(fn + 1.0) / np.log(4.0)
        data_min = nosecone.cd_data_f3[0][0]
        data_max = nosecone.cd_data_f3[0][-1]

        if data_min < self.mach < data_max:
            c3 = np.interp(self.mach, nosecone.cd_data_f3[0], nosecone.cd_data_f3[1])
            return c0 * (c3 / c0) ** exp

        if self.mach <= data_min:
            c3 = nosecone.cd_data_f3[1][0] 
            return c0 * (c3 / c0) ** exp    #TODO i think this may be 0

        c3 = nosecone.cd_data_f3[1][-1]
        return c0 * (c3 / c0) ** exp

    def cd_ogive(self, nosecone):
        if self.mach > 1.0:
            return (0.72 * (nosecone.shape_factor - 0.5) ** 2 + 0.82) * self.cd_cone(nosecone)

        cd_m0 = 0.8 * (np.sin(nosecone.shoulder_angle) ** 2)
        cd_m1 = (0.72 * (nosecone.shape_factor - 0.5) ** 2 + 0.82) * np.sin(nosecone.half_angle)
        cd_m1_slope = 4.0 * (1.0 - 0.5 * np.sin(nosecone.half_angle)) / (self.gamma + 1.0)

        b = cd_m1_slope / (cd_m1 - cd_m0)
        a = cd_m1 - cd_m0
        return a * np.power(self.mach, b) + cd_m0

    def cd_cone(self, nosecone):
        if self.mach >= 1.3:
            return 2.1 * (np.sin(nosecone.half_angle) ** 2) + 0.5 * np.sin(nosecone.half_angle) / np.sqrt((self.mach ** 2) - 1.0)

        cd1 = np.sin(nosecone.half_angle)
        cd2 = 2.1 * (np.sin(nosecone.half_angle) ** 2) + 0.5 * np.sin(nosecone.half_angle) / np.sqrt((1.3 ** 2) - 1.0)
        cd1_slope = 4.0 * (1.0 - 0.5 * np.sin(nosecone.half_angle)) / (self.gamma + 1.0)

        a = (-10.0 / 9.0) * (10.0 * cd1 - 10.0 * cd2 + 3.0 * cd1_slope)
        b = (200.0 / 9.0) * (cd1 - cd2) + (23.0 * cd1_slope / 3.0)
        c = (1.0 / 9.0) * (-91.0 * cd1 + 100.0 * cd2 - 39.0 * cd1_slope)
        return a * self.mach ** 2 + b * self.mach + c

    def cd_boattail(self, boattail):
        if getattr(boattail, 'length', None) is None or getattr(boattail, 'radius_0', None) is None or getattr(boattail, 'radius_l', None) is None:
            return 0.0

        delta_r = boattail.radius_0 - boattail.radius_l
        if np.isclose(delta_r, 0.0):
            return 0.0

        lh_ratio = 0.5 * boattail.length / delta_r
        if lh_ratio < 1.0:
            coeff = 1.0
        elif lh_ratio < 3.0:
            coeff = (3.0 - lh_ratio) / 2.0
        else:
            coeff = 0.0

        #TODO double check these area are correct ones

        area_ref = max(float(getattr(boattail, 'area_l', boattail.area_0)), float(getattr(boattail, 'area_0', boattail.area_l)), 1e-9)
        return coeff * (max(float(boattail.area_0), area_ref) / area_ref) * self.cd_base()

    def cd_base(self):
        # dont add this on top of boattail drag I believe #TODO double check
        if self.mach < 1:
            cd = 0.12+10.13*self.mach**2
        else:
            cd = 0.25/self.mach
        return cd

    def cd_fins(self, fins):
        le_angle = np.atan(fins.sweep_length / fins.span)

        if self.mach < 0.9:
            cd_le = ((1.0 - self.mach ** 2) ** -0.417) - 1.0
        elif self.mach < 1.0:
            cd_le = 1.0 - 1.785 * (self.mach - 0.9)
        else:
            cd_le = 1.214 - 0.502 / (self.mach ** 2) + 0.1085 / (self.mach ** 4)
        return cd_le * np.cos(le_angle) ** 2

    def cd_total(self, rocket):
        if len(getattr(rocket, 'parts', [])) == 0:
            return 0.0

        cd_tot = 0.0
        nosecone_name = rocket.parts[0].__class__.__name__
        if nosecone_name == 'nosecone_vk':
            cd_tot += self.cd_vk(rocket.parts[0]) * (rocket.parts[0].area_l / self.Aref)
        elif nosecone_name == 'nosecone_ogive':
            cd_tot += self.cd_ogive(rocket.parts[0]) * (rocket.parts[0].area_l / self.Aref)

        for part in rocket.parts:
            if part.__class__.__name__ == 'transition_ogive':
                cd_tot += self.cd_ogive(part) * ((part.area_l - part.area_0) / self.Aref)

        if getattr(rocket, 'fins', None) is not None:
            cd_tot += self.cd_fins(rocket.fins) * ((4.0 * rocket.fins.thickness * rocket.fins.span) / self.Aref)

        aft_part = rocket.parts[-1]
        if aft_part.__class__.__name__ == 'boattail':
            cd_tot += self.cd_boattail(aft_part) * (aft_part.area_l / self.Aref)
        else:
            cd_tot += self.cd_base() * (aft_part.area_l / self.Aref)

        cd_tot += self.skin_friction(rocket)

        a = self.alpha * 180.0 / np.pi
        if a <= 17.0:
            scale = -(3.0 / 24565.0) * a ** 3 + (9.0 / 2890.0) * a ** 2 + 1.0
        else:
            scale = (6.68351e-6) * a ** 3 + (-0.0010727) * a ** 2 + (0.0306773) * a + 1.05566

        return scale * cd_tot









        
         



        