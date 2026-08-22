import numpy as np

class nosecone:
    def __init__(self, radius, length, Aref):
        self.radius = radius
        self.length = length
        self.Aref = Aref
        self.area_base = np.pi*radius**2

        # Volume
        x = np.linspace(0,length)
        y = np.sqrt(np.acos(1-2*x/length)-0.5*np.sin(2*np.acos(1-2*x/length)))
        integral = np.trapezoid(y,x)
        self.volume = 2*np.sqrt(np.pi)*integral*self.radius**2


    def C_na(self, alpha):
        # function to find normal force coefficient for a von karman nose cone
        # note: daybreak uses ogive
        if alpha != 0:
            c_na = 2*self.area_base*np.sin(alpha)/(alpha*self.Aref)
        else:
            c_na = 2*self.area_base/self.Aref

        return c_na

    def C_ma(self, alpha):
        # function to find moment coeffecient for a von karman nose cone
        if alpha != 0:
            c_ma = (self.length*self.area_base-self.volume)*np.sin(alpha)/(alpha*self.Aref*self.radius)
        else:
            c_ma = (self.length*self.area_base-self.volume)/(self.Aref*self.radius)
        return c_ma

    def C_lift(self, alpha):
        #  function to find lift coeffecient
        K = 1.1
        c_lift = K*2*self.radius*self.length*(np.sin(alpha)**2)/self.Aref
        return c_lift

    def X_lift(self):
        # function to find lift location
        x = np.linspace(0,self.length)
        y = x*2*self.radius*np.sqrt(np.acos(1-2*x/self.length)-0.5*np.sin(2*np.acos(1-2*x/self.length))/np.pi)
        integral = np.trapezoid(y,x)
        return integral/(self.length*2*self.radius)

    def X_n(self,alpha):
        # function to find centre of pressure
        X_n = self.length-self.volume/self.area_base
        return X_n
    



