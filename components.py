import numpy as np

class nosecone_vk:
    def __init__(self, radius, length):
        self.radius = radius
        self.length = length
        self.area_l = np.pi*radius**2
        self.cp = None

        # Volume
        x = np.linspace(0,length)
        y = np.sqrt(np.acos(1-2*x/length)-0.5*np.sin(2*np.acos(1-2*x/length)))
        integral = np.trapezoid(y,x)
        self.volume = 2*np.sqrt(np.pi)*integral*self.radius**2

    def r(self,x):
        # function to find radius at a given x location
        r = self.radius*np.sqrt(np.acos(1-2*x/self.length)-0.5*np.sin(2*np.acos(1-2*x/self.length))/np.pi)
        return r


class body_tube:
    def __init__(self, radius, length):
        self.radius = radius
        self.length = length
        self.area_l = np.pi*radius**2
        self.area_0 = np.pi*radius**2
        self.volume = np.pi*radius**2*length
        self.cp = self.length/2


class nosecone_ogive:
    def __init__(self, radius, length, shape_factor):
        self.radius = radius
        self.length = length
        self.area_l = np.pi*radius**2
        self.cp = None
        self.shape_factor = shape_factor

        # Volume
        x = np.linspace(0,length)
        self.rho = (radius+(length**2)/radius)/shape_factor
        self.alpha = np.atan(radius/length)-np.arccos(np.sqrt(radius**2+length**2)/2*self.rho)
        y = np.sqrt(self.rho**2-(x-self.rho*np.cos(self.alpha))**2)+self.rho*np.sin(self.alpha)
        integral = np.trapezoid(y,x)
        self.volume = 2*np.pi*integral

    def r(self,x):
        # function to find radius at a given x location
        r = np.sqrt(self.rho**2-(x-self.rho*np.cos(self.alpha))**2)+self.rho*np.sin(self.alpha)
        return r

    


    



