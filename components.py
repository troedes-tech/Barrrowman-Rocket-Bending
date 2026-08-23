import numpy as np

class nosecone_vk:
    def __init__(self, radius, length):
        self.radius = radius
        self.length = length
        self.area_l = np.pi*radius**2
        self.area_0 = 0
        self.cl = None

        # Volume
        x = np.linspace(0,length)
        y = np.sqrt(np.acos(1-2*x/length)-0.5*np.sin(2*np.acos(1-2*x/length)))
        integral = np.trapezoid(y,x)
        self.volume = 2*np.sqrt(np.pi)*integral*self.radius**2

    def r(self,x):
        # function to find radius at a given x location
        r = self.radius*np.sqrt(np.acos(1-2*x/self.length)-0.5*np.sin(2*np.acos(1-2*x/self.length))/np.pi)
        return r


class tube:
    def __init__(self, radius, length):
        self.radius = radius
        self.length = length
        self.area_l = np.pi*radius**2
        self.area_0 = np.pi*radius**2
        self.volume = np.pi*radius**2*length
        self.cl = self.length/2


class nosecone_ogive:
    def __init__(self, radius, length, shape_factor):
        self.radius = radius
        self.length = length
        self.area_l = np.pi*radius**2
        self.area_0 = 0
        self.cl = None
        self.shape_factor = shape_factor

        # Volume
        x = np.linspace(0,length)
        self.rho = ((radius**2)+(length**2))/(2*radius*shape_factor)
        self.alpha = np.atan(radius/length)-np.arccos(np.sqrt(radius**2+length**2)/2*self.rho)
        y = np.sqrt(self.rho**2-(x-self.rho*np.cos(self.alpha))**2)+self.rho*np.sin(self.alpha)
        integral = np.trapezoid(y,x)
        self.volume = 2*np.pi*integral

    def r(self,x):
        # function to find radius at a given x location
        r = np.sqrt(self.rho**2-(x-self.rho*np.cos(self.alpha))**2)+self.rho*np.sin(self.alpha)
        return r

class transition_ogive:
    #is assumed to be tangent ogive
    def __init__(self, radius_0, radius_l, length):
        self.radius_0 = radius_0
        self.radius_l = radius_l
        self.radius = radius_l
        self.length = length
        self.area_l = np.pi*radius_l**2
        self.area_0 = np.pi*radius_0**2
        self.cl = None

        self.cutoff = -length-np.sqrt(-((length**2+radius_0*(radius_0-radius_l))*(radius_0-radius_l)*radius_l))/(radius_0-radius_l)
        
        # Volume
        x = np.linspace(self.cutoff,self.cutoff+length)
        self.rho = ((radius_l**2)+(length**2))/(2*radius_l)
        y = np.sqrt(self.rho**2-(x-self.length)**2)+self.radius_l-self.rho
        integral = np.trapezoid(y,x)
        self.volume = 2*np.pi*integral

    def r(self,x):
        # function to find radius at a given x location
        r = np.sqrt(self.rho**2-((x+self.cutoff)-self.length)**2)+self.radius_l-self.rho
        return r


class boattail:
    def __init__(self, radius_0, radius_l, length):
        self.radius_0 = radius_0
        self.radius_l = radius_l
        self.radius = radius_0
        self.length = length
        self.area_0 = np.pi*radius_0**2
        self.area_l = np.pi*radius_l**2
        self.volume = np.pi*length*(radius_0**2 + radius_0*radius_l + radius_l**2)/3
        self.cl = (radius_0+2*radius_l)*length/(6*radius_0)


    



