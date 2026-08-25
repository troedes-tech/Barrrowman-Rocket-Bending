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


class fin:
    def __init__(self, root_chord, tip_chord, span, sweep_length):
        self.root_chord = root_chord
        self.tip_chord = tip_chord
        self.span = span
        self.sweep_length = sweep_length
        self.area = (root_chord+tip_chord)*span/2
        self.ar = 2*self.area*self.span**2
        self.Lc = np.atan2(0.5*root_chord+sweep_length-0.5*tip_chord,span)

    def cp(self, mach):
        # function to find centre of pressure of a fin
        y = (self.span/3)*(self.root_chord+2*self.tip_chord)/(self.root_chord+self.tip_chord)
        x = (self.sweep_length/3)*(self.root_chord+2*self.tip_chord)/(self.root_chord+self.tip_chord)+(self.root_chord**2+self.root_chord*self.tip_chord+3*self.tip_chord**2)/(6*(self.root_chord+self.tip_chord))
        if mach < 0.5:
            return x,y
        else:
            beta = np.sqrt(np.abs(1-mach**2))
            f = (self.ar*beta-0.67)/(2*self.ar*beta-1)
            x = f*x
            return x,y


