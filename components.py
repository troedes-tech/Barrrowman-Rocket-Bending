import numpy as np

class nosecone_vk:
    def __init__(self, radius, length):
        """Model a von Karman nose cone.
        
            Args:
                radius (float): Maximum nose radius.
                length (float): Nose cone length.
        
            Returns:
                None: Sets up nose cone geometry and derived aerodynamic properties.
            """
        self.radius = radius
        self.length = length
        self.area_l = np.pi*radius**2
        self.area_0 = 0
        self.cl = None

        # data mach v cd for von karman nose cones of fineness 3
        # source: OR repository
        # The original double braces create nested sets and fail at runtime.
        # Keep Mach and Cd data ordered for interpolation.
        self.cd_data_f3 = np.array([
            [0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.6, 2.0, 3.0],
            [0, 0.010, 0.027, 0.055, 0.070, 0.081, 0.095, 0.097, 0.091, 0.083]
        ])

        # Volume and Surface Area
        x = np.linspace(0,length)
        r = self.radius*np.sqrt(np.acos(1-2*x/self.length)-0.5*np.sin(2*np.acos(1-2*x/self.length))/np.pi)
        self.volume = np.pi*np.trapezoid(r**2,x)
        self.Awet = 2*np.pi*np.trapezoid(r,x)   # Surface Area

    def r(self,x):
        """Return the local radius at a given axial location.

        Args:
            x (float or np.ndarray): Axial position(s) measured from the nose tip.

        Returns:
            float or np.ndarray: Radius of the nose cone at the requested position(s).
        """
        # function to find radius at a given x location
        r = self.radius*np.sqrt(np.acos(1-2*x/self.length)-0.5*np.sin(2*np.acos(1-2*x/self.length))/np.pi)
        return r


class tube:
    def __init__(self, radius, length):
        """Initialize a tube section.

        Args:
            radius (float): Tube radius.
            length (float): Tube length.

        Returns:
            None: Stores the tube geometry, area, and centroid information.
        """
        self.radius = radius
        self.length = length
        self.area_l = np.pi*radius**2
        self.area_0 = np.pi*radius**2
        self.volume = np.pi*radius**2*length
        self.Awet = 2*np.pi*radius*length
        self.cl = self.length/2


class nosecone_ogive:
    def __init__(self, radius, length, shape_factor):
        """Initialize an ogive nose cone.

        Args:
            radius (float): Maximum radius of the ogive nose cone.
            length (float): Length of the ogive nose cone.
            shape_factor (float): Shape factor used to define the ogive profile.

        Returns:
            None: Stores the ogive dimensions, volume, and wetted area.
        """
        self.radius = radius
        self.length = length
        self.area_l = np.pi*radius**2
        self.area_0 = 0
        self.cl = None
        self.shape_factor = shape_factor
        self.half_angle = np.atan(radius/length)

        # Volume and Surface Area
        x = np.linspace(0,length)
        k = shape_factor
        self.rho_squared = ((radius**2)+(length**2))*(((k*radius)**2)+(((2-k)*length)**2))/(4*(radius*k)**2)
        r = np.sqrt(self.rho_squared-(length/k - x)**2) - np.sqrt(self.rho_squared-(length/k)**2)
        self.volume = np.pi*np.trapezoid(r**2,x)
        self.Awet = 2*np.pi*np.trapezoid(r,x)   # Surface Area

        #shoulder angle
        self.shoulder_angle = -np.atan(2*(k-1)*length*radius/np.abs((k-2)*(length**2)-k*radius**2))

    def r(self,x):
        """Return the local radius of the ogive profile at a given axial station.

        Args:
            x (float or np.ndarray): Axial station(s) from the nose tip.

        Returns:
            float or np.ndarray: Ogive radius at the requested station(s).
        """
        # function to find radius at a given x location
        k = self.shape_factor
        r = np.sqrt(self.rho_squared-(self.length/k - x)**2) - np.sqrt(self.rho_squared-(self.length/k)**2)
        return r

class transition_ogive:
    #is assumed to be tangent ogive
    def __init__(self, radius_0, radius_l, length):
        """Initialize an ogive transition.

        Args:
            radius_0 (float): Initial radius.
            radius_l (float): Final radius.
            length (float): Transition length.

        Returns:
            None: Stores transition parameters and calculated geometry.
        """
        self.radius_0 = radius_0
        self.radius_l = radius_l
        self.radius = radius_l
        self.length = length
        self.area_l = np.pi*radius_l**2
        self.area_0 = np.pi*radius_0**2
        self.shoulder_angle = 0
        self.cl = None

        self.cutoff = -length-np.sqrt(-((length**2+radius_0*(radius_0-radius_l))*(radius_0-radius_l)*radius_l))/(radius_0-radius_l)
        
        # Volume
        x = np.linspace(self.cutoff,self.cutoff+length)
        self.rho = ((radius_l**2)+(length**2))/(2*radius_l)
        r = np.sqrt(self.rho**2-((x+self.cutoff)-self.length)**2)- np.sqrt(self.rho**2-self.length**2)
        self.volume = np.pi*np.trapezoid(r**2,x)
        self.Awet = 2*np.pi*np.trapezoid(r,x)   # Surface Area

    def r(self,x):
        """Return the local radius of the transition section.

        Args:
            x (float or np.ndarray): Axial station(s) in the transition coordinate system.

        Returns:
            float or np.ndarray: Radius at the requested transition station(s).
        """
        # function to find radius at a given x location
        r = np.sqrt(self.rho**2-((x+self.cutoff)-self.length)**2)- np.sqrt(self.rho**2-self.length**2)
        return r


class boattail:
    def __init__(self, radius_0, radius_l, length):
        """Initialize a boattail section.

        Args:
            radius_0 (float): Forward radius.
            radius_l (float): Rear radius.
            length (float): Boattail length.

        Returns:
            None: Stores boattail dimensions and computed geometry.
        """
        self.radius_0 = radius_0
        self.radius_l = radius_l
        self.radius = radius_0
        self.length = length
        self.area_0 = np.pi*radius_0**2
        self.area_l = np.pi*radius_l**2
        self.volume = np.pi*length*(radius_0**2 + radius_0*radius_l + radius_l**2)/3
        self.Awet = np.pi*(radius_0+radius_l)*np.sqrt((radius_0-radius_l)**2+length**2)
        self.cl = (radius_0+2*radius_l)*length/(6*radius_0)


class fin:
    def __init__(self, root_chord, tip_chord, span, sweep_length, thickness, location):
        """Initialize a trapezoidal fin.

        Args:
            root_chord (float): Root chord length.
            tip_chord (float): Tip chord length.
            span (float): Fin span.
            sweep_length (float): Sweep length.
            thickness (float): Fin thickness.
            location (float): Position of the fin relative to the rocket tip.

        Returns:
            None: Stores fin dimensions and derived planform metrics.
        """
        self.root_chord = root_chord
        self.tip_chord = tip_chord
        self.span = span
        self.sweep_length = sweep_length
        self.area = (root_chord+tip_chord)*span/2
        self.thickness = thickness
        # For one trapezoidal fin, planform aspect ratio is span**2 / area.
        # Use 2*area only if the aerodynamic model defines area as one-sided
        # area while span is the full two-sided span.
        self.ar = 2*self.span**2/self.area
        self.Lc = np.atan2(0.5*root_chord+sweep_length-0.5*tip_chord,span)
        self.location = location #location of top of part from top of rocket
        self.c_hat = np.sqrt((0.5*root_chord+sweep_length-0.5*tip_chord)**2+span**2)# mean aerodynamic chord length


class rocket:
    def __init__(self, fins, parts, roughness):
        """Initialize a rocket model.

        Args:
            fins (fin): Fin definition used to locate the fin section.
            parts (list): Rocket body sections.
            roughness (float): Surface roughness value.

        Returns:
            None: Stores assembly length, diameter, wetted area, and fineness ratio.
        """
        self.fins = fins
        self.parts = parts
        self.roughness = roughness
        
        self.length = 0
        self.max_diameter = 0
        self.Awet_body = 0
        for part in parts:
            if self.length <= fins.location and fins.location <= self.length + part.length:
                self.fin_tube = part
            self.length += part.length
            if self.max_diameter < 2*part.radius:
                self.max_diameter = 2*part.radius
            self.Awet_body += part.Awet

        # Mathematical precondition: parts must be non-empty and have a
        # positive diameter; otherwise this ratio is undefined.
        self.fineness_ratio = self.length/self.max_diameter

        
        


    



