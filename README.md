# MMA3001 Project Documentation

## 1. Engineering problem
The Monash High Powered Rocketry team launches rockets at competitions with the aim of getting as close to 10,000 ft. To ensure these rockets are optimised and safe for this problem the team has a  simulator. This simulator requires an input of aerodynamic data about the rocket. For this project I have decided to write code that is able to generate an aerodynamic database appropriate for putting into the team’s simulator to help them make the safest and most efficient rockets possible.

## 2. Inputs and outputs
The major inputs to my project are the geometry of the rocket and its aerodynamic regime. Specifically for the geometry it is a list of components from top to bottom; nosecone, tubes, shoulders, boattail and fins as well as their specific dimensions. All numbers for this project are inputted in SI units so for geometry it is all in metres.

The aerodynamic regime is what values and combinations of Mach number, Reynolds number and Angle of attack the rocket is expected to experience. These form the basis of a database for which the drag coefficients for force and moment as well as the centre of pressure are contained. How fine of a resolution the database is is also an input. This is to say what increment of mach number or of angle of attack is wanted. This is the format that is used by the team's simulator allowing them to accurately simulate the rocket throughout flight.

All functions have appropriate checking on the data type of all inputs and will return appropriate errors if they are not the specified type. All function inputs and outputs have their docs made automatically using the pdoc module on the repository.

## 3. Computational solution
The way my solution works is by using the Extended Barrowman equations to find the drag on the rocket. First it parses a geometry and the current aerodynamic conditions. For each component in the rocket it calculates the normal force, lift and normal moment coefficient via the given equations. It then calculates the net normal force and moment coefficients and centre of pressure.One step in this which is significant is that the Extended Barrowman equations require many integrals to be done which must be completed numerically for certain nose cone profiles in particular. For my implementation I have gone with a 4th order Runge Kutta method. 

After this the axial drag coefficient and skin friction for both body and fins is calculated. This is all combined into a drag coefficient vector based on body coordinates for both force and moment. These are outputted into a database and this process is repeated for all Mach numbers, angles of attack and Reynolds numbers specified by the user. 
This database can be exported or linearly interpolated between to find drag values input setpoints. 

There are a lot of assumptions made within the Extended Barrowman equations that can be found within the resource itself (Niskanen, 2013) . Primarily it becomes highly error prone in the transonic and supersonic regimes. 
The main assumption made by myself is the appropriateness to linearly interpolate between values. This can become particularly error prone for large increments of Mach number particularly around Mach 1 where the aerodynamics are changing significantly and require high resolution. Some resources (Zipfel, 2007)  also indicate that it can be appropriate to not interpolate by Reynolds for rockets where the kinematic viscosity is fairly constant throughout flight as it changes roughly in proportion to mach number and is an assumption that Monash HPR has had success with making in the past.

## 4. Alternative solutions
There are quite a few alternative methods for creating aerodynamic databases. The method I chose has its strengths in low computational complexity. It allows for fast agile design processes allowing the team to iterate on many different geometries. Previously the team has used CFD for these databases and whilst they are incredibly accurate with our team having simulated to within 5 m of actual apogee the computational cost is incredibly high even with the universities large computer clusters. This has held us back from simulating multiple design options early on in the project. 

The Extended Barrowman equations are well known for their use in rocketry and are used in one of the most common rocket simulators, OpenRocket. The implementation differs slightly here as the drag is recalculated at every timestep based on the rocket’s current state rather than interpolating through a database of pre-calculated values. The interpolation allows for very low run times in the teams simulator allowing for capabilities doing things such as Monte Carlo simulations where hundreds of simulations are run to characterise the rockets behaviour across a variety of conditions.

The final option is one which can fill a very similar role and that is using a panel code method. It is a medium fidelity option, lying in between the extended barrowman equations and full CFD in terms of both accuracy and computational time. It is my understanding that another member from Monash HPR is doing this for their MMA3001 project. From a systems engineering perspective all of the options outlined are great for the team in different scenarios. Extended Barrowmans are great during the early design phase, when vastly different designs are being considered and the rocket needs to be optimised. Panel code is perfect for the lead up to manufacturing, making sure the rocket is behaving as expected but not investing so much time that a small geometry change will throw things off the rails. CFD is important for the lead up to a launch to have the utmost confidence in having a safe launch, as well as to capture more difficult relations such as the effect of exhaust or fin flutter.

### Method
### Extended Barrowmans
#### Pros
- Low computational cost
- No meshing required
#### Cons
- Low accuracy
- Struggles in the supersonic regime
### Panel Code
#### Pros
- Improved supersonic and even hypersonic regime handling
#### Cons
- Meshing required
- Medium computational cost
### CFD
#### Pros
- Highest accuracy
- Extended capabilities with fin flutter and exhaust
- More complex geometries
#### Cons
- Extremely high computational cost

Ultimately I chose to use the Extended Barrowmans equations to model the aerodynamics of the rocket as it was the option that the team currently did not have.

## 5. Validation
The major way I was able to validate my code is by comparison to an open source software called OpenRocket. This is a trusted benchmark as it has been used for hundreds of rockets across the world and is trusted by all the world’s rocketry competitions for a safe rocket launch. 
I used this repository for validation in two ways. The first and primary method was to compare output to output and see that they were within an acceptable range from each other. This meant simulating the exact same geometry within both code bases and seeing that all coefficients were the same, normal force, normal moment, total drag etc. I did this for the team's most recent rocket, Meridian. This allowed me to be confident that not only was my final result correct but also that the individual functions were accurate. This also helped with error handling.
The other way it was used for validation was going into the codebase and comparing the exact derived coefficients for some of the equations used and checking that they matched. 

This validation only goes to the extent of checking the maths has been implemented and corrected and utilises the validation already done for the OpenRocket codebase. In the future it would be important to compare my code itself to wind tunnel, CFD or DATCOM data so it is independently verified. This would become particularly important as more complex geometries such as for airbrakes or canards are added as the team moves forward. Unfortunately that was not within the scope or manageable within the timeframe of this project.

The numerical integration completed for the nosecones was validated using Mathematica as a source of truth so as to validate that my step size was of an appropriate size. This was also validated when I was using numerical integration for a nose cone shape I did not need to and then posthumously compared it to the analytical solution (conical nosecones). 

The linear interpolation I very simply validated by just choosing a data point halfway between two of my data points and doing the linear interpolation by hand to check that it had been done correctly.

## 6. Optimisation and performance
I mostly wanted to optimise the interpolator as I felt this is the area with the most room for improvement. I felt that in real life drag is continuous and smooth and so a cubic spline interpolator could be of help. However this could add serious run time so I ran the code 3 times and got an average run time of 0.007793s. This would allow me to best weigh up whether it is a worthy advantage to do cubic spline interpolation. If there is no significant increase then it is worth it but otherwise the method may as well lean into what it is good at and just do more refined precomputing. Leave it as linear interpolation with a greater resolution of mach number and angle of attack.

Very interestingly after changing it to cubic spline interpolation with some help from Artificial intelligence all the results still looked rather good however the run time decreased to 0.00275s. I believe this is because now the code runs a cubic spline interpolant twice rather than using scipy’s N-dimensional interpolator. This was definitely a great optimisation to make and shows the benefit in sometimes doing a bit more hard coding rather than using a module.

## 7. AI use and reflection
During this task I used AI to help me code. This was primarily the gemini agent that is free and built in to vs code. I used it in a variety of ways that I felt not only maintained the quality of the repository but also helped me learn when AI is and isn’t useful. One feature that I loved about the gemini agent is that it would have green tick boxes for all the changes it made and show you exactly where those were so that you could decide if you wanted to keep them.

When starting off I typed all the equations up myself based on the resources I had. I didn’t feel comfortable with AI doing this step as it is where the core of my understanding of the processes would come from and also I can’t imagine it would contain many training sets that had these equations. What I did do once I had typed up the equations myself and tested that they functioned was ask AI to check over them and provide any improvements. This worked really well as it added a lot of great quality of life features such as checking inputs, highlighting a few points where I had made mistakes. It also reassured me in my approach as some of its suggested changes I disagreed with from the literature and ultimately didn’t accept.

When it came to putting it all together into a database and linearly interpolating between these results I thought that this could be a good opportunity to learn how capable AI is and allow it to have the first pass this time. I thought this would also be a more appropriate task as I am sure the AI agent has seen linear interpolation millions of times in its training so it was a much simpler task for it to get correct. This didn’t work very well with the AI writing a lot of unnecessary fluff and in future I think I would type out some pseudocode for it first. I also don’t believe it could work across files in the same repository surprisingly and so when helping with the linear interpolation putting a sample of it in the prompt helped immensely. 

I also used AI to help me cite my sources and develop the docstrings.

## 8. References
Niskanen, S. (2013) OpenRocket technical documentation. 10 May 
Zipfel, P.H. (2007) Modeling and simulation of aerospace vehicle dynamics. 2nd edn. Reston, VA: American Institute of Aeronautics and Astronautics. 
Crowell, G.A. (1996) The descriptive geometry of nose cones. 
Niskanen, S. (2009) Development of an open source model rocket simulation software. MSc thesis. Helsinki University of Technology. Available at: http://openrocket.sourceforge.net/documentation.html  (Accessed: 25 October 2026).
