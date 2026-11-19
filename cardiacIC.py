##### IMPORTS #####
from datetime import datetime
from scipy.integrate import solve_ivp
import os
import numpy as np
import sympy as sp
##### ENDS OF IMPORTS #####

##### USER INPUTS #####

#INPUT: Please set if you want to use geometrized or SI units
geometrized_units = True

#INPUT: Please define the symbols of your coordinate system with the timelike coordinate being the FIRST symbol.
coords = sp.symbols("t x y z")

for coord in coords: #DO NOT TOUCH THIS LOOP
    globals()[str(coord)] = coord

#INPUT: Please set if the spacelike coordinates represent a "length" or an "angle" where 1=angular and 0=length.
is_angular = [
    0 ,                                         # SPACELIKE COORD 1
    0 ,                                         # SPACELIKE COORD 2
    0                                           # SPACELIKE COORD 3
]

#INPUT: Please set the plotting ranges of the coordinates (only important for visualizing singularities) and types of singularities to plot.
coordinate_ranges = [
    (0 , 1000) ,                                # TIMELIKE  COORD
    (0 , 1000) ,                                  # SPACELIKE COORD 1
    (0 , 1000) ,                               # SPACELIKE COORD 2
    (0 , 1000)                             # SPACELIKE COORD 3
]
calculate_CoordSingularities = True
calculate_CurveSingularities = False

#INPUT: Please define the transformation of your spatial coordinates to orthonormal cartesian coordinates.
cartesian_transformation = [
    x ,           # X = ?
    y ,           # Y = ?
    z                          # Z = ?
]

#INPUT: Please set the particle mass and charge.
particle_mass = 1
particle_charge = 0

#INPUT: Please set the 4-potential of spacetime with each component in same order as defined by coords.
Apot = sp.Matrix([
    0 ,                                        # TIMELIKE  COMPONENT
    0 ,                                        # SPACELIKE COMPONENT 1
    0 ,                                        # SPACELIKE COMPONENT 2
    0                                          # SPACELIKE COMPONENT 3
])

#INPUT: Please define the metric of spacetime with each column in same order as defined by coords, each vector is a row.
dpar = 1000
dperp = (1/3) * dpar
deltatheta = .01
thet = (-deltatheta / 2) + z*(deltatheta / .1)

d11 = dpar*sp.cos(thet)**2 + dperp*sp.sin(thet)**2
d22 = dpar*sp.sin(thet)**2 + dperp*sp.cos(thet)**2
d12 = (dpar - dperp)*sp.cos(thet)*sp.sin(thet)
d21 = (dpar - dperp)*sp.cos(thet)*sp.sin(thet)

D = sp.Matrix([
    [-1 , 0 , 0 , 0] ,
    [0 , d11 , d12 , 0 ] ,
    [0 , d21 , d22 , 0 ] ,
    [0 , 0 , 0 , dperp]
])

g = D.inv()

#INPUT: Please input if this metric should be physically viable.
physically_viable = False

#INPUT: Please input your initial 4-position
ri = [
    0 ,                                       # TIMELIKE  COMPONENT
    0 ,                                      # SPACELIKE COMPONENT 1
    0 ,                                 # SPACELIKE COMPONENT 2
    0                                         # SPACELIKE COMPONENT 3
]

#INPUT: Please input your initial 3-velocity.
vi = [
    .1 ,                                       # SPACELIKE COMPONENT 1
    .1 ,                                       # SPACELIKE COMPONENT 2
    .5                                       # SPACELIKE COMPONENT 3
]

#INPUT: Please input the amount of "integration time" and the step size.
lamTot = 100
stepsize = .1

#INPUT: Please input if you want a cloud of particles to also be integrated.
make_cloud = False

##### END OF USER INPUTS #####
