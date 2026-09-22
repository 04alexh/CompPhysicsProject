##### IMPORTS #####
from datetime import datetime
from scipy.integrate import solve_ivp
import os
import numpy as np
import sympy as sp
##### ENDS OF IMPORTS #####

##### USER INPUTS #####

#INPUT: Please define the symbols of your coordinate system with the timelike coordinate being the FIRST symbol.
coords = sp.symbols("t r theta phi")

for c in coords: #DO NOT TOUCH THIS LOOP
    globals()[str(c)] = c

#INPUT: Please set if the spacelike coordinates represent a "length" or an "angle" where 1=angular and 0=length.
is_angular = [
    0 ,                                         # SPACELIKE COORD 1
    1 ,                                         # SPACELIKE COORD 2
    1                                           # SPACELIKE COORD 3
]

#INPUT: Please set the plotting ranges of the coordinates (only important for visualizing singularities).
coordinate_ranges = [
    (0 , 1000) ,                                # TIMELIKE  COORD
    (0 , 20) ,                                  # SPACELIKE COORD 1
    (0 , np.pi) ,                               # SPACELIKE COORD 2
    (0 , 2 * np.pi)                             # SPACELIKE COORD 3
]

#INPUT: Please define the transformation of your spatial coordinates to orthonormal cartesian coordinates.
cartesian_transformation = [
    r * sp.sin(theta) * sp.cos(phi) ,           # X = ?
    r * sp.sin(theta) * sp.sin(phi) ,           # Y = ?
    r * sp.cos(theta)                           # Z = ?
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
g = sp.Matrix([
    [-(1-(10/r)) , 0 , 0 , 0] ,                # TIMELIKE  ROW
    [0 , 1/(1-(10/r)) , 0 , 0] ,               # SPACELIKE ROW 1
    [0 , 0 , r**2 , 0] ,                       # SPACELIKE ROW 2
    [0 , 0 , 0 , r**2 * sp.sin(theta)**2]      # SPACELIKE ROW 3
])

#INPUT: Please input if this metric should be physically viable.
physically_viable = True

#INPUT: Please input your initial 4-position
ri = [
    0 ,                                       # TIMELIKE  COMPONENT
    20 ,                                      # SPACELIKE COMPONENT 1
    np.pi/2 ,                                 # SPACELIKE COMPONENT 2
    0                                         # SPACELIKE COMPONENT 3
]

#INPUT: Please input your initial 3-velocity.
vi = [
    0 ,                                       # SPACELIKE COMPONENT 1
    0 ,                                       # SPACELIKE COMPONENT 2
    .01                                       # SPACELIKE COMPONENT 3
]

#INPUT: Please input the amount of "integration time" and the step size.
lamTot = 100
stepsize = .1

##### END OF USER INPUTS #####




##### FUNCTIONS AND VARIABLES #####

### Define some useful variables
# Is particle massive?
massless_particle = particle_mass == 0

# What is the charge to mass ratio?
if massless_particle == False:
    qmrat = particle_charge / particle_mass
else:
    qmrat = None

# Create dictionary of coordinate information
coordinate_info = {
    "coords" : coords ,
    "ranges" : {
        str(coords[0]) : coordinate_ranges[0],
        str(coords[1]) : coordinate_ranges[1],
        str(coords[2]) : coordinate_ranges[2],
        str(coords[3]) : coordinate_ranges[3]
    } ,
    "is_angular" : is_angular
}


### Define function for calculation of the Christoffel symbols
def calcChristoffel(g , indices , coords):

    # Find g inverse
    ginv = g.inv()

    # Initialize connection coefficients as a 4x4x4 array of zeros
    christoffel = sp.MutableDenseNDimArray.zeros(4 , 4 , 4)

    # Calculate coefficients using the Levi-Civita connection:
    # Γ^m_np = .5 * g^ms * (deriv_p[g_sn] + deriv_n[g_sp] - deriv_s[g_np])
    for mu in range(0 , 4):
        for nu in range(0 , 4):
            for rho in range(0 , 4):
                expr = 0
                for sigma in range(0 , 4):
                    expr += .5 * ginv[mu , sigma] * (
                        sp.diff(g[sigma , nu] , indices[rho]) +
                        sp.diff(g[sigma , rho] , indices[nu]) -
                        sp.diff(g[nu , rho] , indices[sigma])
                    )

                christoffel[mu , nu , rho] = expr

    # Convert christoffel into a function that can sub values into the variables if required
    Gamma_Function = [[[
        sp.lambdify(coords , christoffel[mu , nu , rho] , "numpy") for rho in range(0 , 4)]
            for nu in range(0 , 4)]
            for mu in range(0 , 4)]

    return (Gamma_Function , ginv , christoffel)


### Define function for calculation of the Faraday tensor
def calcFaraday(Apot , indices , coords , ginv):

    # Initialize the Faraday tensor as a 4x4 array of zeros
    F_2covar = sp.MutableDenseNDimArray.zeros(4 , 4)

    # Calculate Faraday components according to:
    # F_munu = deriv_mu[A_nu] - deriv_nu[A_mu]
    for mu in range(0 , 4):
        for nu in range(0 , 4):
            F_2covar[mu , nu] = (
                sp.diff(Apot[nu] , indices[mu]) -
                sp.diff(Apot[mu] , indices[nu])
            )

    # Now we must raise an index of the Faraday tensor since Lorentz force requires
    # LorentzForce ~ F^mu_nu * u^nu
    F_1covar1contravar = sp.MutableDenseNDimArray.zeros(4 , 4)

    for mu in range(0 , 4):
        for nu in range(0 , 4):
            expr = 0
            for alpha in range(0 , 4):
                expr += (
                    ginv[mu , alpha] * F_2covar[alpha , nu]
                )
            F_1covar1contravar[mu , nu] = expr

    # Make Faraday tensor a function for future use
    F_Function = [[
        sp.lambdify(coords , F_1covar1contravar[mu , nu] , "numpy") for nu in range(0 , 4)]
        for mu in range(0 , 4)]

    return (F_Function , F_2covar) #Save the 2covar in case we want to viz the EM fields in animation


### Define function for calculation of the Riemann tensor
def calcRiemann(christoffels , coords , indices):

    # Initialize the Riemann tensor as a 4x4x4x4 array
    Riemann = sp.MutableDenseNDimArray.zeros(4 , 4 , 4 , 4)

    # Calculate the Riemann tensor according to:
    # R^rho_sigmamunu = d_mu Gamma^rho_nusigma - d_nu Gamma^rho_musigma + Gamma^rho_mulambda * Gamma^lambda_nusigma - Gamma^rho_nulambda * Gamma^lambda_musigma
    for a in range (0 , 4):
        for b in range (0 , 4):
            for c in range(0 , 4):
                for d in range(0 , 4): #abc are lower, d is upper
                    expr = 0
                    for e in range(0 , 4):
                        expr += (
                            christoffels[e , a , c] * christoffels[d , b , e] -
                            christoffels[e , a , b] * christoffels[d , c , e]
                        )
                    Riemann[d , a , b , c] = (
                            sp.diff(christoffels[d , a , c] , indices[b]) -
                            sp.diff(christoffels[d , a , b] , indices[c]) +
                            expr )

    # Make tensor have a functional version too
    Rie_Function = [[[[
        sp.lambdify(coords , Riemann[d , a , b , c] , "numpy") for d in range(0 , 4)]
        for c in range(0 , 4)]
        for b in range(0 , 4)]
        for a in range(0 , 4)]

    return (Riemann , Rie_Function)


### Define function for calculation of the Ricci tensor
def calcRicciT(Riemann , coords):

    # Calculate Ricci tensor according to:
    # R_munu = R^a_muanu
    RicciT = sp.MutableDenseNDimArray.zeros(4, 4)
    for mu in range(0, 4):
        for nu in range(0, 4):
            expr = 0
            for a in range(0, 4):
                expr += (
                    Riemann[a, mu, a, nu]
                )
            RicciT[mu, nu] = expr

    # Create functional version
    RicciT_Function = [[
        sp.lambdify(coords , RicciT[mu , nu] , "numpy") for nu in range(0 , 4)]
        for mu in range(0 , 4)]

    return (RicciT , RicciT_Function)


### Define function for calculation of the Ricci scalar
def calcRicciS(RicciT , ginv , coords):

    # Calculate Ricci scalar according to:
    # R = R^mu_mu = R_munu * g^munu
    RicciS = 0
    for mu in range(0, 4):
        for nu in range(0, 4):
            RicciS += RicciT[mu, nu] * ginv[mu, nu]

    # Create functional version
    RicciS_Function = sp.lambdify(coords , RicciS , "numpy")

    return (RicciS , RicciS_Function)


### Define function for calculation of the Kretschmann scalar
def calcKret(Riemann , g , ginv , coords):

    # Create a fully covariant RiemannT
    # R^e_abc * g_ed = R_abcd
    R_fullcovar = sp.MutableDenseNDimArray.zeros(4 , 4 , 4 , 4)
    for d in range(0 , 4):
        for a in range(0 , 4):
            for b in range(0 , 4):
                for c in range(0 , 4):
                    expr = 0
                    for e in range(0 , 4):
                        expr += Riemann[e , a , b , c] * g[e , d]
                    R_fullcovar[a , b , c , d] = expr

    # Create a fully contravariant RiemannT
    # R^d_fhi * g^fa * g^ib * g^hc = R^abcd
    R_fullcontra = sp.MutableDenseNDimArray.zeros(4 , 4 , 4 , 4)
    for d in range(0 , 4):
        for a in range(0 , 4):
            for b in range(0 , 4):
                for c in range(0 , 4):
                    expr = 0
                    for f in range(0 , 4):
                        for i in range(0 , 4):
                            for h in range(0 , 4):
                                for k in range(0 , 4):
                                    expr += (
                                        ginv[a , f] * ginv[b , i] * ginv[c , h] * ginv[d , k] *
                                        R_fullcovar[f , i , h , k]
                                    )
                    R_fullcontra[a , b , c , d] = expr

    # Contract to get the scalar
    Kret = 0
    for a in range(0 , 4):
        for b in range(0 , 4):
            for c in range(0 , 4):
                for d in range(0 , 4):
                    Kret += R_fullcovar[a , b , c , d] * R_fullcontra[a , b , c , d]

    # Create functional version
    Kret = sp.factor(
        sp.cancel(
            sp.simplify(Kret)
        )
    )
    Kret_Function = sp.lambdify(coords , Kret , "numpy")

    return (Kret , Kret_Function)


### Define function to determine the initial timelike component of the 4-velocity
def findUt(is_null , r , v , g , indices):

    # There are two normalization conditions for 4-velocity:
    # 1. If particle is massive, u^2 = -1
    # 2. If particle is massless, u^2 = 0

    if is_null:
        condition = 0
    else:
        condition = -1

    # Sub in initial positions into the metric
    subs_dict = {
        indices[0] : r[0] ,
        indices[1] : r[1] ,
        indices[2] : r[2] ,
        indices[3] : r[3]
    }
    numeric_g = g.subs(subs_dict)

    # Initialize the unknown ut
    ut = sp.symbols('ut' , positive = True)
    u = [ut , v[0] , v[1] , v[2]]

    # Take the inner product of u with itself according to
    # u^2 = g_munu * u^mu * u^nu
    expr = 0
    for mu in range(0 , 4):
        for nu in range(0 , 4):
            expr += u[mu] * u[nu] * numeric_g[mu, nu]

    # Make the experession an equation by setting it equal to condition
    inner_product_constraint = sp.Eq(expr , condition)

    # Solve for ut
    time_part = sp.solve(inner_product_constraint , ut)

    # Check if a solution was found; if not, indicate singularity failure
    if len(time_part) == 0:
        print("Error in normalizing initial ut!")
        return np.nan

    return float(time_part[0])


### Define function to verify if given metric is physically allowed
def verifyMetric(g , ri , indices):

    # A valid spacetime metric is
    # 1. Can be represented as a 4x4 matrix
    # 2. Symmetric and real
    # 3. Has a non-zero determinant
    # 4. Has a Lorentzian signature (this program shall assume 1 negative eigenvalue and 3 positive eigenvalues for a (-,+,+,+) signature)

    # Is this metric symmetric?
    assert g.is_symmetric() , "Error: This metric is not symmetric!"

    # Is this metric nonsingular?
    assert g.det() != 0 , "Error: This metric is singular!"

    # Does this metric have the correct signature?
    check_negative = False

    subs_dict = {
        indices[0] : ri[0] ,
        indices[1] : ri[1] ,
        indices[2] : ri[2] ,
        indices[3] : ri[3]
    }

    g_initial = g.subs(subs_dict)
    eigs = g_initial.eigenvals()

    for (key , value) in eigs.items():

        if key < 0:
            if value == 1:
                check_negative = True

    assert check_negative , "Error: This metric does not have the correct signature!"


### Define function to detect coordinate and "real" singularities in the metric
def findSings(g , ginv , christoffels , indices , coords):

    ### Check One: Do any coordinate values cause g to be singular
    # Find det(g)
    detg = sp.factor(g.det())

    # Solve for when det(g) = 0
    det_singularities = {}

    for coord in coords:

        try:
            solutions = sp.solve(
                sp.Eq(detg , 0) ,
                coord
            )

            if solutions:
                det_singularities[str(coord)] = solutions

        except Exception:
            det_singularities[str(coord)] = (
                "No solution to det(g) = 0"
            )


    ### Check Two: Do any components of g have possible areas where they are undefined?
    metric_singularities = {}

    for mu in range(0 , 4):
        for nu in range(0 , 4):

            expr = sp.factor(g[mu , nu])

            # Ignore if g_munu = 0
            if expr == 0:
                continue

            # For each coordinate see if singular points exist
            component_singularities = {}
            for coord in coords:

                try:
                    singular_points = sp.singularities(
                        expr ,
                        coord
                    )

                    if singular_points != sp.EmptySet:
                        component_singularities[str(coord)] = (
                            singular_points
                        )

                except Exception:
                    # Continue if component has no singular value
                    continue

            # Label singularities by correct index
            if component_singularities:

                metric_singularities[(mu , nu)] = (
                    component_singularities
                )


    ### Check Three: Check for real singularities using a curvature invariant
    # Calculate Riemann tensor
    (R , R_Function) = calcRiemann(christoffels = christoffels , coords = coords , indices = indices)

    # Calculate Kretschmann scalar
    (K , K_Function) = calcKret(Riemann = R , g = g , ginv = ginv , coords = coords)

    # Calculate singularities of K
    K_singularities = {}

    for coord in coords:

        try:
            singular_points = sp.singularities(
                K ,
                coord
            )

            if singular_points != sp.EmptySet:
                K_singularities[str(coord)] = (
                    singular_points
                )

        except Exception:
            pass


    ### Finally return as large dictionary
    return {
        "det_singularities": det_singularities ,
        "metric_singularities": metric_singularities ,
        "K_singularities": K_singularities
    }


### Define function to calculate the coordinate 3-speed of the particle
def find3speed(lam , y , g , indices):

    # Initialize speed list
    farvels = []

    # Iterate for each step in affine param
    for i in range(len(lam)):

        # Define 4pos and 4vel
        pos = y[:4 , i]
        u = y[4:8 , i]

        # Find metric at this position
        subs_dict = {
            indices[0] : pos[0] ,
            indices[1] : pos[1] ,
            indices[2] : pos[2] ,
            indices[3] : pos[3]
        }

        numeric_g = np.asarray(
            g.subs(subs_dict) ,
            dtype = float
        )

        # The coordinate three speed is given as v^2 = g_ij * (dx^i/dt) * (dx^j/dt) where t is timelike coord
        # But dx^i/dt = dx^a/dlam * dlam/dt = u[a] * (u[0])^-1
        expr = 0
        for i in range(1 , 4):
            for j in range(1 , 4):
                expr += numeric_g[i , j] * (u[i] * u[0]**-1) * (u[j] * u[0]**-1)
        farvels.append(expr**.5)

    return farvels


### Define function to calculate the local 3-speed of the particle
def findLoc3speed(lam , y , g , indices):

    # Initialize velo array
    nearvels = []

    # Iterate for each affine parameter step
    for k in range(len(lam)):

        # Obtain the particle 4position and 4velocity
        pos = y[:4 , k]
        u = y[4:8 , k]

        # Obtain the metric at this position
        subs_dict = {
            indices[0] : pos[0] ,
            indices[1] : pos[1] ,
            indices[2] : pos[2] ,
            indices[3] : pos[3]
        }

        numeric_g = np.asarray(
            g.subs(subs_dict) ,
            dtype = float
        )

        # The local 3 speed is measured by a stationary observer whose 4-velo is [1,0,0,0] = uobs
        # Thus uobs^2 = -1 = g_munu * u[mu] * u[nu] = g_tt * u[t]^2
        g00 = numeric_g[0 , 0]
        uobs = np.array([
            1 / (-1 * g00)**.5 ,
            0 ,
            0 ,
            0
        ])

        # Lower the index of the 4velocity
        # u_mu = g_munu * u^nu
        uobs_lower = np.zeros(4)
        for mu in range(0 , 4):
            expr = 0
            for nu in range(0 , 4):
                expr += numeric_g[mu , nu] * uobs[nu]
            uobs_lower[mu] = expr

        # Contracting the observer 4-velo with the particle 4-velo gives a measure of energy; in geometrized units just gives Lorentz factor
        # We can safely just do a simple dot product since lowering the index accounted for the metric
        lor_fac = -1 * np.dot(
            uobs_lower ,
            u
        )

        # Find local speed (lor_fac = sqrt(1 / 1-v^2))
        v_local = np.sqrt(
            1 - (1 / lor_fac**2))

        nearvels.append(v_local)

    return nearvels


### Define function to calculate the coordinate acceleration of the particle
def findAccel(sol , change_of_state):

    # Initialize the acceleration container
    accelerations = np.zeros((4 , len(sol.t)))

    # Calculate using change_of_state
    for i in range(len(sol.t)):

        state = sol.y[: , i]

        derivatives = change_of_state(
            sol.t[i] ,
            state
        )

        # Only take the "acceleration" terms
        accelerations[:, i] = derivatives[4:8]

    return accelerations


### Define function to integrate the geodesic equations
def integrateGeodesics(g , F , Fviz , qmrat , christoffels , christoffelsviz , indices , cur_pos , cur_vel , sing_data , coordinate_info , runtime , stepsize):

    # Define the initial state vector
    initial_state = [
        cur_pos[0] ,
        cur_pos[1] ,
        cur_pos[2] ,
        cur_pos[3] ,
        cur_vel[0] ,
        cur_vel[1] ,
        cur_vel[2] ,
        cur_vel[3] ,
    ]

    # Define the differential equation
    def change_of_state(lam , state):

        # Sub in values into Christoffels
        subs_dict = {
            indices[0] : state[0] ,
            indices[1] : state[1] ,
            indices[2] : state[2] ,
            indices[3] : state[3] ,
        }

        numeric_christos = np.zeros((4 , 4 , 4))
        for mu in range(0 , 4):
            for nu in range(0 , 4):
                for rho in range(0 , 4):
                    numeric_christos[mu , nu , rho] = christoffels[mu][nu][rho](state[0] , state[1] , state[2] , state[3])

        # Unpack state for organization
        pos = [state[0] , state[1] , state[2] , state[3]]
        u = [state[4] , state[5] , state[6] , state[7]]

        # Get metric at this position
        numeric_g = g.subs(subs_dict)

        # Print 4-velo magnitude at this position for bookkeeping
        expr = 0
        for mu in range(0 , 4):
            for nu in range(0 , 4):
                expr += numeric_g[mu , nu] * u[mu] * u[nu]
        print(f"4-Velocity magnitude is {expr}")

        # Construct the acceleration components of the state change using the geodesic equation
        # a[mu] = (-Γ^mu_alphabeta * u[alpha][beta]) + (q/m * F^mu_nu * u[nu])
        accel_list = []

        if massless_particle == True or particle_charge == 0:
            for k in range(0 , 4):
                accel = 0
                for mu in range(0 , 4):
                    for nu in range(0 , 4):
                        accel += -1 * numeric_christos[k , mu , nu] * u[mu] * u[nu]
                accel_list.append(accel)

        else:

            # Get Faraday tensor at this position
            numeric_F = np.zeros((4 , 4))
            for mu in range(0 , 4):
                for nu in range(0 , 4):
                    numeric_F[mu , nu] = F[mu][nu](state[0] , state[1] , state[2] , state[3])

            # Do geodesic eq
            for k in range(0 , 4):
                accel = 0
                for mu in range(0 , 4):
                    for nu in range(0 , 4):
                        accel += (-1 * numeric_christos[k , mu , nu] * u[mu] * u[nu]) + (qmrat * numeric_F[k , nu] * u[nu])
                accel_list.append(accel)

        # Return the change in state vector
        return [
            u[0] ,
            u[1] ,
            u[2] ,
            u[3] ,
            accel_list[0] ,
            accel_list[1] ,
            accel_list[2] ,
            accel_list[3] ,
        ]

    # Run radau integration
    sol = solve_ivp(
        fun = change_of_state ,
        t_span = (0 , runtime) ,
        y0 = initial_state ,
        method = 'Radau' ,
        max_step = stepsize
    )

    # Make 3speed array
    spatvel = find3speed(lam = sol.t , y = sol.y , g = g , indices = indices)

    # Make local 3speed array
    locvel = findLoc3speed(lam = sol.t , y = sol.y , g = g , indices = indices)

    # Make "acceleration" array
    accelerations = findAccel(sol = sol , change_of_state = change_of_state)

    # Create filename for .npz file of sim
    now = datetime.now()
    file_timestamp = now.strftime("%Y%m%d_%H%M%S")
    FILENAME = os.path.join(f"geodesicOutput_{file_timestamp}.npz")

    # Save .npz
    np.savez(
        FILENAME ,
        t = sol.t ,
        y = sol.y ,
        a = accelerations ,
        christos = christoffelsviz ,
        vel = spatvel ,
        locvel = locvel ,
        coords = [str(c) for c in coords] ,
        coordinate_info = coordinate_info ,
        cart_tran = cartesian_transformation ,
        metric = g ,
        apot = Apot ,
        F = Fviz ,
        particle_info = (particle_mass , particle_charge) ,
        sing_data = sing_data ,
        init = [cur_pos , cur_vel]
    )

    # Return solution
    return sol

##### END OF FUNCTIONS AND VARIABLES #####




##### RUN SIM #####
### Check if metric is 4x4 and has the correct number of coordinates
assert len(coords) == 4 , "Error: Not enough/too many coordinates!"
assert g.shape == (4,4) , "Error: Metric is not 4x4!"

### Create an indices dictionary
indices = {i : coords[i] for i in range(0 , 4)}


### Check for physically viable metric if required
if physically_viable:
    verifyMetric(g = g , ri = ri , indices = indices)
    print("Metric is physically viable! Continuing...")


### Calculate inverse metric and Christoffel symbols
(Connections , ginv , ConnectionsSym) = calcChristoffel(g = g , indices = indices , coords = coords)


### Analyze metric for singularities
singularity_data = findSings(g = g , ginv = ginv , christoffels = ConnectionsSym , indices = indices , coords = coords)


### Find initial ut
ut_init = findUt(is_null = massless_particle , r = ri , v = vi , g = g , indices = indices)
ui = [ut_init , vi[0] , vi[1] , vi[2]]


### Calculate Faraday tensor
F , Fviz = calcFaraday(Apot = Apot , indices = indices , coords = coords , ginv = ginv)


### Integrate the geodesic equations
sol = integrateGeodesics(g = g , F = F , Fviz = Fviz , qmrat = qmrat ,
                         christoffels = Connections , christoffelsviz = ConnectionsSym ,
                         indices = indices , cur_pos = ri , cur_vel = ui ,
                         sing_data = singularity_data , coordinate_info = coordinate_info ,
                         runtime = lamTot , stepsize = stepsize)
print(sol.message)
print(sol.status)

##### END OF RUN SIM #####







