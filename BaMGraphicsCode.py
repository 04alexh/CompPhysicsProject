##### IMPORTS #####
from matplotlib.animation import FuncAnimation , PillowWriter
from matplotlib.widgets import Slider , Button
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
##### END OF IMPORTS #####


##### GRAPHICS INPUTS #####

#INPUT: Please type the filepath to the simulation data you want to visualize.
data_path = "geodesicOutput_20260923_022647.npz"

#INPUT: Please select which visualizations you would like.
static2D_plots = False
animation2D = False
animation3D = False

#INPUT: Please indicate if you want parameterization by affine parameter or timelike coordinate.
param_by_affine = False

#INPUT: If doing 3D animation, indicate if you want singularities visualized.
visualize_singularities = False

#INPUT: If doing 3D animation, indicate if you want EM fields visualized.
visualize_EM = False

#INPUT: If doing 3D animation, indicate if you want the effects of gravity visualized.
visualize_gravity = False

#INPUT: If doing 3D animation, indicate if you want a cloud of particles visualized (only works if cloud generated when npz created).
visualize_cloud = False

##### END OF GRAPHICS INPUTS #####




##### CREATE GRAPHICS #####

### Define function to make 2D plots of coordinates
def Make2DPlots(data_path):

    # Read the data
    data = np.load(data_path , allow_pickle = True)

    # Unpack
    t = data['t']
    y = data['y']
    coords = data['coords']
    is_angular = data['coordinate_info'].item()['is_angular']

    # Convert angular coordinates so they are periodic
    for i in range(3):

        if is_angular[i] == 1:
            nparray = np.array(y[i + 1] , dtype = float)
            y[i + 1] = nparray % (2 * np.pi)

        else:
            continue

    ### Make plot of variables vs. affine parameter and variables vs. timelike coord
    fig1, ax1 = plt.subplots(nrows=4, ncols=2, figsize=(14, 14))
    fig1.subplots_adjust(hspace=0.4, wspace=0.2)
    fig1.suptitle("Coordinates vs. Affine Parameter")

    # Timelike coord
    ax1[0, 0].plot(t, y[0], color="mediumaquamarine")
    ax1[0, 0].set_xlabel("λ")
    ax1[0, 0].set_ylabel(f"{coords[0]} (timelike)")
    ax1[0, 0].set_title(f"{coords[0]} vs. Affine Parameter")

    ax1[0, 1].plot(y[0], y[0], color="mediumaquamarine")
    ax1[0, 1].set_xlabel(f"{coords[0]} (timelike)")
    ax1[0, 1].set_ylabel(f"{coords[0]} (timelike)")
    ax1[0, 1].set_title(f"{coords[0]} vs. {coords[0]}")

    # Spacelike coord 1
    ax1[1, 0].plot(t, y[1], color="moccasin")
    ax1[1, 0].set_xlabel("λ")
    ax1[1, 0].set_ylabel(f"{coords[1]} (spacelike)")
    ax1[1, 0].set_title(f"{coords[1]} vs. Affine Parameter")

    ax1[1, 1].plot(y[0], y[1], color="moccasin")
    ax1[1, 1].set_xlabel(f"{coords[0]} (timelike)")
    ax1[1, 1].set_ylabel(f"{coords[1]} (spacelike)")
    ax1[1, 1].set_title(f"{coords[1]} vs. {coords[0]}")

    # Spacelike coord 2
    ax1[2, 0].plot(t, y[2], color="crimson")
    ax1[2, 0].set_xlabel("λ")
    ax1[2, 0].set_ylabel(f"{coords[2]} (spacelike)")
    ax1[2, 0].set_title(f"{coords[2]} vs. Affine Parameter")

    ax1[2, 1].plot(y[0], y[2], color="crimson")
    ax1[2, 1].set_xlabel(f"{coords[0]} (timelike)")
    ax1[2, 1].set_ylabel(f"{coords[2]} (spacelike)")
    ax1[2, 1].set_title(f"{coords[2]} vs. {coords[0]}")

    # Spacelike coord 3
    ax1[3, 0].plot(t, y[3], color="orchid")
    ax1[3, 0].set_xlabel("λ")
    ax1[3, 0].set_ylabel(f"{coords[3]} (spacelike)")
    ax1[3, 0].set_title(f"{coords[3]} vs. Affine Parameter")

    ax1[3, 1].plot(y[0], y[3], color="orchid")
    ax1[3, 1].set_xlabel(f"{coords[0]} (timelike)")
    ax1[3, 1].set_ylabel(f"{coords[3]} (spacelike)")
    ax1[3, 1].set_title(f"{coords[3]} vs. {coords[0]}")


    ### Now make phase plots of spatial coordinates
    fig2 = plt.figure(figsize = (18 , 12))
    fig2.subplots_adjust(hspace = .35 , wspace = 0.3)
    fig2.suptitle("Phase Plots of Spacelike Coordinates" , x = .45 , y = .98 , fontsize = 20)

    plot_num = 1

    for i in range(0 , 3):
        for j in range(i + 1 , 3):

            # Ignore plotting coordinate against itself
            if i == j:
                continue

            ang_i = is_angular[i]
            ang_j = is_angular[j]

            # Create condition for polar plotting
            use_polar = (ang_i + ang_j == 1)

            # Make plots
            if use_polar:
                ax = fig2.add_subplot(2 , 3 , plot_num , projection = "polar")

                if ang_i == 1:
                    theta_vals = y[i + 1]
                    r_vals = y[j + 1]
                else:
                    theta_vals = y[j + 1]
                    r_vals = y[i + 1]

                ax.scatter(theta_vals , r_vals , c = t , cmap = "plasma" , s = 6)

            else:
                ax = fig2.add_subplot(2 , 3 , plot_num)
                ax.scatter(y[i + 1] , y[j + 1] , c = t , cmap = "plasma" , s = 6)
                ax.set_xlabel(f"{coords[i + 1]}")
                ax.set_ylabel(f"{coords[j + 1]}")

            ax.set_title(f"{coords[j + 1]} vs {coords[i + 1]}")
            plot_num += 1

    # Add color map legend
    sm = plt.cm.ScalarMappable(cmap = "plasma")
    sm.set_array(t)
    fig2.colorbar(sm , ax = fig2.axes , label = "Affine Parameter")


### Define function to make 2D animations of the trajectory
def Make2DAnimation(data_path , param_by_affine = False , nframes = 1000):

    # Unpack the data
    data = np.load(data_path , allow_pickle = True)

    lam = np.asarray(data["t"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    t = y[0]
    cartesian_transform = data["cart_tran"]
    coords = data["coords"]
    vel = np.asarray(data["vel"], dtype=float)
    locvel = np.asarray(data["locvel"], dtype=float)
    ranges = data["coordinate_info"].item()["ranges"]
    is_angular = data["coordinate_info"].item()["is_angular"]

    # Make angular coordinates periodic
    for i in range(0 , 3):
        if is_angular[i] == 1:
            y[i + 1] = np.asarray(
                y[i + 1] ,
                dtype = float
            ) % (2*np.pi)

    # Interpolate data onto specified parameter
    if param_by_affine == False:

        param_frames = np.linspace(
            t[0] ,
            t[-1] ,
            nframes
        )
        coord_frames = []

        for i in range(0 , 4):

            coord_frames.append(
                np.interp(
                    param_frames ,
                    t ,
                    y[i]
                )
            )

        lam_frames = np.interp(
            param_frames ,
            t ,
            lam
        )

    else:

        param_frames = np.linspace(
            lam[0] ,
            lam[-1] ,
            nframes
        )
        coord_frames = []

        for i in range(0 , 4):

            coord_frames.append(
                np.interp(
                    param_frames ,
                    lam ,
                    y[i]
                )
            )

        t_frames = np.interp(
            param_frames ,
            lam ,
            t
        )

    # Create figure
    fig = plt.figure(figsize = (18 , 12))
    fig.subplots_adjust(hspace = .35 , wspace = 0.3)
    fig.suptitle(
        "Phase Plots of Spacelike Coordinates" ,
        x = .45 , y = .98 , fontsize = 20
    )

    # Create the phase plots
    axes = []
    plot_num = 1

    for i in range(0 , 3):
        for j in range(i + 1 , 3):

            ang_i = is_angular[i]
            ang_j = is_angular[j]

            # Use polar coords if one coord is angular
            use_polar = (ang_i + ang_j == 1)

            if use_polar:
                ax = fig.add_subplot(2 , 3 , plot_num , projection = "polar")

            else:
                ax = fig.add_subplot(2 , 3 , plot_num)
                ax.set_xlabel(f"{coords[i + 1]}")
                ax.set_ylabel(f"{coords[j + 1]}")

            ax.set_title(
                f"{coords[j + 1]} vs {coords[i + 1]}"
            )
            axes.append(ax)
            plot_num += 1

    # Find which coordinates go on each plot
    pair_indices = [
        (0 , 1) ,
        (0 , 2) ,
        (1 , 2)
    ]

    # Set axis lims
    for plot_index , (i , j) in enumerate(pair_indices):

        ax = axes[plot_index]

        xi = coord_frames[i + 1]
        xj = coord_frames[j + 1]

        ang_i = is_angular[i]
        ang_j = is_angular[j]
        use_polar = (ang_i + ang_j == 1)

        if use_polar:

            if ang_i == 1:
                theta = xi
                radius = xj
            else:
                theta = xj
                radius = xi

            ax.set_ylim(
                0 ,
                np.nanmax(radius) * 1.05
            )

        else:

            xmin = np.nanmin(xi)
            xmax = np.nanmax(xi)

            ymin = np.nanmin(xj)
            ymax = np.nanmax(xj)

            xrange = xmax - xmin
            yrange = ymax - ymin

            if xrange == 0:
                xrange = 1
            if yrange == 0:
                yrange = 1

            ax.set_xlim(
                xmin - .05 * xrange ,
                xmax + .05 * xrange
            )
            ax.set_ylim(
                ymin - .05 * yrange ,
                ymax + .05 * yrange
            )

    # Create trail and particles
    phase_trails = []
    phase_particles = []

    for plot_index , (i , j) in enumerate(pair_indices):

        ax = axes[plot_index]

        ang_i = is_angular[i]
        ang_j = is_angular[j]
        use_polar = (ang_i + ang_j == 1)

        if use_polar:

            trail , = ax.plot(
                [] ,
                [] ,
                color = "blue" ,
                lw = 1.5
            )

            particle , = ax.plot(
                [] ,
                [] ,
                color = "red" ,
                marker = "o" ,
                markersize = 7 ,
                linestyle = ""
            )

        else:

            trail , = ax.plot(
                [] ,
                [] ,
                color = "blue" ,
                lw = 1.5
            )

            particle , = ax.plot(
                [] ,
                [] ,
                color = "red" ,
                marker = "o" ,
                markersize = 7 ,
                linestyle = ""
            )

        phase_trails.append(trail)
        phase_particles.append(particle)

    # Initialize texts
    time_txt = fig.text(
        .5 ,
        .2 ,
        "" ,
        ha = "center" ,
        va = "bottom" ,
        fontsize = 14
    )

    posv_txt = fig.text(
        .5 ,
        .15 ,
        "" ,
        ha = "center" ,
        va = "bottom" ,
        fontsize = 12
    )



    # Intialize slider and button
    slider_ax = fig.add_axes(
        [.2 , .08 , .6 , .03]
    )
    slider = Slider(
        slider_ax ,
        "Frame" ,
        0 ,
        nframes - 1 ,
        valinit = 0 ,
        valstep = 1 ,
    )

    button_ax = fig.add_axes(
        [.82 , .075 , .1 , .04]
    )
    button = Button(
        button_ax ,
        "Play"
    )
    playing = False

    # Define update function
    def update(frame):

        frame = int(frame)

        for plot_index , (i , j) in enumerate(pair_indices):

            ax = axes[plot_index]

            ang_i = is_angular[i]
            ang_j = is_angular[j]
            use_polar = (ang_i + ang_j == 1)

            if use_polar:

                if ang_i == 1:
                    theta = coord_frames[i + 1]
                    radius = coord_frames[j + 1]
                else:
                    theta = coord_frames[j + 1]
                    radius = coord_frames[i + 1]

                # Update trajectory
                phase_trails[plot_index].set_data(
                    theta[:frame + 1] ,
                    radius[:frame + 1]
                )
                phase_particles[plot_index].set_data(
                    [theta[frame]] ,
                    [radius[frame]]
                )

            else:

                xvals = coord_frames[i + 1]
                yvals = coord_frames[j + 1]

                # Update trajectory
                phase_trails[plot_index].set_data(
                    xvals[:frame + 1] ,
                    yvals[:frame + 1]
                )
                phase_particles[plot_index].set_data(
                    [xvals[frame]] ,
                    [yvals[frame]]
                )

        # Update texts
        if param_by_affine == False:
            time_txt.set_text(
                rf"$\lambda = {lam_frames[frame]:.3f}"
                rf"\qquad"
                rf"{coords[0]} = {param_frames[frame]:.3f}$"
            )
            posv_txt.set_text(
                rf"${coords[1]} = {y[1][frame]:.3f} | "
                rf"{coords[2]} = {y[2][frame]:.3f} | "
                rf"{coords[3]} = {y[3][frame]:.3f}$"
                "\n"
                rf"$vFar = {vel[frame]:.3f} | "
                rf"vLoc = {locvel[frame]:.3f}$"
            )
        else:
            time_txt.set_text(
                rf"$\lambda = {lam_frames[frame]:.3f}"
                rf"\qquad"
                rf"{coords[0]} = {t_frames[frame]:.3f}$"
            )
            posv_txt.set_text(
                rf"${coords[1]} = {y[1][frame]:.3f} | "
                rf" {coords[2]} = {y[2][frame]:.3f} | "
                rf" {coords[3]} = {y[3][frame]:.3f}$"
                "\n"
                rf"$vFar = {vel[frame]:.3f} | "
                rf"vLoc = {locvel[frame]:.3f}$"
            )

        fig.canvas.draw_idle()

    # Define slider logic
    def slider_update(value):

        update(
            int(value)
        )

    slider.on_changed(slider_update)

    # Define timer logic
    timer = fig.canvas.new_timer(
        interval = 1000 / 30
    )

    def advance():

        if not playing:
            return

        current = int(
            slider.val
        )

        if current >= nframes - 1:

            set_playing(False)
            return

        slider.set_val(current+1)

    timer.add_callback(advance)

    # Define play/pause logic
    def set_playing(state):

        nonlocal playing

        playing = state

        if playing:
            button.label.set_text("Pause")
            timer.start()
        else:
            button.label.set_text("Play")
            timer.stop()

    def button_pressed(event):

        set_playing(not playing)

    button.on_clicked(
        button_pressed
    )

    # Begin at frame 0
    update(0)

    plt.show()


### Define function to help 3DAnimation visualize singularities
def plotSingGeo(
        ax , singular_coord , singular_value , coords , cartesian_transform , ranges , color , res):

    # Ignore singularities that are not real
    if not sp.sympify(singular_value).is_real:
        return None

    # Convert cartesian transforms to sympy expressions
    x_expr = sp.sympify(
        cartesian_transform[0]
    )
    y_expr = sp.sympify(
        cartesian_transform[1]
    )
    z_expr = sp.sympify(
        cartesian_transform[2]
    )

    # Find index of singular coordinate
    singular_index = None

    for i in range(len(coords)):

        if str(coords[i]) == str(singular_coord):
            singular_index = i
            break

    if singular_index is None:

        raise ValueError(
            f"Could not find singular coordinate {singular_coord}"
        )

    # Substitute singular coordinate for the singular value into the cartesian expression
    x_expr = x_expr.subs(
        coords[singular_index] ,
        singular_value
    )
    y_expr = y_expr.subs(
        coords[singular_index] ,
        singular_value
    )
    z_expr = z_expr.subs(
        coords[singular_index] ,
        singular_value
    )

    ### Determine if any coord XYZ remains a free variable even with the singular condition forced
    free_symbols = set()

    free_symbols.update(
        x_expr.free_symbols
    )
    free_symbols.update(
        y_expr.free_symbols
    )
    free_symbols.update(
        z_expr.free_symbols
    )

    # Only say a XYZ is "free" if they still contain a predefined coordinate variable
    free_coords = []

    for coord in coords:
        for symbol in free_symbols:
            if str(coord) == str(symbol):
                free_coords.append(coord)
                break


    # We only care about spatial coords, if the timelike coord remains fix it to zero
    if coords[0] in free_coords:

        free_coords.remove(
            coords[0]
        )

        x_expr = x_expr.subs(
            coords[0] ,
            0
        )
        y_expr = y_expr.subs(
            coords[0] ,
            0
        )
        z_expr = z_expr.subs(
            coords[0] ,
            0
        )

    ### Analyze the free coordinates and determine the geometry that they imply
    number_free = len(free_coords)

    if number_free == 0:

        # If no coords are free, then we have a point
        x_func = sp.lambdify(
            coords ,
            x_expr ,
            "numpy"
        )
        y_func = sp.lambdify(
            coords ,
            y_expr ,
            "numpy"
        )
        z_func = sp.lambdify(
            coords ,
            z_expr ,
            "numpy"
        )

        # Calculate XYZ using the points
        coordinate_values = []

        for coord in coords:

            if coord == coords[0]:
                coordinate_values.append(0)

            elif coord == coords[singular_index]:
                coordinate_values.append(
                    singular_value
                )

            else:
                coordinate_values.append(0)

        X = x_func(
            *coordinate_values
        )
        Y = y_func(
            *coordinate_values
        )
        Z = z_func(
            *coordinate_values
        )

        # Plot the point
        plot_object = ax.scatter(
            [X] ,
            [Y] ,
            [Z] ,
            color = color ,
            marker = "*" ,
            s = 150 ,
            alpha = 1
        )
        return plot_object

    elif number_free == 1:

        # If one coord is free, the object is a line
        free_coord = free_coords[0]
        crange = ranges[free_coord]

        free_values = np.linspace(
            crange[0] ,
            crange[1] ,
            res
        )

        # Create functions for XYZ
        x_func = sp.lambdify(
            free_coord ,
            x_expr ,
            "numpy"
        )
        y_func = sp.lambdify(
            free_coord ,
            y_expr ,
            "numpy"
        )
        z_func = sp.lambdify(
            free_coord ,
            z_expr ,
            "numpy"
        )

        X = x_func(
            free_values
        )
        Y = y_func(
            free_values
        )
        Z = z_func(
            free_values
        )

        # Plot the line
        plot_object = ax.plot(
            X ,
            Y ,
            Z ,
            color = color ,
            lw = 2
        )

        return plot_object

    elif number_free == 2:

        # If two coords are free, object is a surface
        coord_1 = free_coords[0]
        coord_2 = free_coords[1]

        crange1 = ranges[coord_1]
        crange2 = ranges[coord_2]

        values_1 = np.linspace(
            crange1[0] ,
            crange1[1] ,
            res
        )

        values_2 = np.linspace(
            crange2[0] ,
            crange2[1] ,
            res
        )

        # Create surface grid
        U , V = np.meshgrid(
            values_1 ,
            values_2
        )

        # Create XYZ functions
        x_func = sp.lambdify(
            (coord_1 , coord_2) ,
            x_expr ,
            "numpy"
        )
        y_func = sp.lambdify(
            (coord_1 , coord_2) ,
            y_expr ,
            "numpy"
        )
        z_func = sp.lambdify(
            (coord_1 , coord_2) ,
            z_expr ,
            "numpy"
        )

        X = x_func(
            U ,
            V
        )
        Y = y_func(
            U ,
            V
        )
        Z = z_func(
            U ,
            V
        )

        # Plot the surface
        plot_object = ax.plot_surface(
            X ,
            Y ,
            Z ,
            color = color ,
            alpha = .1 ,
            lw = 0
        )

        return plot_object

    else:

        # If more than of the XYZ are free, raise an error
        raise ValueError(
            "Error: Too many free variables"
        )


### Define function to make 3D animation of the trajectory
def Make3DAnimation(data_path , param_by_affine = False , nframes = 1000 ,
                    plot_sings = False , plot_em = False , plot_gravity = False , plot_deviation = False):

    # Unpack the data
    data = np.load(data_path , allow_pickle = True)

    lam = np.asarray(data["t"] , dtype = float)
    y = np.asarray(data["y"] , dtype = float)
    t = y[0]
    cartesian_transform = data["cart_tran"]
    coords = data["coords"]
    vel = np.asarray(data["vel"] , dtype = float)
    locvel = np.asarray(data["locvel"] , dtype = float)
    ranges = data["coordinate_info"].item()["ranges"]
    particle_info = data["particle_info"]

    cloudinfo = data["cloudinfo"]
    if cloudinfo.shape == ():
        cloudinfo = cloudinfo.item()
    if cloudinfo != 0:
        cloudinfo = np.asarray(cloudinfo , dtype = float) #Load particle cloud if exists

    # Ensure cartesian transform is meaningful
    assert len(cartesian_transform) == 3 , "Error: Cartesian transformation not formatted properly."

    # Generate numerical transformations
    x_expr = sp.sympify(
        cartesian_transform[0]
    )
    y_expr = sp.sympify(
        cartesian_transform[1]
    )
    z_expr = sp.sympify(
        cartesian_transform[2]
    )

    x_func = sp.lambdify(
        coords ,
        x_expr ,
        "numpy"
    )
    y_func = sp.lambdify(
        coords ,
        y_expr ,
        "numpy"
    )
    z_func = sp.lambdify(
        coords ,
        z_expr ,
        "numpy"
    )

    # Transform trajectory into animation space
    X = np.asarray(
        x_func(*y[:4]) ,
        dtype = float
    )
    Y = np.asarray(
        y_func(*y[:4]) ,
        dtype = float
    )
    Z = np.asarray(
        z_func(*y[:4]) ,
        dtype = float
    )

    # Generate frames based on uniform specified parameter
    if param_by_affine == False:

        param_frames = np.linspace(
            t[0] ,
            t[-1] ,
            nframes
        )

        # Interpolate trajectory onto timelike coordinate frames
        X_frames = np.interp(
            param_frames ,
            t ,
            X
        )
        Y_frames = np.interp(
            param_frames ,
            t ,
            Y
        )
        Z_frames = np.interp(
            param_frames ,
            t ,
            Z
        )
        lam_frames = np.interp(
            param_frames ,
            t ,
            lam
        )
        vel_frames = np.interp(
            param_frames ,
            t ,
            vel
        )
        locvel_frames = np.interp(
            param_frames ,
            t ,
            locvel
        )

    else:

        param_frames = np.linspace(
            lam[0] ,
            lam[-1] ,
            nframes
        )

        # Interpolate trajectory onto affine parameter frames
        X_frames = np.interp(
            param_frames ,
            lam ,
            X
        )
        Y_frames = np.interp(
            param_frames ,
            lam ,
            Y
        )
        Z_frames = np.interp(
            param_frames ,
            lam ,
            Z
        )
        t_frames = np.interp(
            param_frames ,
            lam ,
            t
        )
        vel_frames = np.interp(
            param_frames ,
            lam ,
            vel
        )
        locvel_frames = np.interp(
            param_frames ,
            lam ,
            locvel
        )

        lam_frames = param_frames

    # Initialize figure
    fig = plt.figure(figsize = (9 , 9))
    ax = fig.add_subplot(111 , projection = "3d")

    # If desired, plot the singularity geometries
    # If desired, plot the singularity geometries
    if plot_sings:

        sing_data = data["sing_data"].item()
        print(sing_data)

        # Determinant singularities
        determinant_singularities = set()
        if sing_data["det_singularities"] != { str(coord) : None for coord in coords }:
            for coord_name, values in sing_data["det_singularities"].items():
                if isinstance(values , sp.ConditionSet):
                    continue
                for value in values:
                    determinant_singularities.add(
                        (
                            str(coord_name),
                            str(value)
                        )
                    )

        # Coordinate singularities
        metric_singularities = set()
        if sing_data["metric_singularities"] != { str(coord) : None for coord in coords }:
            for component, singularities in sing_data["metric_singularities"].items():
                for coord_name, values in singularities.items():
                    if isinstance(values , sp.ConditionSet):
                        continue
                    for value in values:
                        metric_singularities.add(
                            (
                                str(coord_name),
                                value
                            )
                        )

        # Curvature singularities
        K_singularities = set()
        if sing_data["K_singularities"] != { str(coord) : None for coord in coords }:
            for coord_name, values in sing_data["K_singularities"].items():
                if isinstance(values , sp.ConditionSet):
                    continue
                for value in values:
                    K_singularities.add(
                        (
                            str(coord_name),
                            str(value)
                        )
                    )

        print("Plot det")
        for coord_name, value in determinant_singularities:
            plotSingGeo(
                ax=ax,
                singular_coord=coord_name,
                singular_value=value,
                coords=coords,
                cartesian_transform=cartesian_transform,
                color="orange",
                res=20,
                ranges=ranges
            )
        print("Plot met")
        for coord_name, value in metric_singularities:
            plotSingGeo(
                ax=ax,
                singular_coord=coord_name,
                singular_value=value,
                coords=coords,
                cartesian_transform=cartesian_transform,
                color="purple",
                res=20,
                ranges=ranges
            )
        print("Plot K")
        for coord_name, value in K_singularities:
            plotSingGeo(
                ax=ax,
                singular_coord=coord_name,
                singular_value=value,
                coords=coords,
                cartesian_transform=cartesian_transform,
                color="green",
                res=20,
                ranges=ranges
            )

    # Additionally, if desired we can plot the E and B fields that the particle measures
    # This calculates the E and B for each frame
    if plot_em:

        # Store E and B vectors for each frame, as well as container for the EM "acceleration"
        E_frames = np.zeros(
            (nframes , 3)
        )
        B_frames = np.zeros(
            (nframes , 3)
        )
        aem_frames = np.zeros(
            (nframes , 3)
        )

        # Create transform Jacobian to make E and B transforms easier
        cart_expr = sp.Matrix([
            x_expr ,
            y_expr ,
            z_expr
        ])
        cart_jac_expr = cart_expr.jacobian(
            sp.Matrix(coords[1:4])
        )
        cart_jac_func = sp.lambdify(
            coords ,
            cart_jac_expr ,
            "numpy"
        )

        # Unpack metric and Faraday tensor
        metric_expr = sp.Matrix(
            [
                list(row) for row in data["metric"]
            ]
        )
        F_expr = sp.Matrix(
            [
                list(row) for row in data["F"]
            ]
        )

        # Because I added this part after, the sim variable names are prolly a little confusing
        # When I made the calcFaraday function, I needed the (1,1)-tensor version for the lorentz force component of GE.
        # But then when I needed to do this viz, I have the sim now also save a sympy version of F
        # But this saved F is not the (1,1)-tensor but the (0,2)-tensor version
        # So although calcF is returning F as a numpy function (1,1)-tensor
        # The F that is saved is a sympy (0,2)-Tensor :)
        metric_func = sp.lambdify(
            coords ,
            metric_expr ,
            "numpy"
        )
        F_func = sp.lambdify(
            coords ,
            F_expr ,
            "numpy"
        )

        # Now we shall fill the E and B frames
        for frame in range(nframes):

            if param_by_affine == False:

                # Create position vector so we can calculate E and B easier
                position = np.array(
                    [
                        np.interp(
                            param_frames[frame] ,
                            t ,
                            y[i]
                        )
                        for i in range(0 , 4)
                    ] ,
                    dtype = float
                )

                # Interpolate 4-velo
                u = np.array(
                    [
                        np.interp(
                            param_frames[frame] ,
                            t ,
                            y[4 + i]
                        )
                        for i in range(0 , 4)
                    ] ,
                    dtype = float
                )

            else:

                # Same process for but other parameterization technique
                position = np.array(
                    [
                        np.interp(
                            param_frames[frame] ,
                            lam ,
                            y[i]
                        )
                        for i in range(0 , 4)
                    ] ,
                    dtype = float
                )

                u = np.array(
                    [
                        np.interp(
                            param_frames[frame] ,
                            lam ,
                            y[4 + i]
                        )
                        for i in range(0 , 4)
                    ] ,
                    dtype = float
                )

            # Evaluate the metric and Faraday tensor at given position
            g_num = np.asarray(
                metric_func(*position) ,
                dtype = float
            )
            F_num = np.asarray(
                F_func(*position) ,
                dtype = float
            )

            ### The locally measured electric field is given as
            # E_mu = F_munu * u^nu
            Ecovar = np.zeros(4)
            for mu in range(0 , 4):
                expr = 0
                for nu in range(0 , 4):
                    expr += F_num[mu , nu] * u[nu]
                Ecovar[mu] = expr

            # Normal "vectors" are written with contravariant components so we must raise index
            # E_nu * g^munu = E^mu
            ginv = np.linalg.inv(
                g_num
            )

            Econtravar = np.zeros(4)
            for mu in range(0 , 4):
                expr = 0
                for nu in range(0 , 4):
                    expr += Ecovar[nu] * ginv[mu , nu]
                Econtravar[mu] = expr


            ### The locally measured magnetic field is given as
            # B_mu = .5 * LVT_munurhosigma * u^nu * F^rhosigma where LVC is Levi-Civita tensor (LVT * Faraday = Dual Faraday)
            # First lets get a raised contravariant F:
            # F_rhosigma * g^murho * g^nusigma = F^munu
            F_contravar = np.zeros((4 , 4))
            for mu in range(0 , 4):
                for nu in range(0 , 4):
                    expr = 0
                    for rho in range(0 , 4):
                        for sigma in range(0 , 4):
                            expr += F_num[rho , sigma] * ginv[mu , rho] * ginv[nu , sigma]
                    F_contravar[mu , nu] = expr

            # Need determinant of g for LVT
            detg = np.linalg.det(
                g_num
            )
            sqrt_minus_g = np.sqrt(abs(detg))

            # Define Levi Civita symbol logic for LVT
            def levi_civita(i , j , k ,l):

                indices = [i , j , k , l]

                # If an index is the same as another, the LCS equals 0
                if len(set(indices)) < 4:
                    return 0

                # LVC returns 1 if you have an even number or inversions and returns -1 if you have an odd number of inversions
                inversions = 0
                for a in range(0 , 4):
                    for b in range(a + 1 , 4):
                        if indices[a] > indices[b]:
                            inversions += 1

                return (-1)**inversions

            # Now we are ready to calculate B_mu
            B_covar = np.zeros(4)
            for mu in range(0 , 4):
                expr = 0
                for nu in range(0 , 4):
                    for rho in range(0 , 4):
                        for sigma in range(0 , 4):

                            LVT = (
                                sqrt_minus_g * levi_civita(mu , nu , rho , sigma)
                            )
                            expr += (
                                .5
                                * LVT
                                * u[nu]
                                * F_contravar[rho , sigma]
                            )
                B_covar[mu] = expr

            # Again we want our field vectors with contravariant components
            # B_nu * g^munu = B^mu
            B_contravar = np.zeros(4)
            for mu in range(0 , 4):
                expr = 0
                for nu in range(0 , 4):
                    expr += B_covar[nu] * ginv[mu , nu]
                B_contravar[mu] = expr


            ### Now we construct the spatial basis for the particle's rest frame
            spatial_basis = []

            for spatial_index in range(1 , 4):

                basis = np.zeros(4)
                basis[spatial_index] = 1

                # Project coordinate basis vector perp to the particle's 4-velo
                # This is because the particle's 4-velo acts as its "time direction"
                # For an orthogonal coord system all spatial vectors should be perp to this direction
                # Inner product of basis and u: g_munu * basis^mu * u^nu
                inner = 0
                for mu in range(0 , 4):
                    for nu in range(0 , 4):
                        inner += g_num[mu , nu] * basis[mu] * u[nu]
                projected = (
                    basis + inner * u
                )

                # Gram-Schmidt using previous basis to make orthogonal basis
                for previous in spatial_basis:

                    projection = 0
                    for mu in range(0 , 4):
                        for nu in range(0 , 4):
                            projection += g_num[mu , nu] * previous[mu] * projected[nu]

                    projected -= (
                        projection * previous
                    )

                # Normalize basis so its fully orthonormal
                norm_square = 0
                for mu in range(0 , 4):
                    for nu in range(0 , 4):
                        norm_square += g_num[mu , nu] * projected[mu] * projected[nu]

                if norm_square <= 0:
                    continue #Skip if timelike

                projected /= np.sqrt(norm_square)
                spatial_basis.append(projected)

            ### Obtain E and B in local frame
            local_E = np.zeros(3)
            local_B = np.zeros(3)

            for i in range(0 , min(3 , len(spatial_basis))):

                basis = spatial_basis[i]

                # Get projections of Ecovar and Bcovar on the specified spatial basis vector
                # Here I will just use matrix mult cuz it makes more sense in my brain for this specific purpose
                local_E[i] = (
                    Ecovar @ basis
                )
                local_B[i] = (
                    B_covar @ basis
                )

            # Convert local spatial basis into the XYZ of the animation
            J = np.asarray(
                cart_jac_func(*position) ,
                dtype = float
            )

            E_xyz = np.zeros(3)
            B_xyz = np.zeros(3)

            for i in range(0 , min(3 , len(spatial_basis))):

                # Same projection process but now converting to the XYZ
                spatial_vector = spatial_basis[i][1:4]

                cart_vector = (
                    J @ spatial_vector
                )

                E_xyz += local_E[i] * cart_vector
                B_xyz += local_B[i] * cart_vector

            E_frames[frame] = E_xyz
            B_frames[frame] = B_xyz


            ### Lets also make an "acceleration" vector due to the E and B fields!
            particle_mass = particle_info[0]
            particle_charge = particle_info[1]

            # Raise one index of F
            # F_rhonu * g^rhomu = F^mu_nu
            F_1co1contra = np.zeros((4, 4))

            for mu in range(0, 4):
                for nu in range(0, 4):
                    expr = 0
                    for rho in range(0, 4):
                        expr += F_num[rho, nu] * ginv[mu, rho]
                    F_1co1contra[mu, nu] = expr

            # Calculate the EM 4-acceleration:
            # aEM^mu = (q/m) * F^mu_nu * u^nu
            a_em = np.zeros(4)

            for mu in range(0, 4):
                expr = 0
                for nu in range(0, 4):
                    expr += (
                            (particle_charge / particle_mass)
                            * F_1co1contra[mu, nu]
                            * u[nu]
                    )
                a_em[mu] = expr

            # We have to do all the projections
            # Project a_em onto particles local spatial frame
            local_aem = np.zeros(3)

            for i in range(0, min(3, len(spatial_basis))):

                basis = spatial_basis[i]
                expr = 0
                for mu in range(0, 4):
                    for nu in range(0, 4):
                        expr += g_num[mu, nu] * a_em[mu] * basis[nu]
                local_aem[i] = expr

            # Project local aem to spatial coords
            coord_aem = np.zeros(3)

            for i in range(0, min(3, len(spatial_basis))):
                coord_aem += (
                        local_aem[i] * spatial_basis[i][1:4]
                )

            # Convert to XYZ anim coords
            aem_xyz = (
                J @ coord_aem
            )
            aem_frames[frame] = aem_xyz

            # Get magnitudes of E, B, and aem
            E_magnitudes = np.linalg.norm(
                E_frames ,
                axis = 1
            )
            B_magnitudes = np.linalg.norm(
                B_frames ,
                axis = 1
            )
            aem_magnitudes = np.linalg.norm(
                aem_frames ,
                axis = 1
            )

            # Get positive finite values
            def get_log_range(vectors):
                magnitudes = np.linalg.norm(
                    vectors,
                    axis = 1
                )

                positive = magnitudes[
                    np.isfinite(magnitudes)
                    & (magnitudes > 0)
                ]

                if len(positive) == 0:
                    return 0 , 0

                return (
                    np.log10(np.min(positive)) ,
                    np.log10(np.max(positive))
                )

            E_log_min , E_log_max = get_log_range(E_frames)
            B_log_min , B_log_max = get_log_range(B_frames)
            aem_log_min , aem_log_max = get_log_range(aem_frames)

            # Make scaler
            def log_scale_vector(
                    vector , log_min , log_max , min_length = .5 , max_length = 7
            ):
                magnitude = np.linalg.norm(vector)

                if (magnitude <= 0 or not np.isfinite(magnitude)):
                    return np.zeros(3)

                # Convert mags to log10
                log_magnitude = np.log10(
                    magnitude
                )

                # Convert log mag to scale from 0 to 1
                if log_max > log_min:

                    fraction = (
                        log_magnitude - log_min
                    ) / (
                        log_max - log_min
                    )
                    fraction = np.clip(
                        fraction,
                        0 ,
                        1
                    )

                else:

                    fraction = .5

                # Convert to lengths
                arrow_length = (
                    min_length
                    + fraction * (max_length - min_length)
                )
                direction = vector / magnitude

                return direction * arrow_length

    # Additionally, if desired we can plot the effects of gravity
    # In theory, gravity actually causes no acceleration on the particle
    # Gravity however is the curvature of spacetime --> nonzero Riemann Tensor
    # Riemann tensor measured geodesic deviation ie in what way to geodesics diverge/converge
    # We can instead measure the magnitude and direction of such "deviations" to get a good viz of gravity
    if plot_gravity:

        # Create transform Jacobian to make future calculations easier
        cart_expr = sp.Matrix([
            x_expr,
            y_expr,
            z_expr
        ])
        cart_jac_expr = cart_expr.jacobian(
            sp.Matrix(coords[1:4])
        )
        cart_jac_func = sp.lambdify(
            coords,
            cart_jac_expr,
            "numpy"
        )

        # Unpack metric and connections
        metric_expr = sp.Matrix(
            [
                list(row) for row in data["metric"]
            ]
        )
        christoffels_expr = sp.MutableDenseNDimArray(
            data["christos"]
        )

        # Calculate symbolic Riemann tensor
        ### Define function for calculation of the Riemann tensor
        def calcRiemann(christoffels, coords, indices):

            # Initialize the Riemann tensor as a 4x4x4x4 array
            Riemann = sp.MutableDenseNDimArray.zeros(4, 4, 4, 4)

            # Calculate the Riemann tensor according to:
            # R^rho_sigmamunu = d_mu Gamma^rho_nusigma - d_nu Gamma^rho_musigma + Gamma^rho_mulambda * Gamma^lambda_nusigma - Gamma^rho_nulambda * Gamma^lambda_musigma
            for a in range(0, 4):
                for b in range(0, 4):
                    for c in range(0, 4):
                        for d in range(0, 4):  # abc are lower, d is upper
                            expr = 0
                            for e in range(0, 4):
                                expr += (
                                        christoffels[e, a, c] * christoffels[d, b, e] -
                                        christoffels[e, a, b] * christoffels[d, c, e]
                                )
                            Riemann[d, a, b, c] = (
                                    sp.diff(christoffels[d, a, c], indices[b]) -
                                    sp.diff(christoffels[d, a, b], indices[c]) +
                                    expr)


            return Riemann
        riemann_expr = calcRiemann(christoffels = christoffels_expr, coords = coords , indices = {i : coords[i] for i in range(0 , 4)})

        # Lambdify
        metric_func = sp.lambdify(
            coords,
            metric_expr,
            "numpy"
        )

        # To lambdify Riemann, have to convert to list first because sympy doesnt like lambdifiying mutabledensearrays
        riemann_expr_list = riemann_expr.tolist()
        Riemann_func = sp.lambdify(
            coords,
            riemann_expr_list,
            "numpy"
        )

        # Create storage for the neighbor particles and their deviation arrows
        neighbor_frames = np.zeros((nframes , 6 , 3))
        tidal_arrow_frames = np.zeros((nframes , 6 , 3))
        tidal_strength_frames = np.zeros((nframes , 6 ))

        # Set the proper distance separation of the particles
        tidal_sep = 1

        # Calculate tidal accel at every animation frame
        for frame in range(0 , nframes):

            # Get position and 4-velo depending on lambda or timelike param
            if param_by_affine == False:
                position = np.array(
                    [
                        np.interp(
                            param_frames[frame] ,
                            t ,
                            y[i]
                        )
                        for i in range(0 , 4)
                    ] ,
                    dtype = float
                )
                u = np.array(
                    [
                        np.interp(
                            param_frames[frame] ,
                            t ,
                            y[4 + i]
                        )
                        for i in range(0 , 4)
                    ] ,
                    dtype = float
                )

            else:
                position = np.array(
                    [
                        np.interp(
                            param_frames[frame] ,
                            lam ,
                            y[i]
                        )
                        for i in range(0 , 4)
                    ] ,
                    dtype = float
                )
                u = np.array(
                    [
                        np.interp(
                            param_frames[frame] ,
                            lam ,
                            y[4 + i]
                        )
                        for i in range(0 , 4)
                    ] ,
                    dtype = float
                )

            # Evaluate metric and Riemann tensor at this position
            g_num = np.asarray(
                metric_func(*position) ,
                dtype = float
            )
            R_num = np.asarray(
                Riemann_func(*position) ,
                dtype = float
            )

            # Construct local orthonormal basis yaya (same as EM viz method, create orthogonal vector to ut and gram schmit it)
            spatial_basis = []

            for spatial_index in range(1, 4):

                basis = np.zeros(4)
                basis[spatial_index] = 1

                # Project coordinate basis vector perp to the particle's 4-velo
                # This is because the particle's 4-velo acts as its "time direction"
                # For an orthogonal coord system all spatial vectors should be perp to this direction
                # Inner product of basis and u: g_munu * basis^mu * u^nu
                inner = 0
                for mu in range(0, 4):
                    for nu in range(0, 4):
                        inner += g_num[mu, nu] * basis[mu] * u[nu]
                projected = (
                        basis + inner * u
                )

                # Gram-Schmidt using previous basis to make orthogonal basis
                for previous in spatial_basis:

                    projection = 0
                    for mu in range(0, 4):
                        for nu in range(0, 4):
                            projection += g_num[mu, nu] * previous[mu] * projected[nu]

                    projected -= (
                            projection * previous
                    )

                # Normalize basis so its fully orthonormal
                norm_square = 0
                for mu in range(0, 4):
                    for nu in range(0, 4):
                        norm_square += g_num[mu, nu] * projected[mu] * projected[nu]

                if norm_square <= 1e-12:
                    continue  # Skip if timelike

                projected /= np.sqrt(norm_square)
                spatial_basis.append(projected)

            # If on some frame there are less than 3 basis vectors, fill neighbor info with nans to avoid crash
            if len(spatial_basis) < 3:
                neighbor_frames[frame, :, :] = np.nan
                tidal_arrow_frames[frame, :, :] = np.nan
                continue

            ### Now let's calculate the tidal accel for each local basis direction
            # Construct cartesian jacobian for ez computations
            J = np.asarray(
                cart_jac_func(*position),
                dtype=float
            )

            # Iterate through local direction
            for direction in range(0 , min(3 , len(spatial_basis))):

                basis = spatial_basis[direction]
                xi = tidal_sep * basis

                # The geodesic deviation equation is
                # (d^2x^mu/dlambda^2) = "a^mu" = -R^mu_nurhosigma * u^nu * xi^rho * u^sigma
                # What is going on here?
                # a^mu is the "acceleration" of two geodesics ie how does the distance between two geodesics change over time over time
                # xi is the distance vector between the two geodesics
                # u is the tangent vector for the particle
                # R is the riemann tensor and it quantifies the curvature, this curvature causes the deviation

                a_tidal = np.zeros(4)
                for mu in range(0 , 4):
                    expr = 0
                    for nu in range(0, 4):
                        for rho in range(0 , 4):
                            for sigma in range(0 , 4):
                                expr += (
                                    (-1)
                                    * R_num[mu , nu , rho , sigma]
                                    * u[nu]
                                    * xi[rho]
                                    * u[sigma]
                                )
                    a_tidal[mu] = expr

                # Project tidal acceleration vector onto local spatial basis (spatial tetrad parts)
                local_tidal = np.zeros(3)
                for basis_index in range(0 , min(3 , len(spatial_basis))):

                    basis2 = spatial_basis[basis_index]

                    # a_tidal DOT specific_basis = g_munu * a_tidal^mu * specifc_basis^nu = local_a in specified basis direction
                    expr = 0
                    for mu in range(0 , 4):
                        for nu in range(0, 4):
                            expr += g_num[mu , nu] * a_tidal[mu] * basis2[nu]
                    local_tidal[basis_index] = expr

                # Get strength of tidal accel
                tidal_strength = np.linalg.norm(local_tidal) #ok to use np function since tetrad is locally flat

                # Project local spatial acceleration to the coordinate spatial vector (coords of metric)
                spatial_vector = np.zeros(3)
                for basis_index in range(0 , min(3 , len(spatial_basis))):

                    spatial_vector += (
                        local_tidal[basis_index]
                        * spatial_basis[basis_index][1:4]
                    )


                ### Convert coordinate spatial vectors into the cartesian XYZ for animation
                neighborXYZ_offset = (
                    J @ basis[1:4] * tidal_sep
                )
                tidal_xyz = J @ spatial_vector

                # Positive direction particles
                plus_index = 2 * direction
                neighbor_frames[frame , plus_index] = (
                    np.array([
                        X_frames[frame] ,
                        Y_frames[frame] ,
                        Z_frames[frame]
                    ])
                    + neighborXYZ_offset
                )
                tidal_arrow_frames[frame, plus_index] = tidal_xyz
                tidal_strength_frames[frame, plus_index] = tidal_strength

                # Negative direction particles
                minus_index = 2 * direction + 1
                neighbor_frames[frame, minus_index] = (
                    np.array([
                        X_frames[frame] ,
                        Y_frames[frame] ,
                        Z_frames[frame]
                    ])
                    - neighborXYZ_offset
                )
                # Since geodesic deviation linear in xi then (-xi) reverses tidal accel
                tidal_arrow_frames[frame, minus_index] = -tidal_xyz
                tidal_strength_frames[frame, minus_index] = tidal_strength

        # Get range of magnitudes so arrow lengths can be log scale for easier viz
        positive_tidal = tidal_strength_frames[
            np.isfinite(tidal_strength_frames)
        ]
        positive_tidal = positive_tidal[positive_tidal > 0]

        log_min = np.log10(
            np.min(positive_tidal)
        )
        log_max = np.log10(
            np.max(positive_tidal)
        )

    # Additionally, if a particle cloud was simulated we can show the effects of gravity by a particle cloud
    cloud_X = None
    cloud_Y = None
    cloud_Z = None
    cloud_X_frames = None
    cloud_Y_frames = None
    cloud_Z_frames = None
    if plot_deviation:

        if cloudinfo is 0:
            raise ValueError(
                "Error: plot_deviation is true, but this .npz did not run cloud!"
            )

        cloud_X = np.zeros(
            (cloudinfo.shape[0] , cloudinfo.shape[2]),
        )
        cloud_Y = np.zeros(
            (cloudinfo.shape[0] , cloudinfo.shape[2]),
        )
        cloud_Z = np.zeros(
            (cloudinfo.shape[0] , cloudinfo.shape[2]),
        )

        # Fill particle cloud positions
        for particle_index in range(cloudinfo.shape[0]):

            cloud_X[particle_index] = np.asarray(
                x_func(*cloudinfo[particle_index , 0:4]),
                dtype=float
            )
            cloud_Y[particle_index] = np.asarray(
                y_func(*cloudinfo[particle_index , 0:4]),
                dtype=float
            )
            cloud_Z[particle_index] = np.asarray(
                z_func(*cloudinfo[particle_index , 0:4]),
                dtype=float
            )

        # Now we need to interpolate onto the correction frames
        ncloud = cloudinfo.shape[0]

        cloud_X_frames = np.zeros((ncloud , nframes))
        cloud_Y_frames = np.zeros((ncloud , nframes))
        cloud_Z_frames = np.zeros((ncloud , nframes))

        for particle_index in range(ncloud):

            if param_by_affine == False:

                cloud_X_frames[particle_index] = np.interp(
                    param_frames ,
                    t ,
                    cloud_X[particle_index]
                )
                cloud_Y_frames[particle_index] = np.interp(
                    param_frames ,
                    t ,
                    cloud_Y[particle_index]
                )
                cloud_Z_frames[particle_index] = np.interp(
                    param_frames ,
                    t ,
                    cloud_Z[particle_index]
                )

            else:

                cloud_X_frames[particle_index] = np.interp(
                    param_frames ,
                    lam ,
                    cloud_X[particle_index]
                )
                cloud_Y_frames[particle_index] = np.interp(
                    param_frames ,
                    lam ,
                    cloud_Y[particle_index]
                )
                cloud_Z_frames[particle_index] = np.interp(
                    param_frames ,
                    lam ,
                    cloud_Z[particle_index]
                )


    # Scale axes equally
    xmin , xmax = np.nanmin(X) , np.nanmax(X)
    ymin , ymax = np.nanmin(Y) , np.nanmax(Y)
    zmin , zmax = np.nanmin(Z) , np.nanmax(Z)

    max_range = max(
        xmax - xmin ,
        ymax - ymin ,
        zmax - zmin
    )

    xmid = (xmax + xmin) / 2
    ymid = (ymax + ymin) / 2
    zmid = (zmax + zmin) / 2

    ax.set_xlim(
        xmid - max_range ,
        xmid + max_range
    )
    ax.set_ylim(
        ymid - max_range ,
        ymid + max_range
    )
    ax.set_zlim(
        zmid - max_range ,
        zmid + max_range
    )

    # Add labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(
        "3D Particle Trajectory"
    )

    # Add trail for particle to trace
    trail , = ax.plot(
        [] ,
        [] ,
        [] ,
        color = "black" ,
        lw = 1 ,
        ls = "-"
    )

    # Add particle object also
    if particle_info[0] == 0:
        col = "yellow"
    elif particle_info[1] == 0:
        col = "black"
    elif particle_info[1] > 0:
        col = "gray"
    elif particle_info[1] < 0:
        col = "brown"

    particle , = ax.plot(
        [] ,
        [] ,
        [] ,
        color = col ,
        marker = "o" ,
        markersize = 8 ,
        linestyle = ""
    )

    # If desiring the E and B visualized, this will draw initial vector arrows on the particle
    E_arrow = None
    B_arrow = None
    aem_arrow = None
    if plot_em:

        E_vec_plot = log_scale_vector(
            E_frames[0] ,
            E_log_min ,
            E_log_max ,
        )
        B_vec_plot = log_scale_vector(
            B_frames[0] ,
            B_log_min ,
            B_log_max
        )
        aem_vec_plot = log_scale_vector(
            aem_frames[0] ,
            aem_log_min ,
            aem_log_max
        )

        E_arrow = ax.quiver(
            X_frames[0] ,
            Y_frames[0] ,
            Z_frames[0] ,
            E_vec_plot[0] ,
            E_vec_plot[1] ,
            E_vec_plot[2] ,
            color = "blue" ,
            length = 1 ,
            normalize = False
        )

        B_arrow = ax.quiver(
            X_frames[0] ,
            Y_frames[0] ,
            Z_frames[0] ,
            B_vec_plot[0] ,
            B_vec_plot[1] ,
            B_vec_plot[2] ,
            color = "red" ,
            length = 1 ,
            normalize = False
        )

        aem_arrow = ax.quiver(
            X_frames[0] ,
            Y_frames[0] ,
            Z_frames[0] ,
            aem_vec_plot[0] ,
            aem_vec_plot[1] ,
            aem_vec_plot[2] ,
            color = "purple" ,
            length = 1 ,
            normalize = False
        )

    # If desiring the graviational tidal effects visualized, draw the initial vector arrows on the particle
    tidal_arrows = []
    neighbor_particles = []
    if plot_gravity:

        for direction in range(0 , 6):
            nparticle = ax.plot(
                [neighbor_frames[0 , direction , 0]] ,
                [neighbor_frames[0 , direction , 1]] ,
                [neighbor_frames[0 , direction , 2]] ,
                marker = "o" ,
                markersize = 3 ,
                color = "darkolivegreen"
            )[0]
            neighbor_particles.append(nparticle)

        for direction in range(0 , 6):

            # Manually limit arrow length
            arrow_vec = tidal_arrow_frames[0 , direction]
            arrow_mag = np.linalg.norm(arrow_vec)

            if arrow_mag > 0 and np.isfinite(arrow_mag):

                min_length = .5
                max_length = 7

                logmag = np.log10(arrow_mag)

                fraction = (
                    logmag - log_min
                ) / (
                    log_max - log_min
                )

                fraction = np.clip(
                    fraction ,
                    0 ,
                    1
                )

                arrow_length = (
                    min_length
                    + fraction * (max_length - min_length)
                )

                arrow_direction = arrow_vec / arrow_mag
                arrow_vec_plot = arrow_direction * arrow_length

            else:
                arrow_vec_plot = np.zeros(3)

            arrow = ax.quiver(
                neighbor_frames[0 , direction , 0] ,
                neighbor_frames[0 , direction , 1] ,
                neighbor_frames[0 , direction , 2] ,
                arrow_vec_plot[0] ,
                arrow_vec_plot[1] ,
                arrow_vec_plot[2] ,
                color = "green" ,
                length = 1 ,
                normalize = False
            )
            tidal_arrows.append(arrow)

    # If desiring the particle cloud, draw the initial particles and trails
    deviation_particles = []
    deviation_trails = []
    if plot_deviation:

        for particle_index in range(0 , cloudinfo.shape[0]):

            deviation_particle = ax.plot(
                [cloud_X_frames[particle_index , 0]] ,
                [cloud_Y_frames[particle_index , 0]] ,
                [cloud_Z_frames[particle_index , 0]] ,
                marker = "o" ,
                markersize = 3 ,
                color = "darkolivegreen" ,
                linestyle = ""
            )[0]
            deviation_trail = ax.plot(
                [] ,
                [] ,
                [] ,
                color = "green" ,
                lw = .7 ,
            )[0]

            deviation_particles.append(deviation_particle)
            deviation_trails.append(deviation_trail)

    # Initialize counters for lambda/timelike coord and XYZ positions and 3-speed and legend
    time_txt = fig.text(
        .5 ,
        .95 ,
        "" ,
        ha = "center" ,
        va = "top" ,
        fontsize = 14
    )

    posv_txt = fig.text(
        .5 ,
        .02 ,
        "" ,
        ha = "center" ,
        va = "bottom" ,
        fontsize = 12
    )

    # sing_txt = fig.text(
    #     .05 ,
    #     .5 ,
    #     "Legend: \nParticle: Red \nDeterminant Singularity: Orange \nCoordinate Singularity: Purple \nCurvature Singularity: Green" ,
    #     ha = "left" ,
    #     va = "center" ,
    #     fontsize = 14
    # ) # This text is ugly maybe some way to make it better lol

    # Add a slider to move across frame parameter
    slider_ax = fig.add_axes(
        [.2 , .08 , .6 , .03]
    )

    slider = Slider(
        slider_ax ,
        "Frame" ,
        0 ,
        nframes - 1 ,
        valinit = 0 ,
        valstep = 1
    )

    # Add a play/pause button
    button_ax = fig.add_axes(
        [.82 , .075 , .1 , .04]
    )

    button = Button(
        button_ax ,
        "Play"
    )

    playing = False

    ### Define the animation update function
    def update(frame):

        nonlocal E_arrow , B_arrow , aem_arrow , tidal_arrows , neighbor_particles , deviation_particles , deviation_trails
        frame = int(frame)

        # Set trail
        trail.set_data(
            X_frames[:frame + 1] ,
            Y_frames[:frame + 1]
        )
        trail.set_3d_properties(
            Z_frames[:frame + 1]
        )

        # Set particle
        particle.set_data(
            [X_frames[frame]] ,
            [Y_frames[frame]]
        )
        particle.set_3d_properties(
            [Z_frames[frame]]
        )

        # Update EM arrows if wanted
        if plot_em:

            E_arrow.remove()
            B_arrow.remove()
            aem_arrow.remove()

            E_vec_plot = log_scale_vector(
                E_frames[frame] ,
                E_log_min ,
                E_log_max
            )
            B_vec_plot = log_scale_vector(
                B_frames[frame] ,
                B_log_min ,
                B_log_max
            )
            aem_vec_plot = log_scale_vector(
                aem_frames[frame] ,
                aem_log_min ,
                aem_log_max
            )

            E_arrow = ax.quiver(
                X_frames[frame] ,
                Y_frames[frame] ,
                Z_frames[frame] ,
                E_vec_plot[0] ,
                E_vec_plot[1] ,
                E_vec_plot[2] ,
                color = "blue" ,
                length = 1 ,
                normalize = False
            )
            B_arrow = ax.quiver(
                X_frames[frame] ,
                Y_frames[frame] ,
                Z_frames[frame] ,
                B_vec_plot[0] ,
                B_vec_plot[1] ,
                B_vec_plot[2] ,
                color = "red" ,
                length = 1 ,
                normalize = False
            )
            aem_arrow = ax.quiver(
                X_frames[frame] ,
                Y_frames[frame] ,
                Z_frames[frame] ,
                aem_vec_plot[0] ,
                aem_vec_plot[1] ,
                aem_vec_plot[2] ,
                color = "purple" ,
                length = 1 ,
                normalize = False
            )

        # Update gravity arrows and particles if wanted
        if plot_gravity:

            for direction in range(0 , 6):
                neighbor_particles[direction].set_data(
                    [neighbor_frames[frame , direction , 0]] ,
                    [neighbor_frames[frame , direction , 1]]
                )
                neighbor_particles[direction].set_3d_properties(
                    [neighbor_frames[frame , direction , 2]]
                )

            for arrow in tidal_arrows:
                arrow.remove()

            tidal_arrows = []

            for direction in range(0 , 6):

                # Manually limit arrow length so they are visible but change in length due to grav strength
                arrow_vec = tidal_arrow_frames[frame , direction]
                arrow_mag = np.linalg.norm(arrow_vec)

                if arrow_mag > 0 and np.isfinite(arrow_mag):
                    min_length = .5
                    max_length = 7

                    logmag = np.log10(arrow_mag)

                    fraction = (
                        logmag - log_min
                    ) / (
                        log_max - log_min
                    )

                    fraction = np.clip(
                        fraction ,
                        0 ,
                        1
                    )

                    arrow_length = (
                        min_length
                        + fraction * (max_length - min_length)
                    )

                    arrow_direction = arrow_vec / arrow_mag
                    arrow_vec_plot = arrow_direction * arrow_length

                else:

                    arrow_vec_plot = np.zeros(3)

                arrow = ax.quiver(
                    neighbor_frames[frame , direction , 0] ,
                    neighbor_frames[frame , direction , 1] ,
                    neighbor_frames[frame , direction , 2] ,
                    arrow_vec_plot[0] ,
                    arrow_vec_plot[1] ,
                    arrow_vec_plot[2] ,
                    color = "green" ,
                    length = 1 ,
                    normalize = False
                )
                tidal_arrows.append(arrow)

        # Update cloud particles and trails if wanted
        if plot_deviation:

            for particle_index in range(0 , len(deviation_particles)):

                deviation_particles[particle_index].set_data(
                    [cloud_X_frames[particle_index , frame]] ,
                    [cloud_Y_frames[particle_index , frame]]
                )
                deviation_particles[particle_index].set_3d_properties(
                    [cloud_Z_frames[particle_index , frame]]
                )

            for particle_index in range(0 , len(deviation_trails)):

                deviation_trails[particle_index].set_data(
                    cloud_X_frames[particle_index , :frame + 1] ,
                    cloud_Y_frames[particle_index , :frame + 1]
                )
                deviation_trails[particle_index].set_3d_properties(
                    cloud_Z_frames[particle_index , :frame + 1]
                )

        # Set time_txt
        time_txt.set_text(
            rf"$\lambda = {lam_frames[frame]:.3f}"
            rf"\qquad"
            rf"{coords[0]} = {param_frames[frame]:.3f}$"
        )

        # Set posv_txt
        posv_txt.set_text(
            rf"$X = {X_frames[frame]:.3f}$ | "
            rf"$Y = {Y_frames[frame]:.3f}$ | "
            rf"$Z = {Z_frames[frame]:.3f}$"
            "\n"
            rf"$vCoord = {vel_frames[frame]:.3f}$ | "
            rf"$vLoc = {locvel_frames[frame]:.3f}$"
        )

        # Redraw the canvas
        fig.canvas.draw_idle()


    ### Define slider logic
    def slider_update(value):

        update(int(value))

    slider.on_changed(
        slider_update
    )


    ### Define animation timer logic
    timer = fig.canvas.new_timer(
        interval = 1000 / 30
    )

    def advance():

        if not playing:
            return

        current = int(
            slider.val
        )

        if current >= nframes - 1:
            set_playing(False)
            return

        slider.set_val(
            current + 1
        )

    timer.add_callback(
        advance
    )


    ### Define play/pause logic
    def set_playing(state):

        nonlocal playing

        playing = state

        if playing:
            button.label.set_text(
                "Pause"
            )
            timer.start()
        else:
            button.label.set_text(
                "Play"
            )
            timer.stop()

    def button_clicked(event):
        set_playing(
            not playing
        )

    button.on_clicked(
        button_clicked
    )


    # Set beginning frame
    update(0)
    plt.show()


### Define function to make a simulation write-out screen
def MakeSimWriteOut(data_path):

    # Unpack the data
    data = np.load(data_path, allow_pickle=True)

    lam = np.asarray(data["t"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    t = y[0]
    cartesian_transform = data["cart_tran"]
    coords = data["coords"]
    vel = np.asarray(data["vel"], dtype=float)
    locvel = np.asarray(data["locvel"], dtype=float)
    ranges = data["coordinate_info"].item()["ranges"]

    # Create figure
    fig = plt.figure(figsize=(20 , 20))

    # Plot texts
    name = fig.text(
        .5 ,
        .99 ,
        f"{data_path}" ,
        ha = "center" ,
        va = "top" ,
        fontsize = 25
    )

    coordinates = fig.text(
        .05 ,
        .95 ,
        f"Simulation Coordinates: {coords}" ,
        ha = "left" ,
        va = "top" ,
        fontsize = 15
    )

    metric = fig.text(
        .05 ,
        .93 ,
        f"Simulation Metric: \n {data["metric"]}" ,
        ha = "left" ,
        va = "top" ,
        fontsize = 15
    )

    fourpot = fig.text(
        .05 ,
        .81 ,
        f"Simulation 4-Potential: \n {data["apot"]}" ,
        ha = "left" ,
        va = "top" ,
        fontsize = 15
    )

    initpos = fig.text(
        .05 ,
        .69 ,
        f"Simulation Initial Position: \n {data["init"][0]}" ,
        ha = "left" ,
        va = "top" ,
        fontsize = 15
    )

    initvel = fig.text(
        .05 ,
        .64 ,
        f"Simulation Initial Velocity: \n {data["init"][1]}" ,
        ha = "left" ,
        va = "top" ,
        fontsize = 15
    )

    partdat = fig.text(
        .05 ,
        .59 ,
        f"Simulation Particle Data: \n Mass {data["particle_info"][0]} \n Charge {data["particle_info"][1]}" ,
        ha = "left" ,
        va = "top" ,
        fontsize = 15
    )

    singtxt = fig.text(
        .99 ,
        .1 ,
        f"Singularity Information: \n {data["sing_data"].item()}" ,
        ha = "right" ,
        va = "top" ,
        fontsize = 10
    )

    cloudtxt = fig.text(
        .99 ,
        .15 ,
        f"Has integrated perturbed paths: \n {data["cloudinfo"] is 0}" ,
        ha = "right" ,
        va = "top" ,
        fontsize = 10
    )



    plt.show()


### Run graphics
if static2D_plots:
    Make2DPlots(data_path=data_path)
if animation2D:
    Make2DAnimation(data_path=data_path, param_by_affine=param_by_affine, nframes=1000)
if animation3D:
    Make3DAnimation(data_path=data_path, param_by_affine=param_by_affine, nframes=1000,
                    plot_sings=visualize_singularities , plot_em=visualize_EM , plot_gravity=visualize_gravity , plot_deviation=visualize_cloud)

MakeSimWriteOut(data_path=data_path)




