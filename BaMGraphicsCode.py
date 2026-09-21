##### IMPORTS #####
from matplotlib.animation import FuncAnimation , PillowWriter
from matplotlib.widgets import Slider , Button
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
##### END OF IMPORTS #####




##### GRAPHICS INPUTS #####

#INPUT: Please type the filepath to the simulation data you want to visualize.
data_path = "geodesicOutput_20260920_232638.npz"

#INPUT: Please select which visualizations you would like.
static2D_plots = False
animation2D = False
animation3D = True

#INPUT: Please indicate if you want parameterization by affine parameter or timelike coordinate.
param_by_affine = False

#INPUT: If doing 3D animation, indicate if you want singularities visualized.
visualize_singularities = True

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
def Make3DAnimation(data_path , param_by_affine = False , nframes = 1000 , plot_sings = False):

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

        '''frame_indices = np.searchsorted(
            t ,
            param_frames
        )

        frame_indices = np.clip(
            frame_indices ,
            0 ,
            len(t) - 1
        )'''

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

        '''frame_indices = np.searchsorted(
            lam ,
            param_frames
        )

        frame_indices = np.clip(
            frame_indices ,
            0 ,
            len(lam) - 1
        )'''

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
    if plot_sings:

        sing_data = data["sing_data"].item()
        print(sing_data)

        # Determinant singularities
        determinant_singularities = set()
        for coord_name, values in sing_data["det_singularities"].items():
            for value in values:
                determinant_singularities.add(
                    (
                        str(coord_name),
                        str(value)
                    )
                )

        # Coordinate singularities
        metric_singularities = set()
        for component, singularities in sing_data["metric_singularities"].items():
            for coord_name, values in singularities.items():
                for value in values:
                    metric_singularities.add(
                        (
                            str(coord_name),
                            value
                        )
                    )

        # Curvature singularities
        K_singularities = set()
        for coord_name, values in sing_data["K_singularities"].items():
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
        color = "blue" ,
        lw = 1.5
    )

    # Add particle object also
    particle , = ax.plot(
        [] ,
        [] ,
        [] ,
        color = "red" ,
        marker = "o" ,
        markersize = 8 ,
        linestyle = ""
    )

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

    sing_txt = fig.text(
        .05 ,
        .5 ,
        "Legend: \nParticle: Red \nDeterminant Singularity: Orange \nCoordinate Singularity: Purple \nCurvature Singularity: Green" ,
        ha = "left" ,
        va = "center" ,
        fontsize = 14
    )

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



    plt.show()


### Run graphics
MakeSimWriteOut(data_path)

if static2D_plots:
    Make2DPlots(data_path=data_path)
if animation2D:
    Make2DAnimation(data_path=data_path, param_by_affine=param_by_affine, nframes=1000)
if animation3D:
    Make3DAnimation(data_path=data_path, param_by_affine=param_by_affine, nframes=1000, plot_sings=visualize_singularities)





