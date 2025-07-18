import numpy as np
import inspect
from matplotlib import pyplot as plt
from matplotlib import ticker
from scipy.constants import physical_constants
from scipy.optimize import curve_fit
from sopes import Table
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter, rotate
from scipy.interpolate import SmoothBivariateSpline

# Boltzmann constant in SESAME units kJ/K
kB = physical_constants['Boltzmann constant'][0] / 1000. #turning this into kiloJoules because SESAME energy data is in that units
# the lowercase k is standard for constant 


###############################################################################
# Getting the SESAME data
###############################################################################

# Set up the P-T table
P_of_RT = Table('EOS_Pt_DT', 3337) #pressure table as a function of RT, the R stands for rho (density)
e_of_RT = Table('EOS_Ut_DT', 3337) #internal energy (U) table as a function of RT
P_of_Re = Table('EOS_Pt_DUt', 3337) #Pressure table as a function of R (density) and e(internal energy)
ecold_of_R = Table('EOS_Uc_D', 3337) #Loads the cold curve as a function of density e_cold(rho)

# Get the density-temperature grid from SESAME tables
nR, nT = [int(val) for val in P_of_RT.info_many(['NR', 'NT'])] #getting the nR (number of density points) and nT
T_grid = np.array(P_of_RT.info_one('T_Array')) #retrives actual values
density_grid = np.array(P_of_RT.info_one('R_Array'))
density_points, temp_points = np.meshgrid(density_grid, T_grid, indexing='ij') #each are nR x nT

# Get the values on the grid
P_on_grid, _, _ = P_of_RT.interpolate_many(density_points.flatten(),
                                           temp_points.flatten()) #the _, _ just means ignore the other 2 return values
P_on_grid = P_on_grid.reshape(nR, nT)
e_on_grid, dedr_T, dedT_R = e_of_RT.interpolate_many(density_points.flatten(),
                                                     temp_points.flatten()) #finds pressure at every coordinate
e_on_grid = e_on_grid.reshape(nR, nT) 
Cv_on_grid = dedT_R.reshape(nR, nT)

# Get the cold curve. Note that we're interpolating on nR x nT points here.
# Also note that the temperature points are irrelevant
ecold_on_grid, _, _ = ecold_of_R.interpolate_many(density_points.flatten(),
                                                  temp_points.flatten()) #does the same lookups for energy, interpolates using 4 points
ecold_on_grid = ecold_on_grid.reshape(nR, nT)

###############################################################################
# Gaussian
###############################################################################

def smooth_CV_gaussian(CV, sigma=1.25):
    """
    Smooth the 2D CV array using a Gaussian filter.

    Parameters
    ----------
    CV : 2D array of shape (N, M)
        Heat capacity values on the (rho, T) grid.
    sigma : float or sequence of floats
        Standard deviation for Gaussian kernel. Can be a single value or tuple (sigma_rho, sigma_T).

    Returns
    -------
    CV_smooth : 2D array of same shape as CV
        Smoothed CV values.
    """
    return gaussian_filter(CV, sigma=sigma)

def predict_CV_from_grid(rho_vec, T_vec, CV_smooth, rho, T):
    """
    Interpolate smoothed CV values at new (rho, T) points.

    Parameters
    ----------
    rho_vec : 1D array of shape (N,)
        Grid values of density.
    T_vec : 1D array of shape (M,)
        Grid values of temperature.
    CV_smooth : 2D array of shape (N, M)
        Smoothed CV values from smooth_CV_gaussian.
    rho : float or array-like
        New rho values to interpolate at.
    T : float or array-like
        New T values to interpolate at.

    Returns
    -------
    CV_pred : interpolated CV values at (rho, T)
    """
    from scipy.interpolate import RegularGridInterpolator

    interpolator = RegularGridInterpolator((rho_vec, T_vec), CV_smooth, bounds_error=False, fill_value=None)
    rho = np.atleast_1d(rho)
    T = np.atleast_1d(T)
    r, t = np.broadcast_arrays(rho, T)
    points = np.stack([r.ravel(), t.ravel()], axis=-1)
    result = interpolator(points)
    return result.reshape(r.shape)

###############################################################################
# Smoothed B-spline
###############################################################################

# Flatten and prepare input
x = density_points.flatten()
y = temp_points.flatten()
z = Cv_on_grid.flatten()

# Fit smoothed B-spline (s controls smoothing strength)
spline = SmoothBivariateSpline(x, y, z, s=1e-6, kx=3, ky=3)  # Try different s values

# Evaluate on the original grid
Cv_spline = spline(density_grid, T_grid)



def assert_argcount(fn, args_list):
    """
    Raise a ValueError if fn does not accept exactly len(args_list)
    positional arguments (excluding *args and **kwargs).
    """
    sig = inspect.signature(fn)
    params = sig.parameters.values()
    # Count only “regular” positional parameters (no VAR_POSITIONAL or VAR_KEYWORD)
    n_positional = sum(
        1 for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.default is inspect._empty  # no default => required
    )
    # If you’d rather allow defaults to count too, drop the default check:
    # n_positional = sum(1 for p in params if p.kind in (...))
    if len(args_list) != n_positional:
        raise ValueError(
            f"{fn.__name__} takes {n_positional} required positional "
            f"arguments, but you passed a list of length {len(args_list)}."
        )




###############################################################################
# Transforming the data
###############################################################################

# First: subtract cold curve from energies
e_minus_cold_on_grid = e_on_grid - ecold_on_grid

Cv_smooth = smooth_CV_gaussian(Cv_on_grid, sigma=(1.5, 1.5))  # Adjust sigma as needed

# Second: divide by the heat capacity
min_Cv = 1.0e-08  # MJ / K / kg
e_transform = e_minus_cold_on_grid / np.maximum(Cv_on_grid, min_Cv)
e_transform_fit = e_minus_cold_on_grid / np.maximum(Cv_smooth, min_Cv)


##############################################################################################
#Plotting
#############################################################################################

# Plot the various versions of the table
fig, ax = plt.subplots()
ax.set_title("Untransformed")
ax.set_ylabel("Energy (MJ/kg)")
ax.set_yscale('log')
ax.set_xlabel(r"Density (g/cm$^3$)")
ax.set_xscale('log')
ax.plot(density_grid, e_on_grid)
density_lims = ax.get_xlim()
energy_lims = ax.get_ylim()
fig.tight_layout()

locator = ticker.LogLocator(base=10)
fmatter = ticker.LogFormatterSciNotation(base=10)
levels = np.logspace(np.log10(np.maximum(Cv_smooth.min(), 1e-8)),
                     np.log10(Cv_smooth.max()), 100)

# Plot smoothed Cv
fig, ax = plt.subplots()
ax.set_title(r"Smoothed $C_V$ (Gaussian filter)")
ax.set_ylabel(r"Density (g/cm$^3$)")
ax.set_xlabel(r"Temperature (K)")
ax.set_yscale('log')
ax.set_xscale('log')
cplot = ax.contourf(density_points, temp_points, Cv_smooth, levels,
                    cmap='ocean', locator=locator)
clb = plt.colorbar(cplot, format=fmatter, ticks=locator)
ax.set_ylim(np.min(T_grid[T_grid > 0]), np.max(T_grid))
ax.set_xlim(np.min(density_grid[density_grid > 0]), np.max(density_grid))
fig.tight_layout()

# Plot Smooth Bivariate Spline
fig, ax = plt.subplots()
ax.set_title(r"Smoothed B-spline")
ax.set_ylabel(r"Density (g/cm$^3$)")
ax.set_xlabel(r"Temperature (K)")
ax.set_yscale('log')
ax.set_xscale('log')
cplot = ax.contourf(density_points, temp_points, Cv_spline, levels,
                    cmap='ocean', locator=locator)
clb = plt.colorbar(cplot, format=fmatter, ticks=locator)
ax.set_ylim(np.min(T_grid[T_grid > 0]), np.max(T_grid))
ax.set_xlim(np.min(density_grid[density_grid > 0]), np.max(density_grid))
fig.tight_layout()

# fig, ax = plt.subplots()
# ax.set_title("Debye fit to energy")
# ax.set_ylabel("Energy (MJ/kg)")
# ax.set_yscale('log')
# ax.set_xlabel(r"Density (g/cm$^3$)")
# ax.set_xscale('log')
# ax.plot(density_grid, e_fit)  # e_fit should be the Debye fit result here
# ax.set_xlim(*density_lims)
# ax.set_ylim(*energy_lims)
# fig.tight_layout()

# print(f"density_grid shape: {density_grid.shape}, range: {density_grid.min()} to {density_grid.max()}")
# print(f"T_grid shape: {T_grid.shape}, range: {T_grid.min()} to {T_grid.max()}")
# print(f"e_on_grid shape: {e_on_grid.shape}, min: {e_on_grid.min()}, max: {e_on_grid.max()}")
# print(f"ecold_on_grid shape: {ecold_on_grid.shape}, min: {ecold_on_grid.min()}, max: {ecold_on_grid.max()}")
# print(f"e_minus_cold_on_grid min: {e_minus_cold_on_grid.min()}, max: {e_minus_cold_on_grid.max()}")

# print(f"e_fit shape: {e_fit.shape}")
# print(f"e_fit sample values:\n{e_fit[:5, :5]}")
# print(f"Are any values NaN or Inf? {np.isnan(e_fit).any() or np.isinf(e_fit).any()}")

# print("e_on_grid max:", np.max(e_on_grid))
# print("e_fit max:", np.max(e_fit))

# print(f"Initial parameter guesses: {parameter_guesses}")
# print("Fitted parameters:", popt)



fig, ax = plt.subplots()
ax.set_title("Subtract Cold Curve")
ax.set_ylabel("Shifted Energy (MJ/kg)")
ax.set_yscale('log')
ax.set_xlabel(r"Density (g/cm$^3$)")
ax.set_xscale('log')
ax.plot(density_grid, e_minus_cold_on_grid)
fig.tight_layout()

fig, ax = plt.subplots()
ax.set_title(r"Subtract Cold Curve, Divide by $C_V$")
ax.set_ylabel("Transformed Energy (K)")
ax.set_yscale('log')
ax.set_xlabel(r"Density (g/cm$^3$)")
ax.set_xscale('log')
ax.plot(density_grid, e_transform)
fig.tight_layout()

fig, ax = plt.subplots()
ax.set_title(r"Subtract Cold Curve, Divide by $C_V$ fit")
ax.set_ylabel("Transformed Energy (K)")
ax.set_yscale('log')
ax.set_xlabel(r"Density (g/cm$^3$)")
ax.set_xscale('log')
ax.plot(density_grid, e_transform_fit)
fig.tight_layout()

fig, ax = plt.subplots()
locator = ticker.LogLocator(base=10)
fmatter = ticker.LogFormatterSciNotation(base=10)
ax.set_title(r"Heat capacity")
ax.set_ylabel(r"Density (g/cm$^3$)")
ax.set_yscale('log')
ax.set_xlabel(r"Temperature (K)")
ax.set_xscale('log')
levels = np.logspace(np.log10(np.min(Cv_on_grid)),
                     np.log10(np.max(Cv_on_grid)),
                     100)
cplot = ax.contourf(density_points, temp_points, Cv_on_grid, levels,
                    cmap='ocean',
                    locator=locator)
clb = plt.colorbar(cplot, format=fmatter, ticks=locator)
ax.set_ylim(np.min(T_grid[T_grid > 0]), np.max(T_grid))
ax.set_xlim(np.min(density_grid[density_grid > 0]), np.max(density_grid))
fig.tight_layout()

plt.show()