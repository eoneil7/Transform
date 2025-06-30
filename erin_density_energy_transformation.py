
import numpy as np
import inspect
from matplotlib import pyplot as plt
from matplotlib import ticker
from scipy.constants import physical_constants
from scipy.optimize import curve_fit
from sopes import Table
from scipy.integrate import quad

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
# Fitting to a Debye model
###############################################################################

def debye_energy_linear(flat_args, p0, p1, p2, p3):
    """
    Vectorized Debye-model energy surface.
    Debye temperature and prefactor are linear functions of density.

    flat_args: tuple of two 1D arrays (density_flat, temp_flat)
    p0, p1: parameters for Theta_D(d) = p0 + p1*d
    p2, p3: parameters for A(d)       = p2 + p3*d

    Energy per particle is:
    E = 9 * A * kB * T * (T / Theta_D)^3 * Integral_0^{Theta_D/T} (x^3 / (exp(x)-1)) dx
    Here we approximate the integral with the Debye function approximation.
    """
    import scipy.integrate

    dens_flat, T_flat = flat_args
    Theta_D = p0 + p1 * dens_flat
    A = p2 + p3 * dens_flat

    # Avoid zero temperature to prevent division by zero
    T_safe = np.where(T_flat < 1e-6, 1e-6, T_flat)
    x = Theta_D / T_safe

    def debye_integral(x_val):
        # Compute integral of (t^3 / (exp(t) -1)) dt from 0 to x_val
        # If x_val is an array, this needs vectorization
        # We'll vectorize it below with np.vectorize
        def integrand(t):
            return t**3 / (np.exp(t) - 1)
        result, _ = scipy.integrate.quad(integrand, 0, x_val)
        return result

    vectorized_debye_int = np.vectorize(debye_integral)
    D_x = vectorized_debye_int(x)

    # Energy per particle (MJ/kg) - keep consistent units
    with np.errstate(divide='ignore', invalid='ignore'):
        E = 9 * A * kB * T_safe * (T_safe / Theta_D)**3 * D_x
        # For zero temperature points, set energy to zero
        E = np.where(T_flat > 0, E, 0.)
    return E


def debye_heat_capacity_linear(flat_args, p0, p1, p2, p3):
    """
    Vectorized Debye heat capacity surface.
    Prefactor and Debye temperature are linear functions of density.

    flat_args: tuple of two 1D arrays (density_flat, temp_flat)
    p0, p1: parameters for Theta_D(d) = p0 + p1*d
    p2, p3: parameters for A(d)       = p2 + p3*d

    Heat capacity per particle:
    C_V = 9 * A * k_B * (T / Theta_D)^3 * Integral_0^{Theta_D/T} (x^4 * e^x) / ( (e^x - 1)^2 ) dx
    """

    import scipy.integrate

    dens_flat, T_flat = flat_args
    Theta_D = p0 + p1 * dens_flat
    A = p2 + p3 * dens_flat

    T_safe = np.where(T_flat < 1e-6, 1e-6, T_flat)
    x = Theta_D / T_safe

    def integrand(t):
        ex = np.exp(t)
        return (t**4 * ex) / (ex - 1)**2

    def debye_cv_integral(x_val):
        result, _ = scipy.integrate.quad(integrand, 0, x_val)
        return result

    vectorized_cv_int = np.vectorize(debye_cv_integral)
    I_x = vectorized_cv_int(x)

    with np.errstate(divide='ignore', invalid='ignore'):
        Cv = 9 * A * kB * (T_safe / Theta_D)**3 * I_x
        Cv = np.where(T_flat > 0, Cv, 0.)
    return Cv

def fit_debye(densities, temperatures, E_data,
               parameter_guesses, debye_energy=debye_energy_linear):
    """
    Fits the 2D energy array E_data[d_i, T_j] to the Debye model
    with linear density dependence in Theta_D and A.

    Returns:
      popt : optimized parameters [p0, p1, p2, p3]
      pcov : covariance matrix
    """
    D, T = np.meshgrid(densities, temperatures, indexing='ij')
    dens_flat = D.ravel()
    temp_flat = T.ravel()
    E_flat = E_data.ravel()

    flat_args = (dens_flat, temp_flat)

    popt, pcov = curve_fit(
        debye_energy,
        flat_args,
        E_flat,
        p0=parameter_guesses,
        maxfev=10000
    )
    return popt, pcov




###############################################################################
# Fitting to an Einstein model
###############################################################################

def einstein_energy_linear(flat_args, p0, p1, p2, p3):
    """
    Vectorized Einstein‐model energy surface. Prefactor and characteristic
    fequency are linear functions of density.

    flat_args: tuple of two 1D arrays (density_flat, temp_flat)
    p0, p1: parameters for Theta_E(d) = p0 + p1*d
    p2, p3: parameters for A(d)       = p2 + p3*d
    """
    dens_flat, T_flat = flat_args
    Theta = p0 + p1 * dens_flat
    A = p2 + p3 * dens_flat
    # Einstein energy per particle (3 k_B Theta / (exp(x)-1)), times prefactor A
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.clip(Theta / T_flat, 1e-6, 700) #Is this allowed? exp(x) overflows when its out of this range
        return np.where(
            T_flat > 0,
            (A * (3 * kB * Theta)) / (np.exp(x) - 1),
            0.
        )



def einstein_heat_capacity_linear(flat_args, p0, p1, p2, p3):
    """
    Vectorized Einstein‐model energy surface heat capacity. Prefactor and
    characteristic fequency are linear functions of density.

    flat_args: tuple of two 1D arrays (density_flat, temp_flat)
    p0, p1: parameters for Theta_E(d) = p0 + p1*d
    p2, p3: parameters for A(d)       = p2 + p3*d
    """

    dens_flat, T_flat = flat_args
    Theta = p0 + p1 * dens_flat
    A = p2 + p3 * dens_flat
    # Einstein energy per particle (3 k_B Theta / (exp(x)-1)), times prefactor A
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.clip(Theta / T_flat, 1e-6, 700)
        return np.where(
            T_flat > 0,
            (A * 3 * kB * x**2 * np.exp(x)) / ( (np.exp(x) - 1)**2 ),
            0.
        )


def fit_einstein(densities, temperatures, E_data,
                 parameter_guesses, einstein_energy=einstein_energy_linear):
    """
    Fits the 2D energy array E_data[d_i, T_j] to the Einstein model
    with linear density dependence in Theta_E and A.

    Returns:
      popt : optimized parameters [p0, p1, p2, p3]
      pcov : covariance matrix
    """
    # Create meshgrid and flatten
    D, T = np.meshgrid(densities, temperatures, indexing='ij')
    dens_flat = D.ravel()
    temp_flat = T.ravel()
    E_flat    = E_data.ravel()

    # pack arguments for curve_fit
    flat_args = (dens_flat, temp_flat)

    popt, pcov = curve_fit(
        einstein_energy,
        flat_args,
        E_flat,
        p0=parameter_guesses,
        maxfev=10000
    )

    return popt, pcov

def assert_argcount(fn, args_list):
    """
    Raise a ValueError if `fn` does not accept exactly len(args_list)
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

# Choose model for fit
parameter_guesses = [300., 5.0, 100., 1.0]
#model = einstein_energy_linear
model = debye_energy_linear


# Make sure the model takes our parameter guesses plus an initial pair of points
#assert_argcount(model, [(1.0, 1.0), ] + parameter_guesses)

# Fit the parameters, but make sure not to include the zero-Kelvin isotherm
#popt, pcov = fit_einstein(density_grid, T_grid, e_minus_cold_on_grid,
#                          parameter_guesses, einstein_energy=model)
popt, pcov = fit_debye(density_grid, T_grid, e_minus_cold_on_grid,
                          parameter_guesses, debye_energy=debye_energy_linear)
e_fit = model((density_points, temp_points), *popt)
#Cv_fit = einstein_heat_capacity_linear((density_points, temp_points), *popt)
Cv_fit = debye_heat_capacity_linear((density_points, temp_points), *popt)


# Second: divide by the heat capacity
min_Cv = 1.0e-08  # MJ / K / kg
e_transform = e_minus_cold_on_grid / np.maximum(Cv_on_grid, min_Cv)
e_transform_fit = e_minus_cold_on_grid / np.maximum(Cv_fit, min_Cv)


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


# fig, ax = plt.subplots()
# ax.set_title("Einstein fit to energy")
# ax.set_ylabel("Energy (MJ/kg)")
# ax.set_yscale('log')
# ax.set_xlabel(r"Density (g/cm$^3$)")
# ax.set_xscale('log')
# ax.plot(density_grid, e_fit)
# ax.set_xlim(*density_lims)
# ax.set_ylim(*energy_lims)
# fig.tight_layout()


fig, ax = plt.subplots()
ax.set_title("Debye fit to energy")
ax.set_ylabel("Energy (MJ/kg)")
ax.set_yscale('log')
ax.set_xlabel(r"Density (g/cm$^3$)")
ax.set_xscale('log')
ax.plot(density_grid, e_fit)  # e_fit should be the Debye fit result here
ax.set_xlim(*density_lims)
ax.set_ylim(*energy_lims)
fig.tight_layout()

print(f"density_grid shape: {density_grid.shape}, range: {density_grid.min()} to {density_grid.max()}")
print(f"T_grid shape: {T_grid.shape}, range: {T_grid.min()} to {T_grid.max()}")
print(f"e_on_grid shape: {e_on_grid.shape}, min: {e_on_grid.min()}, max: {e_on_grid.max()}")
print(f"ecold_on_grid shape: {ecold_on_grid.shape}, min: {ecold_on_grid.min()}, max: {ecold_on_grid.max()}")
print(f"e_minus_cold_on_grid min: {e_minus_cold_on_grid.min()}, max: {e_minus_cold_on_grid.max()}")

print(f"e_fit shape: {e_fit.shape}")
print(f"e_fit sample values:\n{e_fit[:5, :5]}")
print(f"Are any values NaN or Inf? {np.isnan(e_fit).any() or np.isinf(e_fit).any()}")

print("e_on_grid max:", np.max(e_on_grid))
print("e_fit max:", np.max(e_fit))

print(f"Initial parameter guesses: {parameter_guesses}")
print("Fitted parameters:", popt)





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