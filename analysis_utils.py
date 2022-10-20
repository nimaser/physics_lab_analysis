# Copyright 2022 Nikhil Maserang
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.

# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

# @author Nikhil Maserang
# @email nmaserang@berkeley.edu
# @version 1.3.0
# @date 2022/10/19

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.optimize as opt

### DECORATORS ###

def ndarray_typecheck(func : callable) -> callable:
    """Decorator to verify that the arguments to a function are all np.ndarrays."""
    def wrapper(*args):
        for arg in args:
            err_msg = f"Improper argument type {type(arg)}; need np.ndarray.\nArgument val: {arg}"
            assert type(arg) == np.ndarray, err_msg
        return func(*args)
    return wrapper

### DATA ANALYSIS ###

@ndarray_typecheck
def mean(values : np.ndarray) -> float:
    """Wrapper for np.mean."""
    return np.mean(values)

@ndarray_typecheck
def quadrature(values : np.ndarray) -> float:
    """Returns the quadrature of the array `values`."""
    return np.sqrt(np.sum(np.square(values)))

@ndarray_typecheck
def deviations_from_mean(values : np.ndarray) -> np.ndarray:
    """Returns the differences between the array `values` and its mean."""
    mn = np.mean(values)
    return values - mn

@ndarray_typecheck
def sample_variance(values : np.ndarray) -> float:
    """Returns the sample variance of the array `values`."""
    temp = np.square(deviations_from_mean(values))
    return np.sum(temp) / (len(temp) - 1)

@ndarray_typecheck
def sample_standard_deviation(values : np.ndarray) -> float:
    """Returns the sample standard deviation of the array `values`."""
    return np.sqrt(sample_variance(values))

@ndarray_typecheck
def standard_error_from_values(values : np.ndarray) -> float:
    """Returns the standard error of the mean calculated via a dataset's `values`."""
    return sample_standard_deviation(values) / np.sqrt(len(values))

@ndarray_typecheck
def standard_error_from_errors(errors : np.ndarray) -> float:
    """Returns the standard error of the mean calculated via a dataset's `errors`."""
    return quadrature(errors) / len(errors)

@ndarray_typecheck
def determine_agreement(values : np.ndarray[float, float], errors : np.ndarray[float, float]) -> bool:
    """Returns whether two `values` are in agreement according to their `errors`."""
    difference = abs(values[1] - values[0])
    return difference < 2 * quadrature(errors)

@ndarray_typecheck
def residuals(observed : np.ndarray, predicted : np.ndarray) -> np.ndarray:
    """Finds the residuals of the dataset based on the `observed` and `predicted` values."""
    return observed - predicted

@ndarray_typecheck
def normalized_residuals(observed : np.ndarray, predicted : np.ndarray, errors : np.ndarray) -> np.ndarray:
    """Finds the normalized residuals of the dataset, i.e. the residual values divided by their `errors`."""
    return residuals(observed, predicted) / errors

@ndarray_typecheck
def chi_squared(observed : np.ndarray, predicted : np.ndarray, errors : np.ndarray) -> float:
    """Finds the chi squared value for a fit."""
    return np.sum(np.square(normalized_residuals(observed, predicted, errors)))

def reduced_chisq(observed : np.ndarray, predicted : np.ndarray, errors: np.ndarray, num_params : int) -> float:
    """
    Calculates the reduced chi squared of the dataset.
    
    The `num_params` argument refers to the number of parameters used in the fitting model.
    It is used to calculate the degrees of freedom.
    """
    return chi_squared(observed, predicted, errors) / (len(observed) - num_params)

### FITTING ###

def unweighted_least_squares_fit(model_str : str, syms : list[str], x : np.ndarray, y : np.ndarray, initial_parameters : list = None):
    """
    Performs an unweighted least squares fit for the dataset using the specified `model`.
    `syms` should contain a list of strings, each of which is a symbol used in the model, starting with the independent variable.
    
    Returns a tuple containing:
    - the optimized parameters
    - the error in those parameters
    """
    syms = sp.symbols(syms)
    eqn = sp.sympify(model_str)
    model = sp.lambdify(syms, eqn, "numpy")
    if initial_parameters:
        optimized_params, param_covariance = opt.curve_fit(model, x, y, p0=initial_parameters)
    else:
        optimized_params, param_covariance = opt.curve_fit(model, x, y)
    param_error = np.sqrt(np.diag(param_covariance))
    return optimized_params, param_error

def weighted_least_squares_fit(model_str : str, syms : list[str], x : np.ndarray, y : np.ndarray, y_err : np.ndarray, initial_parameters : list = None):
    """
    Performs a weighted least squares fit for the dataset using the specified `model`.
    `syms` should contain a list of strings, each of which is a symbol used in the model, starting with the independent variable.

    Returns a tuple containing:
    - the optimized parameters
    - the error in those parameters
    """
    syms = sp.symbols(syms)
    eqn = sp.sympify(model_str)
    model = sp.lambdify(syms, eqn, "numpy")
    if initial_parameters:
        optimized_params, param_covariance = opt.curve_fit(model, x, y, sigma=y_err, absolute_sigma=True, p0=initial_parameters)
    else:
        optimized_params, param_covariance = opt.curve_fit(model, x, y, sigma=y_err, absolute_sigma=True)
    param_error = np.sqrt(np.diag(param_covariance))
    return optimized_params, param_error

def get_predicted(model_str : str, syms : list[str], consts : list[str], const_vals : np.ndarray, x : np.ndarray, verbose : bool = False) -> np.ndarray:
    """Returns the predicted values for the independent variable `x` based on the provided `model`."""
    syms = sp.symbols(syms)
    consts = sp.symbols(consts)
    eqn = sp.sympify(model_str).subs(zip(consts, const_vals))
    if verbose: print(f"Getting predicted values...\n\tModel after substituting in values: {eqn}")
    model = sp.lambdify(syms, eqn, "numpy")
    return np.array([model(_) for _ in x])

### MISC UTILS ###

def read_csv_columns(fname : str, datatype=float) -> np.ndarray:
    """Extracts the columns from a rectangular csv file with name `fname` and returns them as a nested np.ndarray."""
    # open file
    with open(fname, "r") as f:
        # read the lines of the file into an array
        lines = f.readlines()
        # iterate through each line, keeping track of its index
        for i, line in enumerate(lines):
            # filter any non-numerical and non-comma characters to avoid parsing errors
            line = ''.join([c for c in line if ord('0') <= ord(c) <= ord('9') or ord(c) == ord(',')])
            # split the line into an array by commas, then convert it to a np array of `type`
            lines[i] = np.array(list(map(datatype, line.split(','))))
        # place the lines in an array, then take the transpose to get data columns
        cols = np.array(lines).transpose()
        return cols

def read_from_docs(fname : str, skiprows : int = 0, datatype=float) -> list[np.ndarray]:
    """Reads file data formatted with linebreaks between datapoints and multiple linebreaks between datasets."""
    datasets = []
    current_dataset = []
    with open(fname, "r") as f:
        lines = f.readlines()
        lines = lines[skiprows:] if skiprows < len(lines) else []
        for line in lines:
            line = line.strip()
            if len(line) == 0 and len(current_dataset) > 0:
                datasets.append(np.array(list(map(datatype, current_dataset))))
                current_dataset = []
            if len(line) > 0: current_dataset.append(line)
        if len(current_dataset) > 0: datasets.append(np.array(list(map(datatype, current_dataset))))
        return datasets

def print_dataset(name : str, values : np.ndarray, errors : np.ndarray, precision : int =None):
    """Prints the values, errors, and mean of a dataset with the provided `precision`. `precision=None` is full precision."""
    print(f"{name} Dataset:")
    if precision == None:
        for value, error in zip(values, errors):
            print(f"\t{value} ± {error}")
        print(f"\t\tMean: {mean(values)}")
        print(f"\t\tStd : {sample_standard_deviation(values)}")
        print(f"\t\tSte from values: {standard_error_from_values(values)}")
        print(f"\t\tSte from errors: {standard_error_from_errors(errors)}")
    else:
        for value, error in zip(values, errors):
            print(f"\t{value : .{precision}f} ± {error : .{precision}f}")
        print(f"\t\tMean:{mean(values) : .{precision}f}")
        print(f"\t\tStd :{sample_standard_deviation(values) : .{precision}f}")
        print(f"\t\tSte from values:{standard_error_from_values(values) : .{precision}f}")
        print(f"\t\tSte from errors:{standard_error_from_errors(errors) : .{precision}f}")

### PLOTTING ###

def configure_plt(title : str, xlabel : str, ylabel : str) -> None:
    """Sets the `title`, `xlabel`, and `ylabel` of plt's plot."""
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def plot_residuals(x : np.ndarray, residuals : np.ndarray) -> None:
    """Plots the residuals of a fit using matplotlib.pyplot."""
    plt.axhline(0, color="red")
    plt.plot(x, residuals, "k .")

### ERROR PROPAGATION ###

def get_error_propagation(eqn : str, vars : list[str], verbose : bool = False) -> tuple[sp.Pow, sp.Basic, dict, dict]:
    """Internal! Returns the sympy `err_eqn`, `eqn`, list of `vars`, list of `alphas` (errors)."""
    if verbose: print("Propagating error...")

    # first, we make name : symbol dictionaries for the variables and errors (alphas)
    alphas = [f"α_{var}" for var in vars]
    alphas = sp.symbols(alphas)
    vars = sp.symbols(vars)
    if verbose:
        print(f"\tvars: {vars}")
        print(f"\terrs: {alphas}")

    # use sympify to turn str to sympy expression, passing in symbols as locals
    eqn = sp.sympify(eqn)
    if verbose: print(f"\teqn: {eqn}")

    # compute partials
    diffs = list(map(eqn.diff, vars))
    if verbose:
        print("\tPartials:")
        for v, d in zip(vars, diffs):
            print(f"\t\twrt {v}: {d}")
    
    # multiply by error and square
    for i in range(len(diffs)): diffs[i] = (alphas[i] * diffs[i]) ** 2

    # sum, then take sqrt
    err_eqn = sp.sqrt(sp.Add(*diffs))
    if verbose: print(f"\terr_eqn: {err_eqn}")
    return err_eqn, eqn, vars, alphas

def get_error_propagation_latex(eqn : str, vars : list[str], verbose : bool = False) -> str:
    """Returns the error propagation formula for the given expression with given symbols in latex format."""
    return sp.latex(get_error_propagation(eqn, vars, verbose)[0])

def calculate_derived_value(eqn : str, vars : list[str], consts : list[str], const_vals : np.ndarray, datasets : list[np.ndarray], errors : list[np.ndarray], verbose : bool = False) -> list[np.ndarray, np.ndarray]:
    """Calculates a derived quantity and its error."""
    err_eqn, eqn, vars, alphas = get_error_propagation(eqn, vars, verbose)

    # substitute in constant values
    if verbose: print("Substituting in constants...")
    consts = sp.symbols(consts)
    if verbose:
        print("\tConstants:")
        for c, cval in zip(consts, const_vals):
            print(f"\t\t{c} = {cval}")
    err_eqn = err_eqn.subs(list(zip(consts, const_vals)))
    eqn = eqn.subs(list(zip(consts, const_vals)))

    if verbose:
        print("\tAfter substituting:")
        print(f"\t\teqn    : {eqn}")
        print(f"\t\terr eqn: {err_eqn}")

    if verbose: print("Lambdifying and evaluating...")

    # evaluate quantity
    eqn_lambda = sp.lambdify(vars, eqn, "numpy")
    quantity = np.array(list(map(eqn_lambda, *datasets)))

    # add lists together
    vars_and_errs = vars + alphas
    vals_and_errs = list(datasets) + list(errors)

    # evaluate quantity error
    err_eqn_lambda = sp.lambdify(vars_and_errs, err_eqn, "numpy")
    quantity_error = np.array(list(map(err_eqn_lambda, *vals_and_errs)))

    if verbose: print("Done!")

    return quantity, quantity_error

### PIPELINE ###

def mean_and_error(values : np.ndarray, errors : np.ndarray = None, precision : int = None):
    """Returns the mean and associated error of a measurement dataset from raw values and associated errors."""
    mn = mean(values)
    stev = standard_error_from_values(values)
    stee = standard_error_from_errors(errors) if not errors is None else 0
    ste = max(stev, stee)
    if precision: return f"{mn : .{precision}f} ± {ste : .{precision}f}"
    return f"{mn} ± {ste}"