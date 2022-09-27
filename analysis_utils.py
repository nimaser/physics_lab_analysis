# @author Nikhil Maserang
# @email nmaserang@berkeley.edu
# @version 1.0.0
# @date 2022/09/27

import numpy as np
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

def unweighted_least_squares_fit(model : callable, x : np.ndarray, y : np.ndarray, initial_parameters : list = None):
    """
    Performs an unweighted least squares fit for the dataset using the specified `model`.
    
    Returns a tuple containing:
    - the optimized parameters
    - the error in those parameters
    """
    if initial_parameters:
        optimized_params, param_covariance = opt.curve_fit(model, x, y, p0=initial_parameters)
    else:
        optimized_params, param_covariance = opt.curve_fit(model, x, y)
    param_error = np.sqrt(np.diag(param_covariance))
    return optimized_params, param_error

def weighted_least_squares_fit(model : callable, x : np.ndarray, y : np.ndarray, y_err : np.ndarray, initial_parameters : list = None):
    """
    Performs a weighted least squares fit for the dataset using the specified `model`.

    Returns a tuple containing:
    - the optimized parameters
    - the error in those parameters
    """
    if initial_parameters:
        optimized_params, param_covariance = opt.curve_fit(model, x, y, sigma=y_err, absolute_sigma=True, p0=initial_parameters)
    else:
        optimized_params, param_covariance = opt.curve_fit(model, x, y, sigma=y_err, absolute_sigma=True)
    param_error = np.sqrt(np.diag(param_covariance))
    return optimized_params, param_error

def get_predicted(model : callable, x : np.ndarray) -> np.ndarray:
    """Returns the predicted values for the independent variable `x` based on the provided `model`."""
    return np.ndarray([model(_) for _ in x])

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

def aggregate_datasets(independent : np.ndarray, dependents : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aggregates several dependent variable datasets which share the same independent variable."""
    aggregate_data = []
    for dataset in dependents:
        for i, point in enumerate(dataset):
            aggregate_data.append((independent[i], point))
    # transpose from points to columns
    return tuple(map(np.array, list(zip(*aggregate_data))))

def print_dataset(name : str, data : np.ndarray, errs : np.ndarray, precision : float =None):
    """Prints the values, errors, and mean of a dataset with the provided `precision`. `precision=None` is full precision."""
    print(f"{name} Dataset:")
    if precision == None:
        for value, error in zip(data, errs):
            print(f"\t{value} ± {error}")
        print(f"\t\tMean: {mean(data)}")
        print(f"\t\tStd : {sample_standard_deviation(data)}")
        print(f"\t\tSte from values: {standard_error_from_values(data)}")
        print(f"\t\tSte from errors: {standard_error_from_errors(errs)}")
    else:
        for value, error in zip(data, errs):
            print(f"\t{value : .{precision}f} ± {error : .{precision}f}")
        print(f"\t\tMean: {mean(data) : .{precision}f}")
        print(f"\t\tStd : {sample_standard_deviation(data) : .{precision}f}")
        print(f"\t\tSte from values: {standard_error_from_values(data) : .{precision}f}")
        print(f"\t\tSte from errors: {standard_error_from_errors(errs) : .{precision}f}")

### PLOTTING ###

def configure_plt(title : str, xlabel : str, ylabel : str) -> None:
    """Sets the `title`, `xlabel`, and `ylabel` of plt's plot."""
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def plot_residuals(x : np.ndarray, residuals : np.ndarray):
    """Plots the residuals of a fit using matplotlib.pyplot."""
    plt.axhline(0, color="red")
    plt.plot(x, residuals, "k .")