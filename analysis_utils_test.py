# @author Nikhil Maserang
# @email nmaserang@berkeley.edu
# @version 1.0.0
# @date 2022/10/19

import analysis_utils as au
import numpy as np

# Error Propagation

eqn = "2*x + b*y"
vars = ["x", "y"]
consts = ["b"]
const_vals = [2]

xdata = np.array([1, 2, 3, 4, 5])
ydata = np.array([-1, -3, -2, 3, 5])
xerr = np.full(5, 1)
yerr = np.full(5, 1)
manual_vals = np.array([0, -2, 2, 14, 20])

quantity, quantity_err = au.calculate_derived_value(eqn, vars, consts, const_vals, [xdata, ydata], [xerr, yerr], True)
au.print_dataset("Some quant", quantity, quantity_err)

# Fitting

model = "m*x + b"
syms = ["x", "m", "b"]
xdata = np.array([0, 1, 2, 3, 4, 5])
ydata = np.array([0, 1, 4, 6, 8, 10])
yerr = np.full(6, 1)
pars, pars_err = au.weighted_least_squares_fit(model, syms, xdata, ydata, yerr)
print(pars, pars_err)
pars, pars_err = au.unweighted_least_squares_fit(model, syms, xdata, ydata)
print(pars, pars_err)


# Testing Get Predicted

model = "m*x + b"
syms = ["x"]
consts = ["m", "b"]
cvals = [2, 1]
xdata = np.array([0, 1, 2, 3, 4, 5])
y = au.get_predicted(model, syms, consts, cvals, xdata, True)
au.print_dataset()