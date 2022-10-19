# @author Nikhil Maserang
# @email nmaserang@berkeley.edu
# @version 1.0.0
# @date 2022/10/19

import analysis_utils as au
import numpy as np

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