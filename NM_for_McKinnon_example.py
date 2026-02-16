import numpy as np


def Nelden_Mead(fct, simplex):
    # evaluate simplex
    simplex_val = fct(simplex)
    # Sort simplex
    sorted_simplex = np.sort(simplex_val)
    

    xc = 1/simplex.shape[0] * np.sum(simplex, axis=0)
    xr = xc + (xc - simplex[-1])


    if fct(xr) > sorted_simplex[-1]:
        xic = xc - 1/2*(xc - sorted_simplex[-1])
        if fct(xic) > sorted_simplex[-1]:
            simplex[-1] = xe1
        else:
            simplex[-1] = xr
