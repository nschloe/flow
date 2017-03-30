# -*- coding: utf-8 -*-
#
'''
Helper functions for PDE consistency tests.
'''
import numpy
import sympy


def compute_numerical_order_of_convergence(Dt, errors):
    return numpy.array([
        numpy.log(errors[k] / errors[k+1]) / numpy.log(Dt[k] / Dt[k+1])
        for k in range(len(Dt)-1)
        ])


def ccode(*args, **kwargs):
    # FEniCS needs to have M_PI replaced by pi
    return sympy.ccode(*args, **kwargs).replace('M_PI', 'pi')
