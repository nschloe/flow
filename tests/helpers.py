# -*- coding: utf-8 -*-
#
'''
Helper functions for PDE consistency tests.
'''
import numpy


def _compute_numerical_order_of_convergence(Dt, errors):
    return numpy.array([
        numpy.log(errors[k] / errors[k+1]) / numpy.log(Dt[k] / Dt[k+1])
        for k in range(len(Dt)-1)
        ])
