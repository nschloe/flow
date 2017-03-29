# -*- coding: utf-8 -*-
#
import flow

from dolfin import (
    UnitSquareMesh, Expression, VectorElement, FiniteElement, DirichletBC,
    errornorm, FunctionSpace
    )
import matplotlib.pyplot as plt
import numpy
import pytest
import sympy


MAX_DEGREE = 5


def _ccode(*args, **kwargs):
    # FEniCS needs to have M_PI replaced by pi
    return sympy.ccode(*args, **kwargs).replace('M_PI', 'pi')


def _get_stokes_rhs(u, p):
    '''Given a solution u of the Cartesian Navier-Stokes equations, return
    a matching right-hand side f.
    '''
    x = sympy.DeferredVector('x')

    # Make sure that the exact solution is indeed analytically div-free.
    d = sympy.diff(u[0], x[0]) + sympy.diff(u[1], x[1])
    d = sympy.simplify(d)
    assert d == 0

    f0 = (
        - (sympy.diff(u[0], x[0]) + sympy.diff(u[0], x[1]))
        + sympy.diff(p, x[0])
        )
    f1 = (
        - (sympy.diff(u[1], x[0]) + sympy.diff(u[1], x[1]))
        + sympy.diff(p, x[1])
        )
    f = (
        sympy.simplify(f0),
        sympy.simplify(f1)
        )

    return f


class Flat(object):
    '''Nothing interesting happening in the domain.
    '''
    def __init__(self):
        x = sympy.DeferredVector('x')
        u = (0 * x[0], 0 * x[1])
        p = -9.81 * x[1]
        self.solution = {
            'u': {'value': u, 'degree': 1},
            'p': {'value': p, 'degree': 1},
            }
        self.f = {
            'value': _get_stokes_rhs(u, p),
            'degree': MAX_DEGREE,
            }
        return

    def mesh_generator(self, n):
        return UnitSquareMesh(n, n, 'left/right')


class Guermond1(object):
    '''Problem 1 from section 3.7.1 in
        An overview of projection methods for incompressible flows;
        Guermond, Minev, Shen;
        Comp. Meth. in Appl. Mech. and Eng., vol. 195, 44-47, pp. 6011-6045;
        <http://www.sciencedirect.com/science/article/pii/S0045782505004640>.
    '''
    def __init__(self):
        from sympy import pi, sin, cos
        x = sympy.DeferredVector('x')

        u = (
            +pi * 2 * sin(pi*x[1]) * cos(pi*x[1]) * sin(pi*x[0])**2,
            -pi * 2 * sin(pi*x[0]) * cos(pi*x[0]) * sin(pi*x[1])**2,
            )
        p = cos(pi*x[0]) * sin(pi*x[1])

        self.solution = {
            'u': {'value': u, 'degree': MAX_DEGREE},
            'p': {'value': p, 'degree': MAX_DEGREE},
            }
        self.f = {
            'value': _get_stokes_rhs(u, p),
            'degree': MAX_DEGREE,
            }
        return

    def mesh_generator(self, n):
        return UnitSquareMesh(n, n, 'left/right')


def assert_order(problem):
    mesh_sizes = [8, 16, 32]
    for mesh_size in mesh_sizes:
        compute_error(problem, mesh_size)
    return


def compute_error(problem, mesh_size):
    mesh = problem.mesh_generator(mesh_size)

    u_sol = Expression(
            (
                _ccode(problem.solution['u']['value'][0]),
                _ccode(problem.solution['u']['value'][1]),
            ),
            degree=problem.solution['u']['degree']
            )
    p_sol = Expression(
            _ccode(problem.solution['p']['value']),
            degree=problem.solution['p']['degree']
            )

    f = Expression(
            (
                _ccode(problem.f['value'][0]),
                _ccode(problem.f['value'][1]),
            ),
            degree=problem.f['degree']
            )

    W = VectorElement('Lagrange', mesh.ufl_cell(), 2)
    P = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    WP = FunctionSpace(mesh, W*P)

    # Get Dirichlet boundary conditions
    u_bcs = DirichletBC(WP.sub(0), u_sol, 'on_boundary')
    p_bcs = DirichletBC(WP.sub(1), p_sol, 'on_boundary')

    u_approx, p_approx = flow.stokes.solve(
        WP,
        bcs=[u_bcs, p_bcs],
        mu=1.0,
        f=f,
        verbose=True,
        tol=1.0e-12
        )

    # compute errors
    u_error = errornorm(u_sol, u_approx)
    p_error = errornorm(p_sol, p_approx)
    return mesh.hmax(), u_error, p_error


def show_errors(hmax, u_errors, p_errors):
    plt.loglog(hmax, u_errors, linestyle='-', marker='.', label='||u - uh||')
    plt.loglog(hmax, p_errors, linestyle='-', marker='.', label='||p - ph||')
    plt.xlabel('hmax')
    plt.legend()
    plt.show()
    return


# TODO add test for spatial order
@pytest.mark.parametrize('problem', [
    Guermond1(),
    ])
def test_order(problem, method):
    assert_order(problem)
    return


if __name__ == '__main__':
    mesh_sizes = [8, 16, 32]
    hmax, u_errors, p_errors = numpy.array([compute_error(
        Flat(),
        # Guermond1(),
        mesh_size
        )
        for mesh_size in mesh_sizes
        ]).T
    show_errors(hmax, u_errors, p_errors)
