# -*- coding: utf-8 -*-
#
from helpers import ccode, compute_numerical_order_of_convergence
import flow.navier_stokes as navsto

from dolfin import (
    UnitSquareMesh, triangle, RectangleMesh, pi, Point, Expression, assemble,
    dx, errornorm, plot, interactive, interpolate, project, DirichletBC,
    Function, FunctionSpace, VectorFunctionSpace
    )
import matplotlib.pyplot as plt
import numpy
import pytest
import sympy
import warnings


MAX_DEGREE = 5


def _truncate_degree(degree, max_degree=10):
    if degree > max_degree:
        warnings.warn(
            'Expression degree (%r) > maximum degree (%d). Truncating.'
            % (degree, max_degree)
            )
        return max_degree
    return degree


def _get_navier_stokes_rhs(u, p):
    '''Given a solution u of the Cartesian Navier-Stokes equations, return
    a matching right-hand side f.
    '''
    x = sympy.DeferredVector('x')
    t, mu, rho = sympy.symbols('t, mu, rho')

    # Make sure that the exact solution is indeed analytically div-free.
    d = sympy.diff(u[0], x[0]) + sympy.diff(u[1], x[1])
    d = sympy.simplify(d)
    assert d == 0

    # Get right-hand side associated with this solution, i.e., according
    # the Navier-Stokes
    #
    #     rho (du_x/dt + u_x du_x/dx + u_y du_x/dy)
    #         = - dp/dx + mu [d^2u_x/dx^2 + d^2u_x/dy^2] + f_x,
    #     rho (du_y/dt + u_x du_y/dx + u_y du_y/dy)
    #         = - dp/dx + mu [d^2u_y/dx^2 + d^2u_y/dy^2] + f_y,
    #     du_x/dx + du_y/dy = 0.
    #
    #     rho (du/dt + (u.\nabla)u) = -\nabla p + mu [\div(\nabla u)] + f,
    #     div(u) = 0.
    #
    f0 = rho * (sympy.diff(u[0], t)
                + u[0] * sympy.diff(u[0], x[0])
                + u[1] * sympy.diff(u[0], x[1])
                ) \
        + sympy.diff(p, x[0]) \
        - mu * (sympy.diff(u[0], x[0], 2) + sympy.diff(u[0], x[1], 2))
    f1 = rho * (sympy.diff(u[1], t)
                + u[0] * sympy.diff(u[1], x[0])
                + u[1] * sympy.diff(u[1], x[1])
                ) \
        + sympy.diff(p, x[1]) \
        - mu * (sympy.diff(u[1], x[0], 2) + sympy.diff(u[1], x[1], 2))

    f = (
        sympy.simplify(f0),
        sympy.simplify(f1)
        )
    return f


def problem_flat():
    '''Nothing interesting happening in the domain.
    '''
    def mesh_generator(n):
        return UnitSquareMesh(n, n, 'left/right')
    cell_type = triangle
    x = sympy.DeferredVector('x')
    u = (0.0 * x[0], 0.0 * x[1])

    # Only grad(p) is of physical significance, and the numerical procedure
    # chooses p such that int(p) = 0. Also, numerical solutions have n.grad(p)
    # = 0 for non-penetration boundaries.  Make sure that the exact solution
    # respects those things to make error analysis easier.
    p = sympy.cos(pi * x[1])

    solution = {
        'u': {'value': u, 'degree': 1},
        'p': {'value': p, 'degree': 1},
        }
    f = {
        'value': _get_navier_stokes_rhs(u, p),
        'degree': MAX_DEGREE,
        }
    mu = 1.0
    rho = 1.0
    return mesh_generator, solution, f, mu, rho, cell_type


def problem_whirl():
    '''Example from Teodora I. Mitkova's text
    "Finite-Elemente-Methoden fur die Stokes-Gleichungen".
    '''
    def mesh_generator(n):
        return UnitSquareMesh(n, n, 'left/right')
    cell_type = triangle
    x = sympy.DeferredVector('x')
    # t = sympy.symbols('t')

    # Note that the exact solution is indeed div-free.
    u = (
        x[0]**2 * (1 - x[0])**2 * 2 * x[1] * (1 - x[1]) * (2 * x[1] - 1),
        x[1]**2 * (1 - x[1])**2 * 2 * x[0] * (1 - x[0]) * (1 - 2 * x[0])
        )
    p = x[0] * (1 - x[0]) * x[1] * (1 - x[1])
    solution = {
        'u': {'value': u, 'degree': 7},
        'p': {'value': p, 'degree': 4},
        }
    f = {
        'value': _get_navier_stokes_rhs(u, p),
        'degree': MAX_DEGREE,
        }
    mu = 1.0
    rho = 1.0
    return mesh_generator, solution, f, mu, rho, cell_type


def problem_guermond1():
    '''Problem 1 from section 3.7.1 in
        An overview of projection methods for incompressible flows;
        Guermond, Minev, Shen;
        Comp. Meth. in Appl. Mech. and Eng., vol. 195, 44-47, pp. 6011-6045;
        <http://www.sciencedirect.com/science/article/pii/S0045782505004640>.
    '''
    def mesh_generator(n):
        return RectangleMesh(Point(-1, -1), Point(1, 1), n, n, 'crossed')
    cell_type = triangle
    x = sympy.DeferredVector('x')
    t = sympy.symbols('t')
    # m = sympy.exp(t) - 0.0
    m = sympy.sin(t)
    u = (
        +pi*m*2*sympy.sin(pi*x[1])*sympy.cos(pi*x[1])*sympy.sin(pi*x[0])**2,
        -pi*m*2*sympy.sin(pi*x[0])*sympy.cos(pi*x[0])*sympy.sin(pi*x[1])**2
        )
    p = m * sympy.cos(pi * x[0]) * sympy.sin(pi * x[1])
    solution = {
        'u': {'value': u, 'degree': MAX_DEGREE},
        'p': {'value': p, 'degree': MAX_DEGREE}
        }
    f = {
        'value': _get_navier_stokes_rhs(u, p),
        'degree': MAX_DEGREE
        }
    mu = 1.0
    rho = 1.0
    return mesh_generator, solution, f, mu, rho, cell_type


def problem_guermond2():
    '''Problem 2 from section 3.7.2 in
        An overview of projection methods for incompressible flows;
        Guermond, Minev, Shen;
        Comp. Meth. in Appl. Mech. and Eng., vol. 195, 44-47, pp. 6011-6045;
        <http://www.sciencedirect.com/science/article/pii/S0045782505004640>.
    '''
    def mesh_generator(n):
        return UnitSquareMesh(n, n, 'crossed')
    cell_type = triangle
    x = sympy.DeferredVector('x')
    t = sympy.symbols('t')
    u = (
        sympy.sin(x[0] + t) * sympy.sin(x[1] + t),
        sympy.cos(x[0] + t) * sympy.cos(x[1] + t)
        )
    p = sympy.sin(x[0] - x[1] + t)
    solution = {
        'u': {'value': u, 'degree': MAX_DEGREE},
        'p': {'value': p, 'degree': MAX_DEGREE},
        }
    f = {
        'value': _get_navier_stokes_rhs(u, p),
        'degree': MAX_DEGREE,
        }
    mu = 1.0
    rho = 1.0
    return mesh_generator, solution, f, mu, rho, cell_type


def problem_taylor():
    '''Taylor--Green vortex, cf.
    <http://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex>.
    '''
    def mesh_generator(n):
        return RectangleMesh(
            Point(0.0, 0.0), Point(2*pi, 2*pi),
            n, n, 'crossed'
            )
    mu = 1.0
    rho = 1.0
    cell_type = triangle
    x = sympy.DeferredVector('x')
    t = sympy.symbols('t')
    x0 = x[0]
    x1 = x[1]
    # F = sympy.exp(-2*mu*t)
    F = 1 - 2*mu*t
    u = (sympy.sin(x0) * sympy.cos(x1) * F,
         -sympy.cos(x0) * sympy.sin(x1) * F,
         0
         )
    p = rho/4 * (sympy.cos(2*x0) + sympy.cos(2*x1)) * F**2
    solution = {
            'u': {'value': u, 'degree': MAX_DEGREE},
            'p': {'value': p, 'degree': MAX_DEGREE},
            }
    f = {
        'value': _get_navier_stokes_rhs(u, p),
        'degree': MAX_DEGREE,
        }
    return mesh_generator, solution, f, mu, rho, cell_type


def compute_time_errors(problem, method, mesh_sizes, Dt):

    mesh_generator, solution, f, mu, rho, cell_type = problem()

    # Translate data into FEniCS expressions.
    u = solution['u']
    sol_u = Expression(
            (ccode(u['value'][0]), ccode(u['value'][1])),
            degree=_truncate_degree(u['degree']),
            t=0.0
            )

    p = solution['p']
    sol_p = Expression(
            ccode(p['value']),
            degree=_truncate_degree(p['degree']),
            t=0.0
            )

    # Deep-copy expression to be able to provide f0, f1 for the Dirichlet-
    # boundary conditions later on.
    fenics_rhs0 = Expression(
            (ccode(f['value'][0]), ccode(f['value'][1])),
            degree=_truncate_degree(f['degree']),
            t=0.0, mu=mu, rho=rho
            )
    fenics_rhs1 = Expression(
            fenics_rhs0.cppcode,
            element=fenics_rhs0.ufl_element(),
            **fenics_rhs0.user_parameters
            )
    # Create initial states.
    p = Expression(
        sol_p.cppcode,
        degree=_truncate_degree(solution['p']['degree']),
        t=0.0,
        cell=cell_type
        )

    # Compute the problem
    errors = {
        'u': numpy.empty((len(mesh_sizes), len(Dt))),
        'p': numpy.empty((len(mesh_sizes), len(Dt)))
        }
    for k, mesh_size in enumerate(mesh_sizes):
        print
        print
        print('Computing for mesh size %r...' % mesh_size)
        mesh = mesh_generator(mesh_size)
        mesh_area = assemble(1.0 * dx(mesh))
        W = VectorFunctionSpace(mesh, 'CG', 2)
        P = FunctionSpace(mesh, 'CG', 1)
        u1 = Function(W)
        p1 = Function(P)
        err_p = Function(P)
        divu1 = Function(P)
        for j, dt in enumerate(Dt):
            # Prepare previous states for multistepping.
            u = Expression(
                sol_u.cppcode,
                degree=_truncate_degree(solution['u']['degree']),
                t=0.0,
                # t=0.5*dt,
                cell=cell_type
                )
            u0 = project(u, W)
            sol_u.t = dt
            u_bcs = [DirichletBC(W, sol_u, 'on_boundary')]
            sol_p.t = dt
            p0 = project(p, P)
            p_bcs = []
            fenics_rhs0.t = 0.0
            fenics_rhs1.t = dt
            u1, p1 = method.step(
                    dt,
                    u0, p0,
                    u_bcs=u_bcs, p_bcs=p_bcs,
                    rho=rho, mu=mu,
                    f0=fenics_rhs0,
                    f1=fenics_rhs1,
                    verbose=False,
                    tol=1.0e-10
                    )

            # plot(sol_u, mesh=mesh, title='u_sol')
            # plot(sol_p, mesh=mesh, title='p_sol')
            # plot(u1, title='u')
            # plot(p1, title='p')
            # from dolfin import div
            # plot(div(u1), title='div(u)')
            # plot(p1 - sol_p, title='p_h - p')
            # interactive()

            sol_u.t = dt
            sol_p.t = dt
            errors['u'][k][j] = errornorm(sol_u, u1)
            # The pressure is only determined up to a constant which makes
            # it a bit harder to define what the error is. For our
            # purposes, choose an alpha_0\in\R such that
            #
            #    alpha0 = argmin ||e - alpha||^2
            #
            # with  e := sol_p - p.
            # This alpha0 is unique and explicitly given by
            #
            #     alpha0 = 1/(2|Omega|) \int (e + e*)
            #            = 1/|Omega| \int Re(e),
            #
            # i.e., the mean error in \Omega.
            alpha = (
                + assemble(sol_p * dx(mesh))
                - assemble(p1 * dx(mesh))
                )
            alpha /= mesh_area
            # We would like to perform
            #     p1 += alpha.
            # To avoid creating a temporary function every time, assume
            # that p1 lives in a function space where the coefficients
            # represent actual function values. This is true for CG
            # elements, for example. In that case, we can just add any
            # number to the vector of p1.
            p1.vector()[:] += alpha
            errors['p'][k][j] = errornorm(sol_p, p1)

            show_plots = False
            if show_plots:
                plot(p1, title='p1', mesh=mesh)
                plot(sol_p, title='sol_p', mesh=mesh)
                err_p.vector()[:] = p1.vector()
                sol_interp = interpolate(sol_p, P)
                err_p.vector()[:] -= sol_interp.vector()
                # plot(sol_p - p1, title='p1 - sol_p', mesh=mesh)
                plot(err_p, title='p1 - sol_p', mesh=mesh)
                # r = Expression('x[0]', degree=1, cell=triangle)
                # divu1 = 1 / r * (r * u1[0]).dx(0) + u1[1].dx(1)
                divu1.assign(project(u1[0].dx(0) + u1[1].dx(1), P))
                plot(divu1, title='div(u1)')
                interactive()
    return errors


@pytest.mark.parametrize('problem', [
    problem_flat,
    # problem_whirl,
    problem_guermond1,
    problem_guermond2,
    # problem_taylor,
    ])
@pytest.mark.parametrize('stabilization', [True, False])
def test_chorin(problem, stabilization, tol=1.0e-10):
    Dt = [1.0e-3, 0.5e-3]
    mesh_sizes = [16, 32]
    assert_time_order(
            problem, navsto.Chorin(stabilization=stabilization), tol=tol,
            Dt=Dt,
            mesh_sizes=mesh_sizes
            )
    return


@pytest.mark.parametrize('problem', [
    # problem_flat,
    # problem_whirl,
    problem_guermond2,
    # problem_taylor,
    ])
@pytest.mark.parametrize('stabilization', [True, False])
def test_ipcs(problem, stabilization, tol=1.0e-10):
    assert_time_order(
        problem,
        navsto.IPCS(stabilization=stabilization, theta=1.0),
        tol
        )
    return


def assert_time_order(
        problem, method,
        tol=1.0e-10,
        mesh_sizes=[8, 16, 32],
        Dt=[0.5**k for k in range(2)]
        ):
    errors = compute_time_errors(problem, method, mesh_sizes, Dt)
    orders = {
        key: compute_numerical_order_of_convergence(Dt, errors[key].T).T
        for key in errors
        }
    # The test is considered passed if the numerical order of convergence
    # matches the expected order in at least the first step in the coarsest
    # spatial discretization, and is not getting worse as the spatial
    # discretizations are refining.
    assert (orders['u'][:, 0] > method.order['velocity'] - 0.1).all()
    assert (orders['p'][:, 0] > method.order['pressure'] - 0.1).all()
    return


def show_timeorder_info(Dt, mesh_sizes, errors):
    '''Performs consistency check for the given problem/method combination and
    show some information about it. Useful for debugging.
    '''
    # Compute the numerical order of convergence.
    orders = {
        key: compute_numerical_order_of_convergence(Dt, errors[key].T).T
        for key in errors
        }

    # Print the data to the screen
    for i, mesh_size in enumerate(mesh_sizes):
        print
        print('Mesh size %d:' % mesh_size)
        print('dt = %e' % Dt[0]),
        for label, e in errors.items():
            print('   err_%s = %e' % (label, e[i][0])),
        print
        for j in range(len(Dt) - 1):
            print('                 '),
            for label, o in orders.items():
                print('   ord_%s = %e' % (label, o[i][j])),
            print
            print('dt = %e' % Dt[j+1]),
            for label, e in errors.items():
                print('   err_%s = %e' % (label, e[i][j+1])),
            print

    # Create a figure
    for label, err in errors.items():
        plt.figure()
        # ax = plt.axes()
        # Plot the actual data.
        for i, mesh_size in enumerate(mesh_sizes):
            plt.loglog(Dt, err[i], '-o', label=mesh_size)
        # Compare with order curves.
        plt.autoscale(False)
        e0 = err[-1][0]
        for o in range(7):
            plt.loglog(
                    [Dt[0], Dt[-1]],
                    [e0, e0 * (Dt[-1] / Dt[0]) ** o],
                    color='0.7'
                    )
        plt.xlabel('dt')
        plt.ylabel('||%s-%s_h||' % (label, label))
        # plt.title('Method: %s' % method['name'])
        plt.legend()
    plt.show()
    return


if __name__ == '__main__':
    mesh_sizes = [8, 16, 32, 64]
    Dt = [0.5**k for k in range(20)]
    errors = compute_time_errors(
        # problem_flat,
        # problem_whirl,
        # problem_guermond1,
        problem_guermond2,
        # problem_taylor,
        #
        # navsto.Chorin(),
        navsto.IPCS(theta=1.0),
        mesh_sizes, Dt
        )
    show_timeorder_info(Dt, mesh_sizes, errors)
