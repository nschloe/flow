#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
'''
Coupled solve of the Navier--Stokes and the heat equation.
'''
import flow

from dolfin import (
        begin, end, Constant, norm, project, DOLFIN_EPS, grad, dot, dx, Mesh,
        FunctionSpace, DirichletBC, VectorElement, FiniteElement, SubDomain,
        TestFunction, TrialFunction, Function, assemble, KrylovSolver
        )
import materials
import meshio
import parabolic
import pygmsh


def create_mesh(lcar):
    geom = pygmsh.Geometry()

    x0 = 0.0
    x1 = 1.0
    y0 = 0.0
    y1 = 2.0

    circle = geom.add_circle([0.5, 0.5, 0.0], 0.2, lcar, make_surface=False)

    geom.add_rectangle(
        x0, x1, y0, y1,
        0.0,
        lcar,
        holes=[circle]
        )

    points, cells, point_data, cell_data, field_data = \
        pygmsh.generate_mesh(geom)

    # https://fenicsproject.org/qa/12891/initialize-mesh-from-vertices-connectivities-at-once
    meshio.write('test.xml', points, cells)

    mesh_eps = 1.0e-12

    class HotBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return (
                on_boundary
                and x0 + mesh_eps < x[0] < x1 - mesh_eps
                and y0 + mesh_eps < x[1] < y1 - mesh_eps
                )
    hot_boundary = HotBoundary()

    class CoolBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return (
                on_boundary and (
                    x[0] < x0 + mesh_eps or
                    x[0] > x1 - mesh_eps or
                    x[1] < y0 + mesh_eps or
                    x[1] < y1 - mesh_eps
                    ))
    cool_boundary = CoolBoundary()

    return Mesh('test.xml'), hot_boundary, cool_boundary


class Heat(object):
    def __init__(self, V, conv, kappa, rho, cp, bcs):
        # TODO stabilization
        # About stabilization for reaction-diffusion-convection:
        # http://www.ewi.tudelft.nl/fileadmin/Faculteit/EWI/Over_de_faculteit/Afdelingen/Applied_Mathematics/Rapporten/doc/06-03.pdf
        # http://www.xfem.rwth-aachen.de/Project/PaperDownload/Fries_ReviewStab.pdf
        #
        # R = u_t \
        #     + dot(u0, grad(trial)) \
        #     - 1.0/(rho(293.0)*cp) * div(kappa*grad(trial))
        # F -= R * dot(tau*u0, grad(v)) * dx
        #
        # Stabilization
        # tau = stab.supg2(
        #         mesh,
        #         u0,
        #         kappa/(rho(293.0)*cp),
        #         Q.ufl_element().degree()
        #         )
        self.V = V
        self.conv = conv
        self.kappa = kappa
        self.rho = rho
        self.cp = cp
        self.bcs = bcs
        return

    def eval_alpha_M_beta_F(self, alpha, beta, u, t):
        # Evaluate  alpha * M * u + beta * F(u, t).
        v = TestFunction(self.V)
        F = (
            + alpha * u * v * dx
            + beta * (
                - dot(self.conv, grad(u)) * v * dx
                - self.kappa
                    * dot(grad(u), grad(v/(self.rho*self.cp))) * dx
                )
            )
        return assemble(F)

    def solve_alpha_M_beta_F(self, alpha, beta, b, t):
        # Solve  alpha * M * u + beta * F(u, t) = b  for u.
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        F = (
            + alpha * u * v * dx
            + beta * (
                - dot(self.conv, grad(u)) * v * dx
                - self.kappa
                    * dot(grad(u), grad(v/(self.rho*self.cp))) * dx
                )
            )

        A = assemble(F)
        for bc in self.bcs:
            bc.apply(A, b)

        solver = KrylovSolver('gmres', 'ilu')
        solver.parameters['relative_tolerance'] = 1.0e-13
        solver.parameters['absolute_tolerance'] = 0.0
        solver.parameters['maximum_iterations'] = 100
        solver.parameters['monitor_convergence'] = True
        solver.set_operator(A)

        u = Function(self.V)
        solver.solve(u.vector(), b)
        return u


def test_boussinesq(target_time=0.1, lcar=0.1):
    mesh, hot_boundary, cool_boundary = create_mesh(lcar)

    # Density depends on temperature.
    rho = materials.water.density
    # Take dynamic viscosity at room temperature.
    mu = materials.water.dynamic_viscosity(293.0)
    cp = materials.water.specific_heat_capacity
    kappa = materials.water.thermal_conductivity

    # Start time, end time, time step.
    dt_max = 0.1
    dt0 = 1.0e-2
    # This should be
    # umax = 1.0e-1
    # dt0 = 0.2 * mesh.hmin() / umax
    t = 0.0

    room_temp = 293.0
    max_heater_temp = 380.0

    # Gravity force.
    g = Constant((0.0, -9.81))

    W_element = VectorElement('Lagrange', mesh.ufl_cell(), 2)
    P_element = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    WP = FunctionSpace(mesh, W_element * P_element)

    Q = FunctionSpace(mesh, 'Lagrange', 2)

    u_bcs = [DirichletBC(WP.sub(0), (0.0, 0.0), 'on_boundary')]
    p_bcs = []

    # Solve Stokes for initial state.
    u0, p0 = flow.stokes.solve(
        WP,
        u_bcs + p_bcs,
        mu,
        f=g,
        verbose=False,
        tol=1.0e-13,
        max_iter=1000
        )

    # Everything at room temperature for starters
    theta0 = project(Constant(293.0), Q)

    # div_u = Function(Q)
    dt = dt0

    while t < target_time + DOLFIN_EPS:
        begin('Time step %e -> %e...' % (t, t+dt))

        # Crank up the heater from room_temp to max_heater_temp in t1 secs.
        t1 = 1.0
        heater_temp = (
            + room_temp
            + min(1.0, t/t1) * (max_heater_temp - room_temp)
            )
        # heater_temp = max_heater_temp

        # Do one heat time step.
        begin('Computing heat...')
        heat_bcs = [
            DirichletBC(Q, heater_temp, hot_boundary),
            DirichletBC(Q, 293.0, cool_boundary),
            ]
        # Use all quanities at room temperature to avoid nonlinearity
        stepper = parabolic.ImplicitEuler(
                Heat(Q, u0, kappa(293.0), rho(293.0), cp(293.0), heat_bcs)
                )
        theta = stepper.step(theta0, t, dt)

        # Do one Navier-Stokes time step.
        begin('Computing flux and pressure...')
        stepper = flow.navier_stokes.IPCS(
                rho(theta), mu,
                theta=1.0  # fully implicit step
                )

        W = u0.function_space()
        u_bcs = [DirichletBC(W, (0.0, 0.0), 'on_boundary')]
        p_bcs = []

        import time
        start_time = time.time()
        try:
            u, p = stepper.step(
                    dt,
                    u0, p0,
                    u_bcs, p_bcs,
                    f0=g,
                    f1=g,
                    verbose=False,
                    tol=1.0e-10
                    )
        except RuntimeError as e:
            print(e.message)
            print('Navier--Stokes solver failed to converge. '
                  'Decrease time step from %e to %e and try again.' %
                  (dt, 0.5*dt)
                  )
            dt *= 0.5
            end()
            end()
            end()
            continue
        elapsed_time = time.time() - start_time
        print('elapsed: %e' % elapsed_time)

        # u = TrialFunction(Q)
        # v = TestFunction(Q)
        # solve(u*v*dx == div(u)*v*dx, div_u)
        # div_u.assign(div(u0))
        # plot(div_u, title='div(u)', rescale=True)
        # interactive()
        end()

        # Assigning and plotting. We do that here so all methods have access
        # to `x` and `x_1` (necessary, for example, for Crank-Nicolson in
        # Navier-Stokes).
        theta0.assign(theta)
        u0.assign(u)
        p0.assign(p)

        # Adaptive stepsize control based solely on the velocity field.
        # CFL-like condition for time step. This should be some sort of average
        # of the temperature in the current step and the target step.
        #
        # More on step-size control for Navier--Stokes:
        #
        #     Adaptive time step control for the incompressible Navier-Stokes
        #     equations;
        #     Volker John, Joachim Rang;
        #     Comput. Methods Appl. Mech. Engrg. 199 (2010) 514-524;
        #     <http://www.wias-berlin.de/people/john/ELECTRONIC_PAPERS/JR10.CMAME.pdf>.
        #
        # Section 3.3 in that paper notes that time-adaptivity for theta-
        # schemes is too costly. They rather reside to DIRK- and Rosenbrock-
        # methods.
        #
        begin('Step size adaptation...')
        u1, u2 = u.split()
        unorm = project(
                abs(u1) + abs(u2),
                Q,
                form_compiler_parameters={'quadrature_degree': 4}
                )
        unorm = norm(unorm.vector(), 'linf')
        # print('||u||_inf = %e' % unorm)
        # Some smooth step-size adaption.
        target_dt = 0.2 * mesh.hmax() / unorm
        print('current dt: %e' % dt)
        print('target dt:  %e' % target_dt)
        # alpha is the aggressiveness factor. The distance between the current
        # step size and the target step size is reduced by |1-alpha|. Hence,
        # if alpha==1 then dt_next==target_dt. Otherwise target_dt is
        # approached slowlier.
        alpha = 0.5
        dt = min(
            dt_max,
            # At most double the step size from step to step.
            dt * min(2.0, 1.0 + alpha*(target_dt - dt)/dt)
            )
        print('next dt:    %e' % dt)
        t += dt
        end()
        end()
    return


if __name__ == '__main__':
    test_boussinesq(target_time=120.0, lcar=0.1)
