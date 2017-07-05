#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
'''
Coupled solve of the Navier--Stokes and the heat equation.
'''
from __future__ import print_function

import os

import flow
from flow import heat

from dolfin import (
    begin, end, Constant, norm, project, DOLFIN_EPS, Mesh, FunctionSpace,
    DirichletBC, VectorElement, FiniteElement, SubDomain, Function, XDMFFile,
    SpatialCoordinate, mpi_comm_world, info
    )
import materials
import meshio
import parabolic
import pygmsh


def create_mesh(lcar):

    x0 = 0.0
    x1 = 0.1
    y0 = 0.0
    y1 = 0.2

    mesh_eps = 1.0e-12

    # pylint: disable=no-self-use
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
                    x[1] > y1 - mesh_eps
                    ))
    cool_boundary = CoolBoundary()

    cache_file = 'boussinesq-{}.msh'.format(lcar)
    if os.path.isfile(cache_file):
        print('Using mesh from cache \'{}\'.'.format(cache_file))
        points, cells, _, _, _ = meshio.read(cache_file)
    else:
        geom = pygmsh.Geometry()

        circle = geom.add_circle(
            [0.05, 0.05, 0.0], 0.02, lcar, make_surface=False
            )

        geom.add_rectangle(
            x0, x1, y0, y1,
            0.0,
            lcar,
            holes=[circle]
            )

        points, cells, _, _, _ = pygmsh.generate_mesh(geom)

        meshio.write(cache_file, points, cells)

    # https://fenicsproject.org/qa/12891/initialize-mesh-from-vertices-connectivities-at-once
    meshio.write('test.xml', points, cells)
    return Mesh('test.xml'), hot_boundary, cool_boundary


def test_boussinesq():
    u1, _, theta1 = compute_boussinesq(target_time=1.0, lcar=0.1, supg=False)
    ref = 7.717866694234539e-06
    assert abs(norm(u1, 'L2') - ref) < 1.0e-6 * ref
    ref = 40.35638391160463
    assert abs(norm(theta1, 'L2') - ref) < 1.0e-6 * ref
    return


def test_boussinesq_with_supg():
    u1, _, theta1 = compute_boussinesq(target_time=1.0, lcar=0.1, supg=True)
    ref = 7.718099717545708e-06
    assert abs(norm(u1, 'L2') - ref) < 1.0e-6 * ref
    ref = 40.35638391160463
    assert abs(norm(theta1, 'L2') - ref) < 1.0e-6 * ref
    return


def compute_boussinesq(target_time, lcar, supg=False):
    mesh, hot_boundary, cool_boundary = create_mesh(lcar)

    room_temp = 293.0

    # Density depends on temperature.
    rho = materials.water.density
    # Take dynamic viscosity at room temperature.
    mu = materials.water.dynamic_viscosity(room_temp)
    cp = materials.water.specific_heat_capacity
    kappa = materials.water.thermal_conductivity

    # Start time, end time, time step.
    dt_max = 1.0
    dt0 = 1.0e-2
    # This should be
    # umax = 1.0e-1
    # dt0 = 0.2 * mesh.hmin() / umax
    t = 0.0

    max_heater_temp = 320.0

    # Gravity accelleration.
    accelleration_constant = -9.81
    g = Constant((0.0, accelleration_constant))

    W_element = VectorElement('Lagrange', mesh.ufl_cell(), 2)
    P_element = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    WP = FunctionSpace(mesh, W_element * P_element)

    Q = FunctionSpace(mesh, 'Lagrange', 2)

    # Everything at room temperature for starters
    theta0 = project(Constant(room_temp), Q)
    theta0.rename('temperature', 'temperature')

    u_bcs = [DirichletBC(WP.sub(0), (0.0, 0.0), 'on_boundary')]
    p_bcs = []

    # Solve Stokes for initial state.
    # Make sure that the right-hand side is the same as in the first step of
    # Navier-Stokes below. This avoids problems with the initial pressure being
    # a bit off, which leads to errors.
    # u0, p0 = flow.stokes.solve(
    #     WP,
    #     u_bcs + p_bcs,
    #     mu,
    #     f=rho(theta0) * g,
    #     verbose=False,
    #     tol=1.0e-15,
    #     max_iter=1000
    #     )
    y = SpatialCoordinate(mesh)[1]
    u0 = project(Constant([0, 0]), FunctionSpace(mesh, W_element))
    u0.rename('velocity', 'velocity')
    p0 = project(
        rho(theta0) * accelleration_constant * y,
        FunctionSpace(mesh, P_element)
        )
    p0.rename('pressure', 'pressure')

    # div_u = Function(Q)
    dt = dt0

    with XDMFFile(mpi_comm_world(), 'boussinesq.xdmf') as xdmf_file:
        xdmf_file.parameters['flush_output'] = True
        xdmf_file.parameters['rewrite_function_mesh'] = False

        while t < target_time + DOLFIN_EPS:
            begin('Time step %e -> %e...' % (t, t+dt))

            # Crank up the heater from room_temp to max_heater_temp in t1 secs.
            t1 = 30.0
            heater_temp = (
                + room_temp
                + min(1.0, t/t1) * (max_heater_temp - room_temp)
                )
            # heater_temp = room_temp

            # Velocity and heat and connected by
            #
            #    theta1 = F_theta(u1, theta0)
            #    u1, p1 = F_u(u0, p0, theta1)
            #
            # One can either replace u1, theta1 on the right-hand side by u0,
            # theta0, respectively, or wrap the whole thing in a Banach
            # iteration 'a la
            #
            #    theta = F_theta(u_prev, theta0)
            #    (u, p) = F_u(u0, p0, theta_prev)
            #
            # and do that until the residuals are close to 0.
            u_prev = Function(u0.function_space())
            u_prev.assign(u0)

            theta_prev = Function(theta0.function_space())
            theta_prev.assign(theta0)
            is_banach_converged = False
            banach_tol = 1.0e-1
            max_banach_steps = 10
            target_banach_steps = 5
            banach_step = 0
            while not is_banach_converged:
                banach_step += 1
                if banach_step > max_banach_steps:
                    info('\nBanach solver failed to converge. '
                         'Decrease time step from %e to %e and try again.\n' %
                         (dt, 0.25*dt)
                         )
                    dt *= 0.25
                    end()  # time step
                    break
                begin('Banach step %d:' % banach_step)
                # Do one heat time step.
                begin('Computing heat...')
                heat_bcs = [
                    DirichletBC(Q, heater_temp, hot_boundary),
                    DirichletBC(Q, room_temp, cool_boundary),
                    ]
                # Use all quantities at room temperature to avoid nonlinearity
                stepper = parabolic.ImplicitEuler(
                        heat.Heat(
                            Q, u_prev,
                            kappa(room_temp), rho(room_temp), cp(room_temp),
                            heat_bcs, Constant(0.0),
                            supg_stabilization=supg
                            )
                        )

                theta1 = stepper.step(theta0, t, dt)
                end()

                # Do one Navier-Stokes time step.
                begin('Computing flux and pressure...')
                # stepper = flow.navier_stokes.Chorin()
                # stepper = flow.navier_stokes.IPCS()
                stepper = flow.navier_stokes.Rotational()
                W = u0.function_space()
                u_bcs = [DirichletBC(W, (0.0, 0.0), 'on_boundary')]
                p_bcs = []
                try:
                    u1, p1 = stepper.step(
                            Constant(dt),
                            {0: u0}, p0,
                            u_bcs, p_bcs,
                            # TODO use rho(theta)
                            rho(room_temp), Constant(mu),
                            f={
                                0: rho(theta_prev) * g,
                                1: rho(theta_prev) * g
                                },
                            verbose=False,
                            tol=1.0e-10
                            )
                except RuntimeError:
                    info('Navier--Stokes solver failed to converge. '
                         'Decrease time step from %e to %e and try again.' %
                         (dt, 0.5*dt)
                         )
                    dt *= 0.5
                    end()  # navier-stokes
                    end()  # banach step
                    end()  # time step
                    # http://stackoverflow.com/a/1859099/353337
                    break
                end()  # navier-stokes

                u1x, u1y = u1.split()
                uprevx, uprevy = u_prev.split()
                unorm = project(
                        abs(u1x - uprevx) + abs(u1y - uprevy),
                        Q,
                        form_compiler_parameters={'quadrature_degree': 4}
                        )
                u_diff_norm = norm(unorm.vector(), 'linf')

                theta_diff = Function(theta1.function_space())
                theta_diff.vector()[:] = theta1.vector() - theta_prev.vector()
                theta_diff_norm = norm(theta_diff.vector(), 'linf')

                info('Banach residuals:')
                info('   ||u - u_prev||         = %e' % u_diff_norm)
                info('   ||theta - theta_prev|| = %e' % theta_diff_norm)

                is_banach_converged = \
                    u_diff_norm < banach_tol and theta_diff_norm < banach_tol

                u_prev.assign(u1)
                theta_prev.assign(theta1)
                end()  # banach step
            else:
                # from dolfin import plot, interactive
                # plot(u0)
                # plot(p0)
                # u1.rename('u1', 'u1')
                # plot(u1)
                # p1.rename('p1', 'p1')
                # plot(p1)
                # interactive()

                # Assigning and plotting. We do that here so all methods have
                # access to `x` and `x_1`.
                theta0.assign(theta1)
                u0.assign(u1)
                p0.assign(p1)

                # write to file
                xdmf_file.write(theta0, t)
                xdmf_file.write(u0, t)
                xdmf_file.write(p0, t)

                # from dolfin import plot, interactive
                # plot(theta0, title='theta')
                # plot(u0, title='u')
                # # plot(div(u), title='div(u)', rescale=True)
                # plot(p0, title='p')
                # interactive()

                end()  # time step

                begin('\nStep size adaptation...')
                # Step-size control can be done purely based on the velocity
                # field; see
                #
                #     Adaptive time step control for the incompressible
                #     Navier-Stokes equations;
                #     Volker John, Joachim Rang;
                #     Comput. Methods Appl. Mech. Engrg. 199 (2010) 514-524;
                #     <http://www.wias-berlin.de/people/john/ELECTRONIC_PAPERS/JR10.CMAME.pdf>.
                #
                # Section 3.3 in that paper notes that time-adaptivity for
                # theta- schemes is too costly. They rather reside to DIRK- and
                # Rosenbrock- methods.
                #
                # Implementation:
                #   ux, uy = u0.split()
                #   unorm = project(
                #           abs(ux) + abs(uy),
                #           Q,
                #           form_compiler_parameters={'quadrature_degree': 4}
                #           )
                #   unorm = norm(unorm.vector(), 'linf')
                #   # print('||u||_inf = %e' % unorm)
                #   # Some smooth step-size adaption.
                #   target_dt = 0.2 * mesh.hmax() / unorm

                # In our case, step failures are almost always because Banach
                # didn't converge. Hence, design a step size adaptation based
                # on the Banach steps.
                target_dt = dt * target_banach_steps / banach_step

                info('current dt: %e' % dt)
                info('target dt:  %e' % target_dt)
                # alpha is the aggressiveness factor. The distance between the
                # current step size and the target step size is reduced by
                # |1-alpha|. Hence, if alpha==1 then dt_next==target_dt.
                # Otherwise target_dt is approached more slowly.
                alpha = 0.5
                dt = min(
                    dt_max,
                    # At most double the step size from step to step.
                    dt * min(2.0, 1.0 + alpha*(target_dt - dt)/dt)
                    )
                info('next dt:    %e\n' % dt)
                t += dt
                end()

    return u1, p1, theta1


if __name__ == '__main__':
    compute_boussinesq(target_time=120.0, lcar=0.3e-2)
