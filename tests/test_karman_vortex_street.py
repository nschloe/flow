# -*- coding: utf-8 -*-
#
import os

from dolfin import (
        Mesh, SubDomain, FunctionSpace, DirichletBC, VectorElement,
        FiniteElement, Constant, plot, begin, end, project, norm, XDMFFile,
        sqrt, Expression, mpi_comm_world
        )
import materials
import meshio
import pygmsh

import flow

x0 = 0.0
x1 = 0.6
y0 = -0.07
y1 = 0.07
obstacle_diameter = 0.04
entrance_velocity = 0.01


def create_mesh(lcar):
    geom = pygmsh.Geometry()

    cache_file = 'karman.msh'
    if os.path.isfile(cache_file):
        print('Using mesh from cache \'{}\'.'.format(cache_file))
        points, cells, _, _, _ = meshio.read(cache_file)
    else:
        # slightly off-center circle
        circle = geom.add_circle(
            [0.1, 1.0e-2, 0.0], 0.5 * obstacle_diameter, lcar,
            make_surface=False
            )

        geom.add_rectangle(
            x0, x1, y0, y1,
            0.0,
            lcar,
            holes=[circle]
            )

        points, cells, point_data, cell_data, field_data = \
            pygmsh.generate_mesh(geom)

        meshio.write(cache_file, points, cells)

    # https://fenicsproject.org/qa/12891/initialize-mesh-from-vertices-connectivities-at-once
    meshio.write('test.xml', points, cells)
    return Mesh('test.xml')


def test_karman(num_steps=2, lcar=0.1, show=False):
    mesh = create_mesh(lcar)

    W_element = VectorElement('Lagrange', mesh.ufl_cell(), 2)
    P_element = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    WP = FunctionSpace(mesh, W_element * P_element)

    W = WP.sub(0)
    # P = WP.sub(1)

    mesh_eps = 1.0e-12

    # Define mesh and boundaries.
    class LeftBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] < x0 + mesh_eps
    left_boundary = LeftBoundary()

    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] > x1 - mesh_eps
    right_boundary = RightBoundary()

    class LowerBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] < y0 + mesh_eps
    lower_boundary = LowerBoundary()

    class UpperBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] > y1 - mesh_eps
    upper_boundary = UpperBoundary()

    class ObstacleBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return (
                on_boundary
                and x0 + mesh_eps < x[0] < x1 - mesh_eps
                and y0 + mesh_eps < x[1] < y1 - mesh_eps
                )
    obstacle_boundary = ObstacleBoundary()

    # Boundary conditions for the velocity.
    # Proper inflow and outflow conditions are a matter of voodoo. See for
    # example Gresho/Sani, or
    #
    #     Boundary conditions for open boundaries for the incompressible
    #     Navier-Stokes equation;
    #     B.C.V. Johansson;
    #     J. Comp. Phys. 105, 233-251 (1993).
    #
    # The latter in particularly suggest for the inflow:
    #
    #     u = u0,
    #     d^r v / dx^r = v_r,
    #     div(u) = 0,
    #
    # where u and v are the velocities in normal and tangential directions,
    # respectively, and r\in{0,1,2}. The setting r=0 essentially means to set
    # (u,v) statically at the left boundary, r=1 means to set u and control
    # dv/dn, which is what we do here (namely implicitly by dv/dn=0).
    # At the outflow,
    #
    #     d^j u / dx^j = 0,
    #     d^q v / dx^q = 0,
    #     p = p0,
    #
    # is suggested with j=q+1. Choosing q=0, j=1 means setting the tangential
    # component of the outflow to 0, and letting the normal component du/dn=0
    # (again, this is achieved implicitly by the weak formulation).
    #
    inflow = Expression(
            '%e * (%e - x[1]) * (x[1] - %e) / %e' %
            (entrance_velocity, y1, y0, (0.5 * (y1 - y0))**2),
            degree=2
            )
    outflow = Expression(
            '%e * (%e - x[1]) * (x[1] - %e) / %e' %
            (entrance_velocity, y1, y0, (0.5 * (y1 - y0))**2),
            degree=2
            )
    u_bcs = [
        DirichletBC(W, (0.0, 0.0), upper_boundary),
        DirichletBC(W, (0.0, 0.0), lower_boundary),
        DirichletBC(W, (0.0, 0.0), obstacle_boundary),
        DirichletBC(W.sub(0), inflow, left_boundary),
        #
        DirichletBC(W.sub(0), outflow, right_boundary),
        ]
    # dudt_bcs = [
    #     DirichletBC(W, (0.0, 0.0), upper_boundary),
    #     DirichletBC(W, (0.0, 0.0), lower_boundary),
    #     DirichletBC(W, (0.0, 0.0), obstacle_boundary),
    #     DirichletBC(W.sub(0), 0.0, left_boundary),
    #     # DirichletBC(W.sub(1), 0.0, right_boundary),
    #     ]

    # If there is a penetration boundary (i.e., n.u!=0), then the pressure must
    # be set somewhere to make sure that the Navier-Stokes problem remains
    # consistent.
    # When solving Stokes with no Dirichlet conditions whatsoever, the pressure
    # tends to 0 at the outlet. This is natural since there, the liquid can
    # flow out at the rate it needs to be under no pressure at all.
    # Hence, at outlets, set the pressure to 0.
    p_bcs = [
        # DirichletBC(P, 0.0, right_boundary)
        ]

    # Getting vortices is not easy. If we take the actual viscosity of water,
    # they don't appear.
    mu = 0.002
    # mu = materials.water.dynamic_viscosity(T=293.0)

    # For starting off, solve the Stokes equation.
    u0, p0 = flow.stokes.solve(
        WP,
        u_bcs + p_bcs,
        mu,
        f=Constant((0.0, 0.0)),
        verbose=False,
        tol=1.0e-13,
        max_iter=10000
        )
    u0.rename('velocity', 'velocity')
    p0.rename('pressure', 'pressure')

    rho = materials.water.density(T=293.0)
    # stepper = flow.navier_stokes.IPCS(theta=1.0)
    # stepper = flow.navier_stokes.Chorin()
    stepper = flow.navier_stokes.Rotational()

    W2 = u0.function_space()
    P2 = p0.function_space()
    u_bcs = [
        DirichletBC(W2, (0.0, 0.0), upper_boundary),
        DirichletBC(W2, (0.0, 0.0), lower_boundary),
        DirichletBC(W2, (0.0, 0.0), obstacle_boundary),
        DirichletBC(W2.sub(0), inflow, left_boundary),
        DirichletBC(W2.sub(0), outflow, right_boundary),
        ]
    # TODO settting the outflow _and_ the pressure at the outlet is actually
    #      not necessary. Even without the pressure Dirichlet conditions, the
    #      pressure correction system should be consistent.
    p_bcs = [
        DirichletBC(P2, 0.0, right_boundary)
        ]

    # Report Reynolds number.
    # https://en.wikipedia.org/wiki/Reynolds_number#Sphere_in_a_fluid
    reynolds = entrance_velocity * obstacle_diameter * rho / mu
    print('Reynolds number:  %e' % reynolds)

    dt = 1.0e-5
    dt_max = 1.0
    t = 0.0

    with XDMFFile(mpi_comm_world(), 'karman.xdmf') as xdmf_file:
        xdmf_file.parameters['flush_output'] = True
        xdmf_file.parameters['rewrite_function_mesh'] = False

        k = 0
        while k < num_steps:
            k += 1
            print
            print('t = %f' % t)
            if show:
                plot(u0)
                plot(p0)
                xdmf_file.write(u0, t)
                xdmf_file.write(p0, t)

            u1, p1 = stepper.step(
                    dt,
                    {0: u0}, p0,
                    u_bcs, p_bcs,
                    rho, mu,
                    f={
                        0: Constant((0.0, 0.0)),
                        1: Constant((0.0, 0.0))
                    },
                    verbose=False,
                    tol=1.0e-10
                    )
            u0.assign(u1)
            p0.assign(p1)

            # Adaptive stepsize control based solely on the velocity field.
            # CFL-like condition for time step. This should be some sort of
            # average of the temperature in the current step and the target
            # step.
            #
            # More on step-size control for Navier--Stokes:
            #
            #     Adaptive time step control for the incompressible
            #     Navier-Stokes equations;
            #     Volker John, Joachim Rang;
            #     Comput. Methods Appl. Mech. Engrg. 199 (2010) 514-524;
            #     <http://www.wias-berlin.de/people/john/ELECTRONIC_PAPERS/JR10.CMAME.pdf>.
            #
            # Section 3.3 in that paper notes that time-adaptivity for theta-
            # schemes is too costly. They rather reside to DIRK- and
            # Rosenbrock-methods.
            #
            begin('Step size adaptation...')
            ux, uy = u0.split()
            unorm = project(
                    sqrt(ux**2 + uy**2),
                    FunctionSpace(mesh, 'Lagrange', 2),
                    form_compiler_parameters={'quadrature_degree': 4}
                    )
            unorm = norm(unorm.vector(), 'linf')

            # print('||u||_inf = %e' % unorm)
            # Some smooth step-size adaption.
            target_dt = 1.0 * mesh.hmax() / unorm
            print('current dt: %e' % dt)
            print('target dt:  %e' % target_dt)
            # alpha is the aggressiveness factor. The distance between the
            # current step size and the target step size is reduced by
            # |1-alpha|. Hence, if alpha==1 then dt_next==target_dt. Otherwise
            # target_dt is approached more slowly.
            alpha = 0.5
            dt = min(
                dt_max,
                # At most double the step size from step to step.
                dt * min(2.0, 1.0 + alpha*(target_dt - dt)/dt)
                )
            print('next dt:    %e' % dt)
            t += dt
            end()

    return


if __name__ == '__main__':
    test_karman(lcar=5.0e-3, num_steps=100000, show=True)
