# -*- coding: utf-8 -*-
#
import flow

from dolfin import (
        Mesh, FunctionSpace, DirichletBC, VectorElement, FiniteElement,
        Constant, plot, XDMFFile
        )
import materials
import meshio
import pygmsh
import sys


def create_mesh(lcar):
    geom = pygmsh.Geometry()

    geom.add_rectangle(0.0, 0.1, 0.0, 0.1, 0.0, lcar)

    points, cells, point_data, cell_data, field_data = \
        pygmsh.generate_mesh(geom)

    # https://fenicsproject.org/qa/12891/initialize-mesh-from-vertices-connectivities-at-once
    meshio.write('test.xml', points, cells)
    return Mesh('test.xml')


def test_sealed_box(num_steps=2, lcar=0.1, show=False):
    mesh = create_mesh(lcar)

    W_element = VectorElement('Lagrange', mesh.ufl_cell(), 2)
    P_element = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    WP = FunctionSpace(mesh, W_element * P_element)

    W = WP.sub(0)
    # P = WP.sub(1)

    u_bcs = [DirichletBC(W, (0.0, 0.0), 'on_boundary')]
    p_bcs = []

    mu = materials.water.dynamic_viscosity(T=293.0)

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

    rho = materials.water.density(T=293.0)
    # stepper = flow.navier_stokes.Chorin()
    stepper = flow.navier_stokes.IPCS()
    # stepper = flow.navier_stokes.Rotational()

    W2 = u0.function_space()
    u_bcs = [DirichletBC(W2, (0.0, 0.0), 'on_boundary')]
    p_bcs = []

    if show:
        u_file = XDMFFile('velocity.xdmf')
        u_file.parameters['flush_output'] = True
        u_file.parameters['rewrite_function_mesh'] = False

    dt = 1.0e-2
    t = 0.0

    k = 0
    while k < num_steps:
        k += 1
        print
        print('t = %f' % t)
        if show:
            plot(u0)
            plot(p0)
            u_file.write(u0, t)

        u1, p1 = stepper.step(
                dt,
                {0: u0}, p0,
                u_bcs, p_bcs,
                rho, mu,
                f={
                    0: Constant((0.0, -9.81)),
                    1: Constant((0.0, -9.81))
                },
                verbose=False,
                tol=1.0e-10
                )
        u0.assign(u1)
        p0.assign(p1)
        t += dt

    return


if __name__ == '__main__':
    test_sealed_box(lcar=5.0e-3, num_steps=sys.maxint, show=True)
