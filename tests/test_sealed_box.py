# -*- coding: utf-8 -*-
#
import flow

from dolfin import (
        Mesh, FunctionSpace, DirichletBC, VectorElement, FiniteElement,
        Constant, plot, XDMFFile, project, SpatialCoordinate, sqrt, norm,
        mpi_comm_world, interactive
        )
import materials
import meshio
import pygmsh
import sys


# def create_mesh(lcar):
#     geom = pygmsh.Geometry()
#
#     geom.add_rectangle(0.0, 0.1, 0.0, 0.1, 0.0, lcar)
#
#     points, cells, point_data, cell_data, field_data = \
#         pygmsh.generate_mesh(geom)
#
#     # https://fenicsproject.org/qa/12891/initialize-mesh-from-vertices-connectivities-at-once
#     meshio.write('test.xml', points, cells)
#     return Mesh('test.xml')


def create_mesh(lcar):
    geom = pygmsh.Geometry()

    x0 = 0.0
    x1 = 0.1
    y0 = 0.0
    y1 = 0.2

    circle = geom.add_circle([0.05, 0.05, 0.0], 0.02, lcar, make_surface=False)

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

    g = -9.81

    # When using Stokes for bootstrapping, the velocity errors are introduced
    # in the first step that IPCS/Rotational will never get rid of.
    # u0, p0 = flow.stokes.solve(
    #     WP,
    #     u_bcs + p_bcs,
    #     mu,
    #     f=Constant((0.0, 0.0)),
    #     verbose=False,
    #     tol=1.0e-13,
    #     max_iter=10000
    #     )
    y = SpatialCoordinate(mesh)[1]
    u0 = project(Constant([0, 0]), FunctionSpace(mesh, W_element))
    u0.rename('velocity', 'velocity')
    p0 = project(g * y, FunctionSpace(mesh, P_element))
    p0.rename('pressure', 'pressure')

    rho = materials.water.density(T=293.0)
    # stepper = flow.navier_stokes.Chorin()
    stepper = flow.navier_stokes.IPCS()
    # stepper = flow.navier_stokes.Rotational()

    W2 = u0.function_space()
    u_bcs = [DirichletBC(W2, (0.0, 0.0), 'on_boundary')]
    p_bcs = []

    dt = 1.0e-2
    t = 0.0

    with XDMFFile(mpi_comm_world(), 'sealed_box.xdmf') as xdmf_file:
        xdmf_file.parameters['flush_output'] = True
        xdmf_file.parameters['rewrite_function_mesh'] = False

        k = 0
        while k < num_steps:
            k += 1
            print
            print('t = %f' % t)
            if show:
                xdmf_file.write(u0, t)
                xdmf_file.write(p0, t)
                plot(u0)
                plot(p0)
                interactive()

            u1, p1 = stepper.step(
                    dt,
                    {0: u0}, p0,
                    u_bcs, p_bcs,
                    rho, mu,
                    f={
                        0: Constant((0.0, g)),
                        1: Constant((0.0, g))
                    },
                    verbose=False,
                    tol=1.0e-10
                    )
            u0.assign(u1)
            p0.assign(p1)
            t += dt

        ux, uy = u0.split()
        unorm = project(
                sqrt(ux**2 + uy**2),
                FunctionSpace(mesh, 'Lagrange', 2),
                form_compiler_parameters={'quadrature_degree': 4}
                )
        unorm = norm(unorm.vector(), 'linf')
        assert unorm < 1.0e-15

    return


if __name__ == '__main__':
    test_sealed_box(lcar=5.0e-3, num_steps=sys.maxint, show=True)
