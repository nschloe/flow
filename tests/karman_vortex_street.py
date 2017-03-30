# -*- coding: utf-8 -*-
#
import flow

from dolfin import (
        Mesh, SubDomain, FunctionSpace, DOLFIN_EPS, Expression, DirichletBC,
        VectorElement, FiniteElement, Constant
        )
import materials
import meshio
import pygmsh

x0 = 0.0
x1 = 5.0
y0 = -0.5
y1 = 0.5


def create_mesh(lcar):
    geom = pygmsh.Geometry()

    circle = geom.add_circle([1.0, 0.0, 0.0], 0.2, lcar, make_surface=False)

    geom.add_rectangle(
        x0, x1, y0, y1,
        0.0,
        lcar,
        holes=[circle]
        )

    return pygmsh.generate_mesh(geom)


def _main():
    points, cells, point_data, cell_data, field_data = create_mesh(lcar=0.1)
    # https://fenicsproject.org/qa/12891/initialize-mesh-from-vertices-connectivities-at-once
    meshio.write('test.xml', points, cells)
    mesh = Mesh('test.xml')

    W_element = VectorElement('Lagrange', mesh.ufl_cell(), 2)
    P_element = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    WP = FunctionSpace(mesh, W_element * P_element)

    W = WP.sub(0)
    P = WP.sub(1)

    # Define mesh and boundaries.
    class LeftBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] < x0 + DOLFIN_EPS
    left_boundary = LeftBoundary()

    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] > x1 - DOLFIN_EPS

    class LowerBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] < y0 + DOLFIN_EPS
    lower_boundary = LowerBoundary()

    class UpperBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] > y1 - DOLFIN_EPS
    upper_boundary = UpperBoundary()

    class ObstacleBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return (
                on_boundary
                and x[0] > x0 + DOLFIN_EPS and x[0] < x1 - DOLFIN_EPS
                and x[1] > y0 + DOLFIN_EPS and x[1] < y1 - DOLFIN_EPS
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
    inflow = Expression('100 * (0.5 + x[1]) * (0.5 - x[1])', degree=2)
    u_bcs = [
        DirichletBC(W, (0.0, 0.0), upper_boundary),
        DirichletBC(W, (0.0, 0.0), lower_boundary),
        DirichletBC(W, (0.0, 0.0), obstacle_boundary),
        DirichletBC(W.sub(0), inflow, left_boundary),
        # DirichletBC(W.sub(1), 0.0, right_boundary),
        ]
    # dudt_bcs = [
    #     DirichletBC(W, (0.0, 0.0), upper_boundary),
    #     DirichletBC(W, (0.0, 0.0), lower_boundary),
    #     DirichletBC(W, (0.0, 0.0), obstacle_boundary),
    #     DirichletBC(W.sub(0), 0.0, left_boundary),
    #     # DirichletBC(W.sub(1), 0.0, right_boundary),
    #     ]
    # If there is a penetration boundary (i.e., n.u!=0), then the pressure must
    # be set at the boundary to make sure that the Navier-Stokes problem
    # remains consistent.
    # It is not quite clear now where exactly to set the pressure to 0. Inlet,
    # outlet, some other place? The PPE system is consistent in all cases.
    # TODO find out more about it
    # p_bcs = [DirichletBC(Q, 0.0, right_boundary)]
    p_bcs = []

    mu = materials.water.dynamic_viscosity(T=293.0)
    # For starting off, solve the Stokes equation.
    u0, p0 = flow.stokes.solve(
        WP,
        u_bcs + p_bcs,
        mu,
        f=Constant((0.0, 0.0)),
        verbose=True,
        tol=1.0e-13,
        max_iter=1000
        )

    from dolfin import plot, interactive
    plot(u0)
    plot(p0)
    interactive()

    # stepper = flow.navier_stokes.IPCS(
    #         W, P, rho, mu,
    #         theta=1.0,
    #         stabilization=False
    #         )

    return mesh, W, P, u_bcs, p_bcs


if __name__ == '__main__':
    _main()
