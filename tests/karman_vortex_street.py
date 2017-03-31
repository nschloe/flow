# -*- coding: utf-8 -*-
#
import flow

from dolfin import (
        Mesh, SubDomain, FunctionSpace, DOLFIN_EPS, Expression, DirichletBC,
        VectorElement, FiniteElement, Constant, plot
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


def test_karman(num_steps=5, show=False):
    points, cells, point_data, cell_data, field_data = create_mesh(lcar=0.1)
    # https://fenicsproject.org/qa/12891/initialize-mesh-from-vertices-connectivities-at-once
    meshio.write('test.xml', points, cells)
    mesh = Mesh('test.xml')

    W_element = VectorElement('Lagrange', mesh.ufl_cell(), 2)
    P_element = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    WP = FunctionSpace(mesh, W_element * P_element)

    W = WP.sub(0)
    P = WP.sub(1)

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
                and x[0] > x0 + DOLFIN_EPS and x[0] < x1 - mesh_eps
                and x[1] > y0 + DOLFIN_EPS and x[1] < y1 - mesh_eps
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
    inflow = Expression('0.1 * (0.5 + x[1]) * (0.5 - x[1])', degree=2)
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
    # be set somewhere to make sure that the Navier-Stokes problem remains
    # consistent.
    # When solving Stokes with no Dirichlet conditions whatsoever, the pressure
    # tends to 0 at the outlet. This is natural since there, the liquid can
    # flow out at the rate it needs to be under no pressure at all.
    # Hence, at outlets, set the pressure to 0.
    p_bcs = [DirichletBC(P, 0.0, right_boundary)]

    mu = materials.water.dynamic_viscosity(T=293.0)
    # For starting off, solve the Stokes equation.
    u0, p0 = flow.stokes.solve(
        WP,
        u_bcs + p_bcs,
        mu,
        f=Constant((0.0, 0.0)),
        verbose=False,
        tol=1.0e-13,
        max_iter=1000
        )

    W2 = u0.function_space()
    P2 = p0.function_space()

    rho = materials.water.density(T=293.0)
    stepper = flow.navier_stokes.IPCS(
            W2, P2,
            rho, mu,
            theta=1.0
            )

    u_bcs = [
        DirichletBC(W2, (0.0, 0.0), upper_boundary),
        DirichletBC(W2, (0.0, 0.0), lower_boundary),
        DirichletBC(W2, (0.0, 0.0), obstacle_boundary),
        DirichletBC(W2.sub(0), inflow, left_boundary),
        # DirichletBC(W2.sub(1), 0.0, right_boundary),
        ]
    p_bcs = [DirichletBC(P2, 0.0, right_boundary)]

    dt = 1.0e-2
    for k in range(num_steps):
        if show:
            plot(u0)
            plot(p0)

        u1, p1 = stepper.step(
                dt,
                u0, p0,
                u_bcs, p_bcs,
                f0=Constant((0.0, 0.0)),
                f1=Constant((0.0, 0.0)),
                verbose=True,
                tol=1.0e-10
                )
        u0.assign(u1)
        p0.assign(p1)

    return


if __name__ == '__main__':
    test_karman(num_steps=1000, show=True)
