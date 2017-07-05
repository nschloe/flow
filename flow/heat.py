#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from dolfin import (
    dx, assemble, dot, grad, TestFunction, TrialFunction, Function, LUSolver,
    div, assemble_system, lhs, rhs
    )

from . import stabilization


class Heat(object):
    '''
    Provides methods for computing

    ..math::

        u' = F(t, u).
    '''
    def __init__(
            self, V, conv, kappa, rho, cp, bcs, source,
            supg_stabilization=False
            ):
        self.V = V
        self.bcs = bcs

        u = TrialFunction(V)
        v = TestFunction(V)
        rho_cp = rho * cp

        # If there are sharp temperature gradients, numerical oscillations may
        # occur. This happens because the resulting matrix is not an M-matrix,
        # caused by the fact that A1 puts positive elements in places other
        # than the main diagonal. To prevent that, it is suggested by
        # Großmann/Roos to use a vertex-centered discretization for the mass
        # matrix part.
        # Check
        # https://bitbucket.org/fenics-project/ffc/issues/145/uflacs-error-for-vertex-quadrature-scheme
        self.M = assemble(
              u * v * dx,
              form_compiler_parameters={
                  'quadrature_rule': 'vertex',
                  'representation': 'quadrature'
                  }
              )
        # self.f1 = assemble(
        #     u * v * dx,
        #     form_compiler_parameters={
        #         'quadrature_rule': 'vertex',
        #         'quadrature_degree': 1
        #         }
        #     )

        f = (
            - kappa * dot(grad(u), grad(v / rho_cp)) * dx
            - dot(conv, grad(u)) * v * dx
            + source * v * dx
            )

        if supg_stabilization:
            # Add SUPG stabilization. It's of the form
            #
            #   R * tau * dot(convection, grad(v))
            #
            # where R is the residual
            #
            #   R = u_t - div(kappa grad(u)) + ...
            #
            # The u_t part of the expression is moved to the left-hand side of
            #
            #    u_t = f(t, u)
            #
            # as part of M.
            assert conv is not None
            mesh = v.function_space().mesh()
            element_degree = v.ufl_element().degree()
            tau = stabilization.supg(mesh, conv, kappa, element_degree)
            #
            self.M += assemble(u * tau * dot(conv, grad(v)) * dx)
            #
            R2 = (
                + div(kappa * grad(u)) / rho_cp
                - dot(conv, grad(u))
                + source / rho_cp
                )
            f += R2 * tau * dot(conv, grad(v)) * dx

        self.A, self.b = assemble_system(lhs(f), rhs(f))
        return

    # pylint: disable=unused-argument
    def eval_alpha_M_beta_F(self, alpha, beta, u, t):
        '''Evaluate  alpha * M * u + beta * F(u, t).
        '''
        uvec = u.vector()
        # Convert to proper `float`s to avoid accidental conversion to
        # numpy.arrays, cf.
        # <https://bitbucket.org/fenics-project/dolfin/issues/874/genericvector-numpyfloat-numpyarray-not>
        alpha = float(alpha)
        beta = float(beta)
        return alpha * (self.M * uvec) + beta * (self.A * uvec + self.b)

    def solve_alpha_M_beta_F(self, alpha, beta, b, t):
        '''Solve  alpha * M * u + beta * F(u, t) = b  for u.
        '''
        A = alpha * self.M + beta * self.A

        # See above for float conversion
        right_hand_side = - float(beta) * self.b.copy()
        if b:
            right_hand_side += b

        for bc in self.bcs:
            bc.apply(A, b)

        # The Krylov solver doesn't converge
        solver = LUSolver()
        solver.set_operator(A)

        u = Function(self.V)
        solver.solve(u.vector(), b)
        return u
