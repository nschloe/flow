#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from dolfin import (
    dx, assemble, dot, grad, TestFunction, TrialFunction, Function, LUSolver
    )


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

    def _get_system(self, alpha, beta, u, v):
        # If there are sharp temperature gradients, numerical oscillations may
        # occur. This happens because the resulting matrix is not an M-matrix,
        # caused by the fact that A1 puts positive elements in places other
        # than the main diagonal. To prevent that, it is suggested by
        # Großmann/Roos to use a vertex-centered discretization for the mass
        # matrix part.
        # Check
        # https://bitbucket.org/fenics-project/ffc/issues/145/uflacs-error-for-vertex-quadrature-scheme
        f1 = assemble(
              u * v * dx,
              form_compiler_parameters={
                  'quadrature_rule': 'vertex',
                  'representation': 'quadrature'
                  }
              )
        # f1 = assemble(
        #     u * v * dx,
        #     form_compiler_parameters={
        #         'quadrature_rule': 'vertex',
        #         'quadrature_degree': 1
        #         }
        #     )
        f2 = assemble(
            - dot(self.conv, grad(u)) * v * dx
            - self.kappa * dot(grad(u), grad(v/(self.rho*self.cp))) * dx
            )
        return alpha * f1 + beta * f2

    # pylint: disable=unused-argument
    def eval_alpha_M_beta_F(self, alpha, beta, u, t):
        # Evaluate  alpha * M * u + beta * F(u, t).
        v = TestFunction(self.V)
        return self._get_system(alpha, beta, u, v)

    def solve_alpha_M_beta_F(self, alpha, beta, b, t):
        # Solve  alpha * M * u + beta * F(u, t) = b  for u.
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        A = self._get_system(alpha, beta, u, v)

        for bc in self.bcs:
            bc.apply(A, b)

        # solver = KrylovSolver('gmres', 'ilu')
        # solver.parameters['relative_tolerance'] = 1.0e-13
        # solver.parameters['absolute_tolerance'] = 0.0
        # solver.parameters['maximum_iterations'] = 1000
        # solver.parameters['monitor_convergence'] = True

        # The Krylov solver doesn't converge
        solver = LUSolver()
        solver.set_operator(A)

        u = Function(self.V)
        solver.solve(u.vector(), b)
        return u
