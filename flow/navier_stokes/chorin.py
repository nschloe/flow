# -*- coding: utf-8 -*-
#
'''
Numerical solution schemes for the Navier--Stokes equation

        rho (u' + u.nabla(u)) = - nabla(p) + mu Delta(u) + f,
        div(u) = 0.

For an overview of methods, see

    An overview of projection methods for incompressible flows;
    Guermond, Minev, Shen;
    Comput. Methods Appl. Mech. Engrg., 195 (2006);
    <http://www.math.ust.hk/~mawang/teaching/math532/guermond-shen-2006.pdf>

or

    <http://mumerik.iwr.uni-heidelberg.de/Oberwolfach-Seminar/CFD-Course.pdf>.
'''

from ..message import Message

from dolfin import (
    dot, inner, grad, dx, div, Function, TestFunction, solve, Constant,
    derivative, TrialFunction, assemble, PETScPreconditioner,
    PETScKrylovSolver, as_backend_type, PETScOptions
    )


def _rhs_weak(u, v, f, rho, mu):
    '''Right-hand side of the Navier--Stokes momentum equation in weak form.
    '''
    return (
        inner(f, v) * dx
        - mu * inner(grad(u), grad(v)) * dx
        - rho * 0.5 * (inner(grad(u)*u, v) - inner(grad(v)*u, u)) * dx
        # - rho*inner(grad(u)*u, v) * dx
        )


class Chorin(object):
    '''Chorin's method as described in section 3.1 of

        An overview of projection methods for incompressible flows;
        Guermond, Miev, Shen;
        Comput. Methods Appl. Mech. Engrg. 195 (2006),
        <http://www.math.tamu.edu/~guermond/PUBLICATIONS/guermond_minev_shen_CMAME_2006.pdf>.
    '''
    order = {
        'velocity': 1.0,
        'pressure': 0.5,
        }

    def __init__(self, rho, mu):
        assert mu > 0.0
        # Only works for linear elements.
        if isinstance(rho, float):
            assert rho > 0.0
        else:
            try:
                assert rho.vector().min() > 0.0
            except AttributeError:
                # AttributeError: 'Sum' object has no attribute 'vector'
                pass

        self.rho = rho
        self.mu = mu
        return

    def step(self,
             dt,
             u0, p0,
             u_bcs, p_bcs,
             f1,
             verbose=True,
             tol=1.0e-10
             ):
        # Some initial sanity checkups.
        assert dt > 0.0
        # Define trial and test functions
        v = TestFunction(u0.function_space())
        # Create functions
        # Define coefficients
        k = Constant(dt)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Compute tentative velocity step:
        #
        #   F(u) = 0,
        #   F(u) := rho ((Ui-U0)/dt + (u.\nabla)u) - mu \div(\nabla u) - f.
        #
        with Message('Computing tentative velocity'):
            ui = Function(u0.function_space())

            F1 = (
                + self.rho * inner((ui - u0)/k, v) * dx
                - _rhs_weak(ui, v, f1, self.rho, self.mu)
                )

            # Get linearization and solve nonlinear system.
            # If the scheme is fully explicit (theta=0.0), then the system is
            # actually linear and only one Newton iteration is performed.
            J = derivative(F1, ui)

            # What is a good initial guess for the Newton solve?
            # Three choices come to mind:
            #
            #    (1) the previous solution u0,
            #    (2) the intermediate solution from the previous step ui0,
            #    (3) the solution of the semilinear system
            #        (u.\nabla(u) -> u0.\nabla(u)).
            #
            # Numerical experiments with the Karman vortex street show that the
            # order of accuracy is (1), (3), (2). Typical norms would look like
            #
            #     ||u - u0 || = 1.726432e-02
            #     ||u - ui0|| = 2.720805e+00
            #     ||u - u_e|| = 5.921522e-02
            #
            # Hence, use u0 as initial guess.
            ui.assign(u0)

            # problem = NonlinearVariationalProblem(F1, ui, u_bcs, J)
            # solver = NonlinearVariationalSolver(problem)
            solve(
                F1 == 0, ui,
                bcs=u_bcs,
                J=J,
                solver_parameters={
                    # 'nonlinear_solver': 'snes',
                    'nonlinear_solver': 'newton',
                    'newton_solver': {
                        'maximum_iterations': 10,
                        'report': True,
                        'absolute_tolerance': 1.0e-9,
                        'relative_tolerance': 0.0,
                        }
                   }
                )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Message('Computing pressure correction'):
            #
            # The following is based on the update formula
            #
            #     rho/dt (u_{n+1}-u*) + \nabla p = 0
            #
            # With div(u_{n+1})=0 one derives
            #
            #   - \nabla^2 p = rho/dt div(u_{n+1} - u*),
            #   - n.\nabla p = rho/dt  n.(u_{n+1} - u*),
            #
            # In its weak form, this is
            #
            #     \int \grad(p).\grad(q)
            #   = - rho/dt \int div(u*) q - rho/dt \int_Gamma n.(u_{n+1}-u*) q.
            #
            # If Dirichlet boundary conditions are applied to both u* and
            # u_{n+1} (the latter in the final step), the boundary integral
            # vanishes.
            #
            # Assume that on the boundary
            #   L2 -= inner(n, rho/k (u_bcs - ui)) * q * ds
            # is zero. This requires the boundary conditions to be set for
            # ui as well as u_final.
            # This creates some problems if the boundary conditions are
            # supposed to remain 'free' for the velocity, i.e., no Dirichlet
            # conditions in normal direction. In that case, one needs to
            # specify Dirichlet pressure conditions.
            #
            p1 = self._pressure_poisson(
                    p0,
                    self.mu, ui,
                    divu=self.rho / Constant(dt) * div(ui),
                    p_bcs=p_bcs,
                    tol=tol,
                    verbose=verbose
                    )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Velocity correction.
        #   U = U* - dt/rho \nabla p.
        with Message('Computing velocity correction'):
            # grad(p) is discontinuous, so we can't just do
            #
            #   u = ui - dt/rho * grad(p)
            #
            # Also, this doesn't respect boundary conditions on u. Rather treat
            # this update as an FEM equation system as well.
            u2 = TrialFunction(u0.function_space())
            a3 = inner(u2, v) * dx
            L3 = inner(ui - k/self.rho * grad(p1), v) * dx
            solve(a3 == L3, u0,
                  bcs=u_bcs,
                  solver_parameters={
                      'linear_solver': 'iterative',
                      'symmetric': True,
                      # 'preconditioner': 'amg',
                      # We'd love to use AMG here, but
                      # <https://bitbucket.org/fenics-project/docker/issues/61/petsc-vectorfunctionspace-amg-malloc>
                      # prevents it.
                      'preconditioner': 'ilu',
                      'krylov_solver': {
                          'relative_tolerance': tol,
                          'absolute_tolerance': 0.0,
                          'maximum_iterations': 100,
                          'monitor_convergence': verbose
                          }
                      })
        p0.assign(p1)
        return u0, p0

    def _pressure_poisson(
            self,
            p0,
            mu, ui,
            divu,
            p_bcs=None,
            tol=1.0e-10,
            verbose=True
            ):
        '''Solve the pressure Poisson equation

            - \Delta p = -div(u),
            boundary conditions.
        '''
        P = p0.function_space()

        p1 = Function(P)
        p = TrialFunction(P)
        q = TestFunction(P)

        a2 = dot(grad(p), grad(q)) * dx
        L2 = -divu * q * dx

        if p_bcs:
            solve(a2 == L2, p1,
                  bcs=p_bcs,
                  solver_parameters={
                      'linear_solver': 'iterative',
                      'symmetric': True,
                      'preconditioner': 'hypre_amg',
                      'krylov_solver': {
                          'relative_tolerance': tol,
                          'absolute_tolerance': 0.0,
                          'maximum_iterations': 100,
                          'monitor_convergence': verbose
                          }
                  })
        else:
            # If we're dealing with a pure Neumann problem here (which is the
            # default case), this doesn't hurt CG if the system is consistent,
            # cf.
            #
            #    Iterative Krylov methods for large linear systems,
            #    Henk A. van der Vorst.
            #
            # And indeed, it is consistent: Note that
            #
            #    <1, rhs> = \sum_i 1 * \int div(u) v_i
            #             = 1 * \int div(u) \sum_i v_i
            #             = \int div(u).
            #
            # With the divergence theorem, we have
            #
            #    \int div(u) = \int_\Gamma n.u.
            #
            # The latter term is 0 if and only if inflow and outflow are
            # exactly the same at any given point in time. This corresponds
            # with the incompressibility of the liquid.
            #
            # Note that this hints towards penetrable boundaries to require
            # Dirichlet conditions on the pressure.
            #
            A = assemble(a2)
            b = assemble(L2)
            #
            # In principle, the ILU preconditioner isn't advised here since it
            # might destroy the semidefiniteness needed for CG.
            #
            # The system is consistent, but the matrix has an eigenvalue 0.
            # This does not harm the convergence of CG, but with
            # preconditioning one has to make sure that the preconditioner
            # preserves the kernel. ILU might destroy this (and the
            # semidefiniteness). With AMG, the coarse grid solves cannot be LU
            # then, so try Jacobi here.
            # <http://lists.mcs.anl.gov/pipermail/petsc-users/2012-February/012139.html>
            #
            prec = PETScPreconditioner('hypre_amg')
            PETScOptions.set(
                'pc_hypre_boomeramg_relax_type_coarse',
                'jacobi'
                )
            solver = PETScKrylovSolver('cg', prec)
            solver.parameters['absolute_tolerance'] = 0.0
            solver.parameters['relative_tolerance'] = tol
            solver.parameters['maximum_iterations'] = 100
            solver.parameters['monitor_convergence'] = verbose
            # Create solver and solve system
            A_petsc = as_backend_type(A)
            b_petsc = as_backend_type(b)
            p1_petsc = as_backend_type(p1.vector())
            solver.set_operator(A_petsc)

            solver.solve(p1_petsc, b_petsc)
        return p1
