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
    DOLFIN_EPS, derivative, TrialFunction, FacetNormal, assemble, ds,
    PETScPreconditioner, PETScKrylovSolver, as_backend_type,
    PETScOptions
    )


def _rhs_weak(u, v, f, rho, mu):
    '''Right-hand side of the Navier--Stokes momentum equation in weak form.
    '''
    # Do no include the boundary term
    #
    #   - mu *inner(grad(u2)*n, v) * ds.
    #
    # This effectively means that at all boundaries where no sufficient
    # Dirichlet-conditions are posed, we assume grad(u)*n to vanish.
    #
    # It was first proposed in (with two intermediate steps)
    #
    #     Sur l'approximation de la solution des 'equations de Navier-Stokes
    #     par la m'ethode des pas fractionnaires (II);
    #     R. Temam;
    #     Arch. Ration. Mech. Anal. 33, (1969) 377-385;
    #     <http://link.springer.com/article/10.1007%2FBF00247696>.
    #
    # to replace the (weak form) convection <(u.\nabla)v, w> by something more
    # appropriate. Note, e.g., that
    #
    #       1/2 (  <(u.\nabla)v, w> - <(u.\nabla)w, v>)
    #     = 1/2 (2 <(u.\nabla)v, w> - <u, \nabla(v.w)>)
    #     = <(u.\nabla)v, w> - 1/2 \int u.\nabla(v.w)
    #     = <(u.\nabla)v, w> - 1/2 (-\int div(u)*(v.w)
    #                               +\int_\Gamma (n.u)*(v.w)
    #                              ).
    #
    # Since for solutions we have div(u)=0, n.u=0, we can consistently replace
    # the convection term <(u.\nabla)u, w> by the skew-symmetric
    #
    #     1/2 (<(u.\nabla)u, w> - <(u.\nabla)w, u>).
    #
    # One distinct advantage of this formulation is that the convective term
    # doesn't contribute to the total energy of the system since
    #
    # d/dt ||u||^2 = 2<d_t u, u>  = <(u.\nabla)u, u> - <(u.\nabla)u, u> = 0.
    #
    # More references and info on skew-symmetry can be found in
    #
    #     <http://www.wias-berlin.de/people/john/lectures_madrid_2012.pdf>,
    #     <http://calcul.math.cnrs.fr/Documents/Ecoles/CEMRACS2012/Julius_Reiss.pdf>.
    #
    # The first lecture is quite instructive and gives info on other
    # possibilities, e.g.,
    #
    #   * Rotational form
    #     <http://www.igpm.rwth-aachen.de/Download/reports/DROPS/IGPM193.pdf>
    #   * Divergence form
    #     This paper
    #     <http://www.cimec.org.ar/ojs/index.php/mc/article/viewFile/486/464>
    #     mentions 'divergence form', but it seems to be understood as another
    #     way of expressing the stress term mu\Delta(u).
    #
    # The different methods are numerically compared in
    #
    #     On the accuracy of the rotation form in simulations of the
    #     Navier-Stokes equations;
    #     Layton et al.;
    #     <http://www.mathcs.emory.edu/~molshan/ftp/pub/RotationForm.pdf>.
    #
    # In
    #
    #     Finite element methods
    #     for the incompressible Navier-Stokes equations;
    #     Ir. A. Segal;
    #     <http://ta.twi.tudelft.nl/users/vuik/burgers/fem_notes.pdf>;
    #
    # it is advised to use (u{k}.\nabla)u^{k+1} for the treatment of the
    # nonlinear term. In connection with the the div-stabilitation, this yields
    # unconditional stability of the scheme. On the other hand, an advantage
    # of treating the nonlinear term purely explicitly is that the resulting
    # problem would be symmetric and positive definite, qualifying for robust
    # AMG preconditioning.
    # One can also find advice on the boundary conditions for axisymmetric flow
    # here.
    #
    # For more information on stabilization techniques and general solution
    # recipes, check out
    #
    #     Finite Element Methods for Flow Problems;
    #     Jean Donea, Antonio Huerta.
    #
    # There are plenty of references in the book, e.g. to
    #
    #     Finite element stabilization parameters
    #     computed from element matrices and vectors;
    #     Tezduyar, Osawa;
    #     Comput. Methods Appl. Mech. Engrg. 190 (2000) 411-430;
    #     <http://www.tafsm.org/PUB_PRE/jALL/j89-CMAME-EBTau.pdf>
    #
    # where more details on SUPG are given.
    #
    return (
        inner(f, v) * dx
        - mu * inner(grad(u), grad(v)) * dx
        - rho * 0.5 * (inner(grad(u)*u, v) - inner(grad(v)*u, u)) * dx
        # - rho*inner(grad(u)*u, v) * dx
        )


class PressureProjection(object):
    '''General pressure projection scheme as described in section 3.4 of

        An overview of projection methods for incompressible flows;
        Guermond, Miev, Shen;
        Comput. Methods Appl. Mech. Engrg. 195 (2006).
    '''
    def __init__(self, rho, mu, theta):
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

        self.theta = theta
        self.rho = rho
        self.mu = mu
        return

    def step(self,
             dt,
             u0, p0,
             u_bcs, p_bcs,
             f0=None, f1=None,
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
        #     F(u) = 0,
        #     F(u) := rho (U0 + (u.\nabla)u) - mu \div(\nabla u) - f = 0.
        #
        with Message('Computing tentative velocity'):
            # TODO higher-order scheme for time integration
            #
            # For higher-order schemes, see
            #
            #     A comparison of time-discretization/linearization approaches
            #     for the incompressible Navier-Stokes equations;
            #     Volker John, Gunar Matthies, Joachim Rang;
            #     Comput. Methods Appl. Mech. Engrg. 195 (2006) 5995-6010;
            #     <http://www.wias-berlin.de/people/john/ELECTRONIC_PAPERS/JMR06.CMAME.pdf>.
            #

            ui = Function(u0.function_space())

            F1 = self.rho * inner((ui - u0)/k, v) * dx

            if abs(self.theta) > DOLFIN_EPS:
                # Implicit terms.
                # Implicit schemes need f at target step (f1).
                assert f1 is not None
                F1 -= self.theta * _rhs_weak(ui, v, f1, self.rho, self.mu)
            if abs(1.0 - self.theta) > DOLFIN_EPS:
                # Explicit terms.
                # Explicit schemes need f at current step (f0).
                assert f0 is not None
                F1 -= (1.0 - self.theta) \
                    * _rhs_weak(u0, v, f0, self.rho, self.mu)

            if p0:
                F1 += inner(grad(p0), v) * dx

            # if stabilization:
            #     tau = stab.supg2(V.mesh(),
            #                      u_1,
            #                      mu/rho,
            #                      V.ufl_element().degree()
            #                      )
            #     R = rho*(ui - u_1)/k
            #     if abs(theta) > DOLFIN_EPS:
            #         R -= theta * _rhs_strong(ui, f1, rho, mu)
            #     if abs(1.0-theta) > DOLFIN_EPS:
            #         R -= (1.0-theta) * _rhs_strong(u_1, f0, rho, mu)
            #     if p_1:
            #         R += grad(p_1)
            #     # TODO use u_1 or ui here?
            #     F1 += tau * dot(R, grad(v)*u_1) * dx

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

            # Wrap the solution in a try-catch block to make sure we call end()
            # if necessary.
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
                        # 'linear_solver': 'iterative',
                        # # # The nonlinear term makes the problem generally
                        # # # nonsymmetric.
                        # # 'symmetric': False,
                        # #  If the nonsymmetry is too strong, e.g., if u_1 is
                        # #  large, then AMG preconditioning might not work
                        # #  very well.
                        # 'preconditioner': 'ilu',
                        # # 'preconditioner': 'hypre_amg',
                        # 'krylov_solver': {
                        #     'relative_tolerance': tol,
                        #     'absolute_tolerance': 0.0,
                        #     'maximum_iterations': 1000,
                        #     'monitor_convergence': verbose
                        #     }
                        }
                   }
                )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Message('Computing pressure correction'):
            #
            # The following is based on the update formula
            #
            #     rho/dt (u_{n+1}-u*) + \nabla phi = 0
            #
            # with
            #
            #     phi = (p_{n+1} - p*) + chi*mu*div(u*)
            #
            # and div(u_{n+1})=0. One derives
            #
            #   - \nabla^2 phi = rho/dt div(u_{n+1} - u*),
            #   - n.\nabla phi = rho/dt  n.(u_{n+1} - u*),
            #
            # In its weak form, this is
            #
            #     \int \grad(phi).\grad(q)
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
            rotational_form = False
            p1 = self._pressure_poisson(
                    p0,
                    self.mu, ui,
                    divu=self.rho / Constant(dt) * div(ui),
                    p_bcs=p_bcs,
                    rotational_form=rotational_form,
                    tol=tol,
                    verbose=verbose
                    )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Velocity correction.
        #   U = U0 - dt/rho \nabla p.
        with Message('Computing velocity correction'):
            u2 = TrialFunction(u0.function_space())
            a3 = inner(u2, v) * dx
            if p0:
                phi = Function(p0.function_space())
                phi.assign(p1)
                phi -= p0
            else:
                phi = p1
            if rotational_form:
                phi += self.mu * div(ui)
            L3 = inner(ui,  v) * dx \
                - k/self.rho * inner(grad(phi), v) * dx
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
        # u = project(ui - k/rho * grad(phi), V)
        # print '||u||_div = %e' % norm(u, 'Hdiv0')
        # uu = TrialFunction(Q)
        # vv = TestFunction(Q)
        # div_u = Function(Q)
        # solve(uu*vv*dx == div(u)*vv*dx, div_u,
        #       #bcs=DirichletBC(Q, 0.0, 'on_boundary')
        #       )
        # plot(div_u, title='div(u)')
        # interactive()
        p0.assign(p1)
        return u0, p0

    def _pressure_poisson(
            self,
            p0,
            mu, ui,
            divu,
            p_bcs=None,
            p_n=None,
            rotational_form=False,
            tol=1.0e-10,
            verbose=True
            ):
        '''Solve the pressure Poisson equation

            - \Delta phi = -div(u),
            boundary conditions,

        for

            \nabla p = u.
        '''
        P = p0.function_space()

        p1 = Function(P)
        p = TrialFunction(P)
        q = TestFunction(P)

        a2 = dot(grad(p), grad(q)) * dx
        L2 = -divu * q * dx

        if p0:
            L2 += dot(grad(p0), grad(q)) * dx
        if p_n:
            n = FacetNormal(P.mesh())
            L2 += dot(n, p_n) * q * ds

        if rotational_form:
            L2 -= mu * dot(grad(div(ui)), grad(q)) * dx

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

            # TODO clear everything
            # <https://fenicsproject.org/qa/12916/clear-petscoptions>
            # PETScOptions.clear('ksp_view')
            # PETScOptions.clear('ksp_monitor_true_residual')
            # PETScOptions.clear('pc_type')
            # PETScOptions.clear('pc_fieldsplit_type')
            # PETScOptions.clear('pc_fieldsplit_detect_saddle_point')
            # PETScOptions.clear('fieldsplit_0_ksp_type')
            # PETScOptions.clear('fieldsplit_0_pc_type')
            # PETScOptions.clear('fieldsplit_1_ksp_type')
            # PETScOptions.clear('fieldsplit_1_pc_type')

            # p = TrialFunction(P)
            # q = TestFunction(P)
            # a2 = dot(grad(p), grad(q)) * dx
            # A = assemble(a2)
            # from dolfin import project
            # b = assemble(project(Constant(0.0), P) * q * dx)
            # p1 = assemble(project(Constant(0.0), P) * q * dx)

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

            # # TODO necessary?
            # # Check if the system is indeed consistent.
            # #
            # # If the right hand side is flawed (e.g., by round-off errors),
            # # then it may have a component b1 in the direction of the null
            # # space, orthogonal to the image of the operator:
            # #
            # #     b = b0 + b1.
            # #
            # # When starting with initial guess x0=0, the minimal achievable
            # # relative tolerance is then
            # #
            # #    min_rel_tol = ||b1|| / ||b||.
            # #
            # # If ||b|| is very small, which is the case when ui is almost
            # # divergence-free, then min_rel_to may be larger than the
            # # prescribed relative tolerance tol.
            # #
            # # Use this as a consistency check, i.e., bail out if
            # #
            # #     tol < min_rel_tol = ||b1|| / ||b||.
            # #
            # # For computing ||b1||, we use the fact that the null space is
            # # one-dimensional, i.e.,  b1 = alpha e,  and
            # #
            # #     e.b = e.(b0 + b1) = e.b1 = alpha ||e||^2,
            # #
            # # so  alpha = e.b/||e||^2  and
            # #
            # #     ||b1|| = |alpha| ||e|| = e.b / ||e||
            # #
            # e = Function(P)
            # e.interpolate(Constant(1.0))
            # evec = e.vector()
            # evec /= norm(evec)
            # alpha = b.inner(evec)
            # normB = norm(b)
            # info('Linear system convergence failure.')
            # info(error.message)
            # message = (
            #     'Linear system not consistent! '
            #     '<b,e> = %g, ||b|| = %g, <b,e>/||b|| = %e, tol = %e.'
            #     ) \
            #     % (alpha, normB, alpha/normB, tol)
            # info(message)
            # if tol < abs(alpha) / normB:
            #     info('\int div(u)  =  %e' % assemble(divu * dx))
            #     # n = FacetNormal(Q.mesh())
            #     # info('\int_Gamma n.u = %e' % assemble(dot(n, u)*ds))
            #     # info('\int_Gamma u[0] = %e' % assemble(u[0]*ds))
            #     # info('\int_Gamma u[1] = %e' % assemble(u[1]*ds))
            #     # # Now plot the faulty u on a finer mesh (to resolve the
            #     # # quadratic trial functions).
            #     # fine_mesh = Q.mesh()
            #     # for k in range(1):
            #     #     fine_mesh = refine(fine_mesh)
            #     # V1 = FunctionSpace(fine_mesh, 'CG', 1)
            #     # W1 = V1*V1
            #     # uplot = project(u, W1)
            #     # #uplot = Function(W1)
            #     # #uplot.interpolate(u)
            #     # plot(uplot, title='u_tentative')
            #     # plot(uplot[0], title='u_tentative[0]')
            #     # plot(uplot[1], title='u_tentative[1]')
            #     plot(divu, title='div(u_tentative)')
            #     interactive()
            #     exit()
            #     raise RuntimeError(message)
            # else:
            #     exit()
            #     raise RuntimeError('Linear system failed to converge.')
        return p1


class IPCS(PressureProjection):
    '''
    Incremental pressure correction scheme.
    '''
    # See Guermond, Minev, Shen.
    order = {
        'velocity': 2,
        'pressure': 1,
        }

    def __init__(self, rho, mu, theta):
        super(IPCS, self).__init__(rho, mu, theta)
        return
