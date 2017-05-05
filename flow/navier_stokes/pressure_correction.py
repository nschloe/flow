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
# from ..stabilization import supg

from dolfin import (
    dot, inner, grad, dx, div, Function, TestFunction, solve, Constant,
    derivative, TrialFunction, assemble, PETScPreconditioner,
    PETScKrylovSolver, as_backend_type, PETScOptions
    )


def _rhs_strong(u, f, rho, mu):
    '''Right-hand side of the Navier--Stokes momentum equation in strong form.
    '''
    return f \
        - mu * div(grad(u)) \
        - rho * (grad(u)*u + 0.5*div(u)*u)


def _rhs_weak(u, v, f, rho, mu):
    '''Right-hand side of the Navier--Stokes momentum equation in weak form.
    '''
    # Do no include the boundary term
    #
    #   - mu * inner(grad(u)*n, v) * ds.
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
    #     Finite Element Methods for the Simulation of Incompressible Flows,
    #     Volker John,
    #     <http://www.wias-berlin.de/people/john/lectures_madrid_2012.pdf>,
    #
    # and
    #
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


def _pressure_poisson(
        p0,
        mu, ui,
        divu,
        p_bcs=None,
        p_n=None,
        p_function_space=None,
        rotational_form=False,
        tol=1.0e-10,
        verbose=True
        ):
    '''Solve the pressure Poisson equation

        - \Delta phi = -div(u),
        boundary conditions,

    for p with

        \\nabla p = u.
    '''
    if p0:
        P = p0.function_space()
    else:
        P = p_function_space

    p1 = Function(P)
    p = TrialFunction(P)
    q = TestFunction(P)

    a2 = dot(grad(p), grad(q)) * dx
    L2 = -divu * q * dx

    if p0:
        L2 += dot(grad(p0), grad(q)) * dx
    # if p_n:
    #     n = FacetNormal(P.mesh())
    #     L2 += dot(n, p_n) * q * ds

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
                      'monitor_convergence': verbose,
                      'error_on_nonconvergence': True
                      }
              })
    else:
        # If we're dealing with a pure Neumann problem here (which is the
        # default case), this doesn't hurt CG if the system is consistent, cf.
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
        # The latter term is 0 if and only if inflow and outflow are exactly
        # the same at any given point in time. This corresponds with the
        # incompressibility of the liquid.
        #
        # Another lesson from this:
        # If the mesh has penetration boundaries, you either have to specify
        # the normal component of the velocity such that \int(n.u) = 0, or
        # specify Dirichlet conditions for the pressure somewhere.
        #
        A = assemble(a2)
        b = assemble(L2)

        # If the right hand side is flawed (e.g., by round-off errors), then it
        # may have a component b1 in the direction of the null space,
        # orthogonal to the image of the operator:
        #
        #     b = b0 + b1.
        #
        # When starting with initial guess x0=0, the minimal achievable
        # relative tolerance is then
        #
        #    min_rel_tol = ||b1|| / ||b||.
        #
        # If ||b|| is very small, which is the case when ui is almost
        # divergence-free, then min_rel_to may be larger than the prescribed
        # relative tolerance tol. This happens, for example, when the time
        # steps is very small.
        # Sanitation of right-hand side is easy with
        #
        #     e = Function(P)
        #     e.interpolate(Constant(1.0))
        #     evec = e.vector()
        #     evec /= norm(evec)
        #     print(b.inner(evec))
        #     b -= b.inner(evec) * evec
        #
        # However it's hard to decide when the right-hand side is inconsistent
        # because of round-off errors in previous steps, or because the system
        # is actually inconsistent (insufficient boundary conditions or
        # something like that). Hence, don't do anything and rather try to
        # fight the cause for round-off.

        # In principle, the ILU preconditioner isn't advised here since it
        # might destroy the semidefiniteness needed for CG.
        #
        # The system is consistent, but the matrix has an eigenvalue 0. This
        # does not harm the convergence of CG, but with preconditioning one has
        # to make sure that the preconditioner preserves the kernel. ILU might
        # destroy this (and the semidefiniteness). With AMG, the coarse grid
        # solves cannot be LU then, so try Jacobi here.
        # <http://lists.mcs.anl.gov/pipermail/petsc-users/2012-February/012139.html>
        #

        # TODO clear everything; possible in FEniCS 2017.1
        # <https://fenicsproject.org/qa/12916/clear-petscoptions>
        # PETScOptions.clear()

        prec = PETScPreconditioner('hypre_amg')
        PETScOptions.set(
            'pc_hypre_boomeramg_relax_type_coarse',
            'jacobi'
            )
        solver = PETScKrylovSolver('cg', prec)
        solver.parameters['absolute_tolerance'] = 0.0
        solver.parameters['relative_tolerance'] = tol
        solver.parameters['maximum_iterations'] = 1000
        solver.parameters['monitor_convergence'] = verbose
        solver.parameters['error_on_nonconvergence'] = True

        # Create solver and solve system
        A_petsc = as_backend_type(A)
        b_petsc = as_backend_type(b)
        p1_petsc = as_backend_type(p1.vector())
        solver.set_operator(A_petsc)

        solver.solve(p1_petsc, b_petsc)
    return p1


def _step(
        dt,
        u, p0,
        u_bcs, p_bcs,
        rho, mu,
        stabilization,
        time_step_method,
        f,
        rotational_form=False,
        verbose=True,
        tol=1.0e-10,
        ):
    '''Incremental pressure correction scheme scheme as described in section
    3.4 of

        An overview of projection methods for incompressible flows;
        Guermond, Miev, Shen;
        Comput. Methods Appl. Mech. Engrg. 195 (2006),
        <http://www.math.tamu.edu/~guermond/PUBLICATIONS/guermond_minev_shen_CMAME_2006.pdf>.
    '''
    # Some initial sanity checkups.
    assert dt > 0.0
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

    # Define trial and test functions
    v = TestFunction(u[0].function_space())
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

        ui = Function(u[0].function_space())

        # F1 is scaled with `k / rho`.
        if time_step_method == 'forward euler':
            alpha = 1.0
            F1 = (
                inner(ui - u[0], v) * dx
                - k/rho * _rhs_weak(u[0], v, f[0], rho, mu)
                )
        elif time_step_method == 'backward euler':
            alpha = 1.0
            F1 = (
                inner(ui - u[0], v) * dx
                - k/rho * _rhs_weak(ui, v, f[1], rho, mu)
                )
        elif time_step_method == 'crank-nicolson':
            alpha = 1.0
            F1 = (
                inner(ui - u[0], v) * dx
                - k/rho * 0.5 * (
                    _rhs_weak(u[0], v, f[0], rho, mu) +
                    _rhs_weak(ui,   v, f[1], rho, mu)
                    )
                )
        else:
            assert time_step_method == 'bdf2'
            alpha = 1.5
            F1 = (
                inner(1.5 * ui - 2 * u[0] + 0.5 * u[-1], v) * dx
                - k/rho * _rhs_weak(ui, v, f[1], rho, mu)
                )

        if p0:
            F1 += k/rho * inner(grad(p0), v) * dx

        # if stabilization:
        #     tau = supg(u[0], mu/rho)
        #     R = rho*(ui - u[0])/k
        #     if abs(theta) > DOLFIN_EPS:
        #         R -= theta * _rhs_strong(ui, f[1], rho, mu)
        #     if abs(1.0-theta) > DOLFIN_EPS:
        #         R -= (1.0-theta) * _rhs_strong(u[0], f[0], rho, mu)
        #     if p0:
        #         R += grad(p0)
        #     # TODO use u0 or ui here?
        #     F1 += k/rho * tau * dot(R, grad(v)*u[0]) * dx

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
        ui.assign(u[0])

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
                    'absolute_tolerance': tol,
                    'relative_tolerance': 0.0,
                    'error_on_nonconvergence': True
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
        # If Dirichlet boundary conditions are applied to both u* and u_{n+1}
        # (the latter in the final step), the boundary integral vanishes.
        #
        # Assume that on the boundary
        #   L2 -= inner(n, rho/k (u_bcs - ui)) * q * ds
        # is zero. This requires the boundary conditions to be set for ui as
        # well as u_final.
        # This creates some problems if the boundary conditions are supposed to
        # remain 'free' for the velocity, i.e., no Dirichlet conditions in
        # normal direction. In that case, one needs to specify Dirichlet
        # pressure conditions.
        #
        p1 = _pressure_poisson(
                p0,
                mu, ui,
                divu=alpha * rho/k * div(ui),
                p_bcs=p_bcs,
                p_n=None,
                rotational_form=rotational_form,
                tol=tol,
                verbose=verbose
                )
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Velocity correction.
    #   U = U0 - dt/rho \nabla p.
    with Message('Computing velocity correction'):
        u2 = TrialFunction(u[0].function_space())
        a3 = inner(u2, v) * dx

        phi = p1
        if p0:
            phi -= p0
        if rotational_form:
            phi += mu * div(ui)

        L3 = inner(ui,  v) * dx \
            - k/rho * inner(grad(phi), v) * dx
        u1 = Function(u[0].function_space())
        solve(a3 == L3, u1,
              bcs=u_bcs,
              solver_parameters={
                  'linear_solver': 'iterative',
                  'symmetric': True,
                  'preconditioner': 'hypre_amg',
                  'krylov_solver': {
                      'relative_tolerance': tol,
                      'absolute_tolerance': 0.0,
                      'maximum_iterations': 100,
                      'monitor_convergence': verbose,
                      'error_on_nonconvergence': True
                      }
                  })
    return u1, p1


class Chorin(object):
    order = {
        'velocity': 1.0,
        'pressure': 0.5,
        }

    def __init__(self, stabilization=False):
        self.stabilization = stabilization
        return

    # p0 and f0 aren't necessary here, we just keep it around to interface
    # equality with IPCS.
    def step(
            self,
            dt,
            u, p0,
            u_bcs, p_bcs,
            rho, mu,
            f,
            verbose=True,
            tol=1.0e-10
            ):
        return _step(
            dt,
            u, Function(p0.function_space()),
            u_bcs, p_bcs,
            rho, mu,
            self.stabilization,
            'backward euler',
            f,
            verbose=verbose,
            tol=tol,
            )


class IPCS(object):
    order = {
        'velocity': 2.0,
        'pressure': 1.0,
        }

    def __init__(self, time_step_method='backward euler', stabilization=False):
        self.time_step_method = time_step_method
        self.stabilization = stabilization
        return

    def step(
            self,
            dt,
            u, p0,
            u_bcs, p_bcs,
            rho, mu,
            f,
            verbose=True,
            tol=1.0e-10
            ):
        return _step(
            dt,
            u, p0,
            u_bcs, p_bcs,
            rho, mu,
            self.stabilization,
            self.time_step_method,
            f,
            verbose=verbose,
            tol=tol
            )


class Rotational(object):
    order = {
        'velocity': 2.0,
        'pressure': 1.5,
        }

    def __init__(self, time_step_method='backward euler', stabilization=False):
        self.time_step_method = time_step_method
        self.stabilization = stabilization
        return

    def step(
            self,
            dt,
            u, p0,
            u_bcs, p_bcs,
            rho, mu,
            f,
            verbose=True,
            tol=1.0e-10
            ):
        return _step(
            dt,
            u, p0,
            u_bcs, p_bcs,
            rho, mu,
            self.stabilization,
            self.time_step_method,
            f,
            rotational_form=True,
            verbose=verbose,
            tol=tol
            )
