# -*- coding: utf-8 -*-
#
class AB2R():
    # AB2/TR method as described in 3.16.4 of
    #
    #     Incompressible flow and the finite element method;
    #     Volume 2: Isothermal laminar flow;
    #     P.M. Gresho, R.L. Sani.
    #
    # Here, the Navier-Stokes equation is written as
    #
    #     Mu' + (K+N(u)) u + Cp = f,
    #     C^T u = g.
    #
    # For incompressible Navier-Stokes,
    #
    #     rho (u' + u.nabla(u)) = - nabla(p) + mu Delta(u) + f,
    #     div(u) = 0,
    #
    # we have
    #
    #     M = rho,
    #     K = - mu \Delta,
    #     N(u) = rho * u.nabla(u),
    #     C = nabla,
    #     C^T = div,
    #     g = 0.
    #
    def __init__(self):
        return

    # Initial AB2/TR step.
    def ab2tr_step0(u0,
                    P,
                    f,  # right-hand side
                    rho,
                    mu,
                    dudt_bcs=None,
                    p_bcs=None,
                    eps=1.0e-4,  # relative error tolerance
                    verbose=True
                    ):
        if dudt_bcs is None:
            dudt_bcs = []

        if p_bcs is None:
            p_bcs = []

        # Make sure that the initial velocity is divergence-free.
        alpha = norm(u0, 'Hdiv0')
        if abs(alpha) > DOLFIN_EPS:
            warn('Initial velocity not divergence-free (||u||_div = %e).'
                 % alpha
                 )
        # Get the initial u0' and p0 by solving the linear equation system
        #
        #     [M   C] [u0']   [f0 - (K+N(u0)u0)]
        #     [C^T 0] [p0 ] = [ g0'            ],
        #
        # i.e.,
        #
        #     rho u0' + nabla(p0) = f0 + mu\Delta(u0) - rho u0.nabla(u0),
        #     div(u0')            = 0.
        #
        W = u0.function_space()
        WP = W*P

        # Translate the boundary conditions into product space. See
        # <http://fenicsproject.org/qa/703/boundary-conditions-in-product-space>.
        dudt_bcs_new = []
        for dudt_bc in dudt_bcs:
            dudt_bcs_new.append(DirichletBC(WP.sub(0),
                                            dudt_bc.value(),
                                            dudt_bc.user_sub_domain()))
        p_bcs_new = []
        for p_bc in p_bcs:
            p_bcs_new.append(DirichletBC(WP.sub(1),
                                         p_bc.value(),
                                         p_bc.user_sub_domain()))

        new_bcs = dudt_bcs_new + p_bcs_new

        (u, p) = TrialFunctions(WP)
        (v, q) = TestFunctions(WP)

        # a = rho * dot(u, v) * dx + dot(grad(p), v) * dx \
        a = rho * inner(u, v) * dx - p * div(v) * dx \
            - div(u) * q * dx
        L = _rhs_weak(u0, v, f, rho, mu)

        A, b = assemble_system(a, L, new_bcs)

        # Similar preconditioner as for the Stokes problem.
        # TODO implement something better!
        prec = rho * inner(u, v) * dx \
            - p*q*dx
        M, _ = assemble_system(prec, L, new_bcs)

        solver = KrylovSolver('gmres', 'amg')

        solver.parameters['monitor_convergence'] = verbose
        solver.parameters['report'] = verbose
        solver.parameters['absolute_tolerance'] = 0.0
        solver.parameters['relative_tolerance'] = 1.0e-6
        solver.parameters['maximum_iterations'] = 10000

        # Associate operator (A) and preconditioner matrix (M)
        solver.set_operators(A, M)
        # solver.set_operator(A)

        # Solve
        up = Function(WP)
        solver.solve(up.vector(), b)

        # Get sub-functions
        dudt0, p0 = up.split()

        # Choosing the first step size for the trapezoidal rule can be tricky.
        # Chapters 2.7.4a, 2.7.4e of the book
        #
        #     Incompressible flow and the finite element method,
        #     volume 1: advection-diffusion;
        #     P.M. Gresho, R.L. Sani,
        #
        # give some hints.
        #
        #     eps ... relative error tolerance
        #     tau ... estimate of the initial 'time constant'
        tau = None
        if tau:
            dt0 = tau * eps**(1.0/3.0)
        else:
            # Choose something 'reasonably small'.
            dt0 = 1.0e-3
        # Alternative:
        # Use a dissipative scheme like backward Euler or BDF2 for the first
        # couple of steps. This makes sure that noisy initial data is damped
        # out.
        return dudt0, p0, dt0

    def ab2tr_step(
            W, P, dt0, dt_1,
            mu, rho,
            u0, u_1, u_bcs,
            dudt0, dudt_1, dudt_bcs,
            p_1, p_bcs,
            f0, f1,
            tol=1.0e-12,
            verbose=True
            ):
        # General AB2/TR step.
        #
        # Steps are labeled in the following way:
        #
        #   * u_1: previous step.
        #   * u0:  current step.
        #   * u1:  next step.
        #
        # The same scheme applies to all other entities.
        #
        WP = W * P

        # Make sure the boundary conditions fit with the space.
        u_bcs_new = []
        for u_bc in u_bcs:
            u_bcs_new.append(DirichletBC(WP.sub(0),
                                         u_bc.value(),
                                         u_bc.user_sub_domain()))
        p_bcs_new = []
        for p_bc in p_bcs:
            p_bcs_new.append(DirichletBC(WP.sub(1),
                                         p_bc.value(),
                                         p_bc.user_sub_domain()))

        # Predict velocity.
        if dudt_1:
            u_pred = u0 \
                + 0.5*dt0*((2 + dt0/dt_1) * dudt0 - (dt0/dt_1) * dudt_1)
        else:
            # Simple linear extrapolation.
            u_pred = u0 + dt0 * dudt0

        uu = TrialFunctions(WP)
        vv = TestFunctions(WP)

        # Assign up[1] with u_pred and up[1] with p_1.
        # As of now (2013/09/05), there is no proper subfunction assignment in
        # Dolfin, cf.
        # <https://bitbucket.org/fenics-project/dolfin/issue/84/subfunction-assignment>.
        # Hence, we need to be creative here.
        # TODO proper subfunction assignment
        #
        # up1.assign(0, u_pred)
        # up1.assign(1, p_1)
        #
        up1 = Function(WP)
        a = (dot(uu[0],  vv[0]) + uu[1] * vv[1]) * dx
        L = dot(u_pred, vv[0]) * dx
        if p_1:
            L += p_1 * vv[1] * dx
        solve(a == L, up1,
              bcs=u_bcs_new + p_bcs_new
              )

        # Split up1 for easier access.
        # This is not as easy as it may seem at first, see
        # <http://fenicsproject.org/qa/1123/nonlinear-solves-with-mixed-function-spaces>.
        # Note in particular that
        #     u1, p1 = up1.split()
        # doesn't work here.
        #
        u1, p1 = split(up1)

        # Form the nonlinear equation system (3.16-235) in Gresho/Sani.
        # Left-hand side of the nonlinear equation system.
        F = 2.0/dt0 * rho * dot(u1, vv[0]) * dx \
            + mu * inner(grad(u1), grad(vv[0])) * dx \
            + rho * 0.5 * (inner(grad(u1)*u1, vv[0])
                           - inner(grad(vv[0]) * u1, u1)) * dx \
            + dot(grad(p1), vv[0]) * dx \
            + div(u1) * vv[1] * dx

        # Subtract the right-hand side.
        F -= dot(rho*(2.0/dt0*u0 + dudt0) + f1, vv[0]) * dx

        # J = derivative(F, up1)

        # Solve nonlinear system for u1, p1.
        solve(
            F == 0, up1,
            bcs=u_bcs_new + p_bcs_new,
            # J = J,
            solver_parameters={
              # 'nonlinear_solver': 'snes',
              'nonlinear_solver': 'newton',
              'newton_solver': {
                  'maximum_iterations': 5,
                  'report': True,
                  'absolute_tolerance': tol,
                  'relative_tolerance': 0.0
                  },
              'linear_solver': 'direct',
              # 'linear_solver': 'iterative',
              # # The nonlinear term makes the problem
              # # generally nonsymmetric.
              # 'symmetric': False,
              # # If the nonsymmetry is too strong, e.g., if
              # # u_1 is large, then AMG preconditioning
              # # might not work very well.
              # 'preconditioner': 'ilu',
              # #'preconditioner': 'hypre_amg',
              # 'krylov_solver': {'relative_tolerance': tol,
              #                   'absolute_tolerance': 0.0,
              #                   'maximum_iterations': 100,
              #                   'monitor_convergence': verbose}
              })

        # # Simpler access to the solution.
        # u1, p1 = up1.split()

        # Invert trapezoidal rule for next du/dt.
        dudt1 = 2 * (u1 - u0)/dt0 - dudt0

        # Get next dt.
        if dt_1:
            # Compute local trunction error (LTE) estimate.
            d = (u1 - u_pred) / (3*(1.0 + dt_1 / dt))
            # There are other ways of estimating the LTE norm.
            norm_d = numpy.sqrt(inner(d, d) / u_max**2)
            # Get next step size.
            dt1 = dt0 * (eps / norm_d)**(1.0/3.0)
        else:
            dt1 = dt0
        return u1, p1, dudt1, dt1
