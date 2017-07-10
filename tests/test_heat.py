#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
'''
Coupled solve of the Navier--Stokes and the heat equation.
'''
from __future__ import print_function

import os

from flow import heat

from dolfin import (
    Constant, Mesh, FunctionSpace, DirichletBC, SubDomain, XDMFFile,
    mpi_comm_world, UnitSquareMesh
    )
import meshio
import pygmsh


def create_mesh(lcar):

    x0 = 0.0
    x1 = 0.1
    y0 = 0.0
    y1 = 0.2

    mesh_eps = 1.0e-12

    # pylint: disable=no-self-use
    class HotBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return (
                on_boundary
                and x0 + mesh_eps < x[0] < x1 - mesh_eps
                and y0 + mesh_eps < x[1] < y1 - mesh_eps
                )
    hot_boundary = HotBoundary()

    class CoolBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return (
                on_boundary and (
                    x[0] < x0 + mesh_eps or
                    x[0] > x1 - mesh_eps or
                    x[1] < y0 + mesh_eps or
                    x[1] > y1 - mesh_eps
                    ))
    cool_boundary = CoolBoundary()

    cache_file = 'boussinesq-{}.msh'.format(lcar)
    if os.path.isfile(cache_file):
        print('Using mesh from cache \'{}\'.'.format(cache_file))
        points, cells, _, _, _ = meshio.read(cache_file)
    else:
        geom = pygmsh.Geometry()

        circle = geom.add_circle(
            [0.05, 0.05, 0.0], 0.02, lcar, make_surface=False
            )

        geom.add_rectangle(
            x0, x1, y0, y1,
            0.0,
            lcar,
            holes=[circle]
            )

        points, cells, _, _, _ = pygmsh.generate_mesh(geom)

        meshio.write(cache_file, points, cells)

    # https://fenicsproject.org/qa/12891/initialize-mesh-from-vertices-connectivities-at-once
    meshio.write('test.xml', points, cells)
    return Mesh('test.xml'), hot_boundary, cool_boundary


def compute_heat(target_time, lcar):
    # mesh, hot_boundary, cool_boundary = create_mesh(lcar)
    mesh_eps = 1.0e-12

    class HotBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] < mesh_eps and x[1] < 0.5
    hot_boundary = HotBoundary()

    class CoolBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (
                x[0] >= mesh_eps or x[1] >= 0.5
                )
    cool_boundary = CoolBoundary()

    mesh = UnitSquareMesh(100, 100)

    room_temp = 0.0
    heater_temp = 1.0

    kappa = 1.0e-5
    rho = 1.0
    cp = 1.0

    Q = FunctionSpace(mesh, 'Lagrange', 2)

    heat_bcs = [
        DirichletBC(Q, heater_temp, hot_boundary),
        DirichletBC(Q, room_temp, cool_boundary),
        ]

    # Use all quantities at room temperature to avoid nonlinearity
    convection = Constant([1.0, 0.0])
    problem = heat.Heat(
                Q, convection,
                kappa, rho, cp,
                heat_bcs, Constant(0.0),
                supg_stabilization=True
                )

    theta0 = problem.solve_stationary()
    theta0.rename('temperature', 'temperature')

    with XDMFFile(mpi_comm_world(), 'heat.xdmf') as xdmf_file:
        xdmf_file.write(theta0)

    return theta0


if __name__ == '__main__':
    compute_heat(target_time=120.0, lcar=0.3e-2)
