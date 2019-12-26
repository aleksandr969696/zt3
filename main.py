import numpy as np
import sympy
from fenics import *
from sympy import diff
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
from sympy.printing import ccode
import sympy.functions.elementary.trigonometric as sym_trigfrom
from mshr import *
import math

x, y, alpha, a, t = sympy.symbols('x[0], x[1], alpha, a, t')
alpha_value = 1

def gradient(u):
    return diff(u, x), diff(u, y)


def laplass(u):
    return diff(u, x, x)+diff(u, y, y)


def boundary(x, on_boundary):
    return on_boundary and (x[1] < 0)


def task1(u_init_str, f_str, g_str, h_str, r):
    mesh = generate_mesh(Circle(Point(0, 0), r), 20)

    V = FunctionSpace(mesh, 'P', 1)

    u_init = Expression(ccode(u_init_str), degree=2)

    print(mesh)
    bc = DirichletBC(V, u_init, boundary)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression(f_str, degree=2, alpha=alpha_value)
    g = Expression(g_str, degree=2)
    a = dot(grad(u), grad(v)) * dx + alpha * u * v * dx

    L = f * v * dx + g * v * ds
    u = Function(V)
    solve(a == L, u, bc)

    # Compute error in L2 norm
    error_L2 = errornorm(u_init, u, 'L2')
    # Compute maximum error at vertices
    vertex_values_u_init = u_init.compute_vertex_values(mesh)
    vertex_values_u = u.compute_vertex_values(mesh)
    error_max = np.max(np.abs(vertex_values_u_init - vertex_values_u))
    # Print errors
    print('error_L2  =', error_L2)
    print('error_max =', error_max)

    n = mesh.num_vertices()
    d = mesh.geometry().dim()
    mesh_coordinates = mesh.coordinates().reshape((n, d))
    triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
    triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)
    plt.figure()
    gs = gridspec.GridSpec(1, 2)

    ax1 = plt.subplot(gs[0])
    zfaces = np.asarray([u(cell.midpoint()) for cell in cells(mesh)])
    ax1.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
    ax1.set_title('solution')

    ax2 = plt.subplot(gs[1])
    zfaces = np.asarray([u_init(cell.midpoint()) for cell in cells(mesh)])
    ax2.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
    ax2.set_title('original')
    print('here')
    plt.show()
    return (u)



def heat(u_init_str, g_str, num_steps, name, r):
    T = 2.0
    dt = T / num_steps
    mesh = generate_mesh(Circle(Point(0, 0), r), 20)
    V = FunctionSpace(mesh, 'P', 2)

    u_init = Expression(ccode(u_init_str), degree=2, t=0)


    def boundary(x, on_boundary):
        if on_boundary:
            if x[1] < 0:
                return True
            else:
                return False
        else:
            return False

    bc = DirichletBC(V, u_init, boundary)

    u_n = interpolate(u_init, V)
    u = TrialFunction(V)
    v = TestFunction(V)
    sympy.diff(u_init, t)
    f = Expression(ccode(sympy.diff(u_init, t) - a * laplass(u_init)), a = 1, degree = 2, t = 0)
    g = Expression(g_str, degree = 2, t = 0)

    F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx - dt * v * g * ds
    a, L = lhs(F), rhs(F)
    u = Function(V)
    t=0

    vtkfile = File('heat_poisson/solution_{0}.pvd'.format(name))

    for n in range(num_steps):
        t += dt
        u_init.t = t
        g.t = t
        f.t = t
        solve(a == L, u, bc)
        vtkfile << (u, t)
        u_e = interpolate(u_init, V)
        error = np.abs(u_e.vector().get_local()- u.vector().get_local()).max()
        print(' t = ', t, ',max error = ', error, 'L2 error = ', errornorm(u_e, u, 'L2'))

        u_n.assign(u)

    n = mesh.num_vertices()
    d = mesh.geometry().dim()
    mesh_coordinates = mesh.coordinates().reshape((n, d))
    triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
    triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)
    plt.figure()
    gs = gridspec.GridSpec(1, 2)

    ax1 = plt.subplot(gs[0])
    zfaces = np.asarray([u(cell.midpoint()) for cell in cells(mesh)])
    ax1.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
    ax1.set_title('solution')

    ax2 = plt.subplot(gs[1])
    zfaces = np.asarray([u_init_str(cell.midpoint()) for cell in cells(mesh)])
    ax2.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
    ax2.set_title('original')


def main():
    u_e = x**2+y**2
    g_str = '2*sqrt(x[0]*x[0]+y[0]*y[0])'
    # g_str = 2*math.sqrt(x*x+y*y)
    f_str = '4+alpha*(x[0]*x[0]+x[1]*x[1])'
    h_str = 'x[0]*x[0]+x[1]*x[1]'
    task1(u_e, f_str, g_str, h_str, 2)


main()
