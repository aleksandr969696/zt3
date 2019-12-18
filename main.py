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

x, y, alpha = sympy.symbols('x[0], x[1], alpha')
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

    u_init = Expression(u_init_str, degree=2)

    print(mesh)
    bc = DirichletBC(V, u_init, boundary)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression(f_str, degree=2, alpha=alpha_value)
    g = Expression(g_str, degree=2)
    a = dot(grad(u), grad(v)) * dx + alpha * u * v * dx

    L = f * v * dx + g * v * ds
    u = Function(V)
    sol = solve(a == L, u, bc)

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


def main():
    u_e = x ** 2 + y ** 2
    g_str = '2*sqrt(x[0]**2+y[0]**2)'
    f_str = '4+alpha*(x**2+y**2)'
    h_str = 'x**2+y**2'
    task1(u_e, f_str, g_str, h_str, 2)


main()
