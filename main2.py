from fenics import *
from mshr import *
import numpy as np
import sympy
import imageio
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
from sympy.printing import ccode
from celluloid import Camera

sympy.init_printing()
import sympy.functions.elementary.trigonometric as sym_trig
import matplotlib.animation as animation

x, y, alpha, t = sympy.symbols('x[0], x[1], alpha, t')

def get_grad(f):
    return (sympy.diff(f, x), sympy.diff(f, y))


def get_lap(f):
    return sympy.diff(f, x, x) + sympy.diff(f, y, y)


def boundary(x, on_boundary):
    return on_boundary and x[1] < 0


def get_animation(images, title_name, fps=10):
    with imageio.get_writer(f'{title_name}.avi', fps=fps) as writer:
        for image in images:
            writer.append_data(image)


def plot(f, is_accurate, to_save):
    n = mesh.num_vertices()
    d = mesh.geometry().dim()
    mesh_coordinates = mesh.coordinates().reshape((n, d))
    triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
    triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)

    fig, ax = plt.subplots()
    zfaces = np.asarray([f(cell.midpoint()) for cell in cells(mesh)])
    ax_plot = plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
    fig.colorbar(ax_plot, ax=ax)
    if is_accurate:
        plt.title('Accurate U calculation')
    else:
        plt.title('Approximate U calculation')
    if to_save:
        plt.savefig(f'{f.__str__()}.png')
    plt.close()
    return fig


def plot_solutions(u_e, u, mesh):
    n = mesh.num_vertices()
    d = mesh.geometry().dim()
    mesh_coordinates = mesh.coordinates().reshape((n, d))
    triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
    triangulation = tri.Triangulation(mesh_coordinates[:, 0],
                                      mesh_coordinates[:, 1], triangles)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    z_faces = np.asarray([u_e(cell.midpoint()) for cell in cells(mesh)])
    ax1_plot = ax1.tripcolor(triangulation, facecolors=z_faces, edgecolors='k')
    z_faces = np.asarray([u(cell.midpoint()) for cell in cells(mesh)])
    ax2_plot = ax2.tripcolor(triangulation, facecolors=z_faces, edgecolors='k')

    ax1.set_title('Accurate U calculation')
    ax2.set_title('Approximate U calculation')
    fig.colorbar(ax1_plot, ax=ax1)
    fig.colorbar(ax2_plot, ax=ax2)
    fig.canvas.draw()
    fig_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    fig_plot = fig_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return fig_plot


# Problem implementing
def main(r):
    alpha_val = 1
    T = 1.0
    dt = T / 100
    u_e = sympy.sin(x) ** 2 + sympy.cos(y) ** 2

    mesh = generate_mesh(Circle(Point(0, 0), r), 30)
    V = FunctionSpace(mesh, 'P', 2)
    u_D = Expression(ccode(u_e), degree=2)
    f = -get_lap(u_e) + alpha * u_e

    bc = DirichletBC(V, u_D, boundary)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression(ccode(f), degree=2, alpha=alpha_val)
    gradient = get_grad(u_e)
    g = Expression(
        f'{ccode(gradient[0])} * x[0] / sqrt(x[0]*x[0] + x[1]*x[1]) + {ccode(gradient[1])} * x[1] / sqrt(x[0]*x[0] + x[1]*x[1])',
        degree=2)
    a = dot(grad(u), grad(v)) * dx + alpha_val * u * v * dx
    L = f * v * dx + g * v * ds
    u = Function(V)
    solve(a == L, u, bc)

    # Accuracy calculation

    error_L2 = errornorm(u_D, u, 'L2')
    vertex_values_u_D = u_D.compute_vertex_values(mesh)
    vertex_values_u = u.compute_vertex_values(mesh)
    error_C = np.max(np.abs(vertex_values_u - vertex_values_u_D))

    print(f'L2 error = {error_L2}')
    print(f'C error = {error_C}')

    # Visualisation

    plot(u, is_accurate=False, to_save=True)
    plot(u_D, is_accurate=True, to_save=True)

    # Thermal conductivity
    steps_number = 50
    # u_e = sympy.sin(x)**2 + sympy.cos(y)**2 + t
    u_e = x * sympy.cos(t ** 2) + t * sympy.sin(y)

    u_D = Expression(ccode(u_e), degree=2, t=0)
    bc = DirichletBC(V, u_D, boundary)
    u_n = interpolate(u_D, V)
    u = TrialFunction(V)
    v = TestFunction(V)

    f = Expression(ccode(sympy.diff(u_e, t) - alpha * get_lap(u_e)), alpha=alpha_val, degree=2, t=0)
    gradient = get_gradient(u_e)
    g = Expression(
        f'{ccode(gradient[0])} * x[0] / sqrt(x[0]*x[0] + x[1]*x[1]) + {ccode(gradient[1])} * x[1] / sqrt(x[0]*x[0] + x[1]*x[1])',
        degree=2, t=0
    )

    T = 5.0
    dt = T / steps_number
    a = u * v * dx + dt * dot(grad(u), grad(v)) * dx
    L = (u_n + dt * f) * v * dx + dt * v * g * ds

    sympy.diff(u_e, t)

    F = u * v * dx + dt * dot(grad(u), grad(v)) * dx - (u_n + dt * f) * v * dx - dt * v * g * ds
    a, L = lhs(F), rhs(F)
    u = Function(V)
    t = 0
    plots = []
    errors_L2 = []
    errors_max = []

    for n in range(steps_number):
        t += dt
        u_D.t = t
        g.t = t
        f.t = t
        solve(a == L, u, bc)
        u_n.assign(u)
        u_e = interpolate(u_D, V)
        plots.append(plot_solutions(u_e, u, mesh))
        errors_L2.append(errornorm(u_e, u, 'L2'))
        errors_max.append(np.abs(u_e.vector().get_local() - u.vector().get_local()).max())

    get_animation(plots, 'my_plots')

    time_arr = np.linspace(dt, T, steps_number)
    plt.plot(time_arr, errors_max, label='Max errors')
    plt.plot(time_arr, errors_L2, label='L2 errors')
    plt.legend()
    plt.savefig(f'errors.png')


main(1)