# Approximation

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import matplotlib
matplotlib.use('tkAgg')

def f(x):
    return np.sin(x) + 0.5 * x

def create_plot(x, y, styles, labels, axlabels):
    plt.figure()
    for i in range(len(x)):
        plt.plot(x[i], y[i], styles[i], label=labels[i])
        plt.xlabel(axlabels[0])
        plt.ylabel(axlabels[1])
    plt.legend(loc=0)

x = np.linspace(-2 * np.pi, 2 * np.pi, 50)

create_plot([x], [f(x)], ['b'], ['f(x)'], ['x', 'f(x)'])
plt.show()

# Regression

# monomials as basis functions
res = np.polyfit(x, f(x), deg=1, full=True)   # linear regression step
ry = np.polyval(res[0], x)                    # evaluation using the regression params

create_plot([x, x], [f(x), ry], ['b', 'r.'], ['f(x)', 'regression'], ['x', 'f(x)'])
plt.show()

# again, with a polynomial of degree 5
res = np.polyfit(x, f(x), deg=5, full=True)   # linear regression step
ry = np.polyval(res[0], x)                    # evaluation using the regression params

create_plot([x, x], [f(x), ry], ['b', 'r.'], ['f(x)', 'regression'], ['x', 'f(x)'])
plt.show()

# again, with a polynomial of degree 7
reg = np.polyfit(x, f(x), 7)
ry = np.polyval(reg, x)

np.allclose(f(x), ry)

np.mean((f(x) - ry) ** 2)

create_plot([x, x], [f(x), ry], ['b', 'r.'], ['f(x)', 'regression'], ['x', 'f(x)'])
plt.show()

# individual basis functions
matrix = np.zeros((3 + 1, len(x)))
matrix[3, :] = x ** 3
matrix[2, :] = x ** 2
matrix[1, :] = x
matrix[0, :] = 1

reg = np.linalg.lstsq(matrix.T, f(x), rcond=None)[0]

ry = np.dot(reg, matrix)

create_plot([x, x], [f(x), ry], ['b', 'r.'],
            ['f(x)', 'regression'], ['x', 'f(x)'])
plt.show()

# we can exploit previous knowledge. We know that there is a sin in the function
# therefore, we can include that information in the basis functions
matrix[3, :] = np.sin(x)

reg = np.linalg.lstsq(matrix.T, f(x), rcond=None)[0]

ry = np.dot(reg, matrix)

np.allclose(f(x), ry)

np.mean((f(x) - ry) ** 2)

create_plot([x, x], [f(x), ry], ['b', 'r.'],
            ['f(x)', 'regression'], ['x', 'f(x)'])
plt.show()

# noisy data
xn = np.linspace(-2 * np.pi, 2 * np.pi, 50)
xn = xn + 0.15 * np.random.standard_normal(len(xn))
yn = f(xn) + 0.25 * np.random.standard_normal(len(xn))

reg = np.polyfit(xn, yn, 7)
ry = np.polyval(reg, xn)

create_plot([x, x], [f(x), ry], ['b', 'r.'],
            ['f(x)', 'regression'], ['x', 'f(x)'])
plt.show()

# unsorted data
xu = np.random.rand(50) * 4 * np.pi - 2 * np.pi
yu = f(xu)

reg = np.polyfit(xu, yu, 5)
ry = np.polyval(reg, xu)

create_plot([xu, xu], [yu, ry], ['b.', 'ro'],
            ['f(x)', 'regression'], ['x', 'f(x)'])
plt.show()

# multiple dimensions
def fm(p):
    x, y = p
    return np.sin(x) + 0.25 * x + np.sqrt(y) + 0.05 * y ** 2

x = np.linspace(0, 10, 20)
y = np.linspace(0, 10, 20)
X, Y = np.meshgrid(x, y)

Z = fm((X, Y))
x = X.flatten()
y = Y.flatten()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2,
                       cmap='coolwarm', linewidth=0.5,
                       antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# we can also change the set of basis functions as well
matrix = np.zeros((len(x), 6 + 1))
matrix[:, 6] = np.sqrt(y)
matrix[:, 5] = np.sin(x)
matrix[:, 4] = y ** 2
matrix[:, 3] = x ** 2
matrix[:, 2] = y
matrix[:, 1] = x
matrix[:, 0] = 1

reg = np.linalg.lstsq(matrix, fm((x, y)), rcond=None)[0]

RZ = np.dot(matrix, reg).reshape((20, 20))

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2,
                       cmap='coolwarm', linewidth=0.5,
                       antialiased=True)
surf2 = ax.plot_wireframe(X, Y, RZ, rstride=2, cstride=2, label='regression')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# Interpolation

# here is a linear splines interpolation
import scipy.interpolate as spi

x = np.linspace(-2 * np.pi, 2 * np.pi, 25)

def f(x):
    return np.sin(x) + 0.5 * x

ipo = spi.splrep(x, f(x), k=1)

iy = spi.splev(x, ipo)

np.allclose(f(x), iy)

create_plot([x, x], [f(x), iy], ['b', 'ro'],
            ['f(x)', 'interpolation'], ['x', 'f(x)'])
plt.show()


#
xd = np.linspace(1.0, 3.0, 50)
iyd = spi.splev(xd, ipo)

create_plot([xd, xd], [f(xd), iyd], ['b', 'ro'],
            ['f(x)', 'interpolation'], ['x', 'f(x)'])
plt.show()

# repetition of the exercise, using cubic splines
ipo = spi.splrep(x, f(x), k=3)
iyd = spi.splev(xd, ipo)

np.allclose(f(xd), iyd)

np.mean((f(xd) - iyd) ** 2)

create_plot([xd, xd], [f(xd), iyd], ['b', 'ro'],
            ['f(x)', 'interpolation'], ['x', 'f(x)'])
plt.show()

# Convex Optimization
def fm(p):
    def fn(v):
        return np.sin(v) + 0.05 * v ** 2
    x, y = p
    return fn(x) + fn(y)

x, y = np.linspace(-10, 10, 50), np.linspace(-10, 10, 50)
X, Y = np.meshgrid(x, y)
Z = fm((X, Y))

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2,
                       cmap='coolwarm', linewidth=0.5,
                       antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# global optimization
import scipy.optimize as sco

def fo(p, output: bool = True):
    x, y = p
    z = np.sin(x) + 0.05 * x ** 2 + np.sin(y) + 0.05 * y ** 2
    if output:
        print(f'{x:8.4f} | {y:8.4f} | {z:8.4f}')
    return z


space = (-10, 10.1, 5)  # from, to, step-size
sco.brute(fo, (space, space), finish=None)

from cytoolz import curry
no_output_fo = curry(fo, output=False)

space = (-10, 10.1, 0.1)  # from, to, step-size
opt1 = sco.brute(no_output_fo, (space, space), finish=None)
print(f"The brute forced minima is: {opt1} which has the value {fm(opt1):.3f}")

# local optimization
opt2 = sco.fmin(fo, opt1, xtol=0.001, ftol=0.001, maxiter=15, maxfun=20)
fm(opt2)

sco.fmin(fo, (2.0, 2.0), maxiter=250)

# constrained optimization
import math

def Eu(p):
    """The function to be minimized in order to maximize utility"""
    def eq(v1, v2, c1, c2):
        return 0.5 * math.sqrt(v1 * c1 + v2 * c2)
    s, b = p
    return -(eq(s, b, 15, 5) + eq(s, b, 5, 12))

# the inequality constraint as a dict object
cons = ({'type': 'ineq',
         'fun': lambda p: 100 - p[0] * 10 - p[1] * 10})

# the boundary values for the parameters (chosen to be wide enough)
bnds = ((0, 1000), (0, 1000))

# the constrained optimization
result = sco.minimize(Eu, [5, 5], method='SLSQP', bounds=bnds, constraints=cons)

result['x']  # the optimal parameter values
-result['fun']  # the negative min function value as the optimal solution value
np.dot(result['x'], [10, 10])  # the budget constraint is binding; all wealth is invested


# Integration
import scipy.integrate as sci

def f(x):
    return np.sin(x) + 0.5 * x


x = np.linspace(0, 10)
y = f(x)
a, b = 0.5, 9.5
Ix = np.linspace(a, b)
Iy = f(Ix)

# Numerical Integration

sci.fixed_quad(f, a, b)  # fixed Gaussian quadrature
sci.quad(f, a, b)        # adaptive quadrature
sci.romberg(f, a, b)     # Romberg integration
