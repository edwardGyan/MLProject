# Source - https://stackoverflow.com/a/43903064
# Posted by Vinícius Figueiredo, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-18, License - CC BY-SA 3.0

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 10

for c, m, zlow, zhigh in [('r', 'o', 0, 100)]:
    xs = randrange(n, 0, 50)
    ys = randrange(n, 0, 50)
    zs = randrange(n, zlow, zhigh)
    ax.scatter(xs, ys, zs, c=c, marker=m)

for c, m, zlow, zhigh in [('b', '^', 0, 100)]:
    xs = randrange(n, 60, 100)
    ys = randrange(n, 60, 100)
    zs = randrange(n, zlow, zhigh)
    ax.scatter(xs, ys, zs, c=c, marker=m)


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

xm,ym = np.meshgrid(xs, ys)

ax.plot_surface(xm, ym, xm, color='green', alpha=0.5) # Data values as 2D arrays as stated in reference - The first 3 arguments is what you need to change in order to turn your plane into a boundary decision plane.  

plt.show()
