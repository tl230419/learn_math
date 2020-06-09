#%matplotlib notebook # why error?

from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

G = 9.8
L1 = 2.0
L2 = 1.0
M1 = 1.0
M2 = 1.0

def derivs(state, t):
    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    del_ = state[2] - state[0]
    den1 = (M1 + M2) * L1 - M2 * L1 * cos(del_) * cos(del_)
    dydx[1] = (M2 * L1 * state[1] * state[1] * sin(del_) * cos(del_) +
                M2 * G * sin(state[2]) * cos(del_) +
               M2 * L2 * state[3] * state[3] * sin(del_) -
               (M1 + M2) * G * sin(state[0])) / den1

    dydx[2] = state[3]

    den2 = (L2 / L1) *den1
    dydx[3] = (-M2 * L2 * state[3] * state[3] * sin(den1) * cos(del_) +
               (M1 + M2) * G * sin(state[0]) * cos(del_) -
               (M1 + M2) * L1 * state[1] * state[1] * sin(del_) -
               (M1 + M2) * G * sin(state[2])) / den2

    return dydx

dt = 0.05 # step
t = np.arange(0.0, 100, dt) # All run time 100 seconds

# th1 th2 Initial degree
# w1 w2 Initial angle speed
th1 = 120.0
w1 = 0.0
th2 = -10.0
w2 = 0.0

# Initial state
state = np.radians([th1, w1, th2, w2])

# integrate your ODE using scipy.integrate
y = integrate.odeint(derivs, state, t)

x1 = L1 * sin(y[:, 0])
y1 = -L1 * cos(y[:, 0])

x2 = L2 * sin(y[:, 2]) + x1
y2 = -L2 *cos(y[:, 2]) + y1

fig = plt.figure(figsize=(5, 5), dpi=80)
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-4, 4), ylim=(-4, 4))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i * dt))
    return line, time_text

ani = animation.FuncAnimation(fig, animation, np.arange(1, len(y)),
                              interval=100, blit=True, init_func=init)
#ani.save('double_pendulum.mp4', fps=15)
plt.show()