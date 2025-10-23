# purepursuit_sim.py
import numpy as np
import matplotlib.pyplot as plt

# --- vehicle / sim params ---
dt = 0.05      # s
L = 1.2        # wheelbase (m)
max_steer = np.deg2rad(35)
total_time = 40.0

# --- reference path: sinusoidal rows (agriculture-like) ---
tref = np.linspace(0, 200, 2001)
path_x = tref * 0.2
path_y = 2.0 * np.sin(0.06 * tref)   # sinusoidal rows
path_yaw = np.arctan2(np.gradient(path_y, path_x), 1.0)

# --- initial state [x, y, yaw, v] ---
state = np.array([0.0, -1.0, 0.0, 1.0])  # start offset from row, speed 1 m/s

# --- helper functions ---
def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def nearest_index(x, y, path_x, path_y):
    d = np.hypot(path_x - x, path_y - y)
    return np.argmin(d), d.min()

# --- Pure Pursuit controller ---
look_ahead_k = 2.0  # meters per m/s (tune)
def pure_pursuit_control(state, path_x, path_y, path_yaw, k=look_ahead_k):
    x, y, yaw, v = state
    Ld = k * max(0.5, v)  # look-ahead distance (min 0.5 m)
    # find target point at distance >= Ld
    dists = np.hypot(path_x - x, path_y - y)
    idxs = np.where(dists >= Ld)[0]
    if len(idxs) == 0:
        idx = len(path_x) - 1
    else:
        idx = idxs[0]
    tx, ty = path_x[idx], path_y[idx]
    # transform to vehicle frame
    dx = tx - x; dy = ty - y
    alpha = wrap_to_pi(np.arctan2(dy, dx) - yaw)
    # steering law (rear-axle pure pursuit)
    delta = np.arctan2(2 * L * np.sin(alpha), Ld)
    delta = np.clip(delta, -max_steer, max_steer)
    return float(delta), idx

# --- simulation ---
xs, ys, yaws, vs, times = [], [], [], [], []
t = 0.0
steps = int(total_time / dt)
for i in range(steps):
    delta, tgt_idx = pure_pursuit_control(state, path_x, path_y, path_yaw)
    # simple speed controller: keep constant desired speed
    desired_v = 1.0
    a = 0.0  # no longitudinal accel for now
    # kinematic bicycle integration
    x, y, yaw, v = state
    x += v * np.cos(yaw) * dt
    y += v * np.sin(yaw) * dt
    yaw += v / L * np.tan(delta) * dt
    yaw = wrap_to_pi(yaw)
    v += a * dt
    # save
    state = np.array([x, y, yaw, v])
    xs.append(x); ys.append(y); yaws.append(yaw); vs.append(v); times.append(t)
    t += dt
    # stop if we reached end of path
    if tgt_idx >= len(path_x)-2:
        break

# --- compute cross-track error over time ---
cte = []
for x, y in zip(xs, ys):
    _, dmin = nearest_index(x, y, path_x, path_y)
    cte.append(dmin)

# --- plots ---
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(path_x, path_y, '--', label='ref path')
plt.plot(xs, ys, '-', label='vehicle')
plt.axis('equal'); plt.title('Path'); plt.legend()

plt.subplot(1,2,2)
plt.plot(times[:len(cte)], cte)
plt.xlabel('time (s)'); plt.ylabel('cross-track error (m)')
plt.title('Cross-track error (RMS {:.3f} m)'.format(np.sqrt(np.mean(np.array(cte)**2))))
plt.tight_layout()
plt.show()
