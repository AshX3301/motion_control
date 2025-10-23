import numpy as np
import math
import matplotlib.pyplot as plt
import csv

# ---------------- Vehicle & Controller Parameters ----------------
WB = 2.9  # wheelbase [m]
dt = 0.1  # time step [s]
MAX_STEER = math.radians(80)  # max steering angle
MAX_STEERING_CHANGE = 0.2 * 2 * math.pi  # max rad per step
TARGET_SPEED = 2.0  # m/s (constant speed)
LOOKAHEAD = 5.0  # Pure Pursuit lookahead distance

# ---------------- Utility Functions ----------------
def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def angle_mod(yaw):
    while yaw > math.pi: yaw -= 2*math.pi
    while yaw < -math.pi: yaw += 2*math.pi
    return yaw

# ---------------- Vehicle State ----------------
class VehicleState:
    def __init__(self, x=0, y=0, yaw=0, v=TARGET_SPEED):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.prev_steering = 0.0

    def update(self, delta):
        # rate limit steering
        delta_change = delta - self.prev_steering
        if delta_change > MAX_STEERING_CHANGE:
            delta = self.prev_steering + MAX_STEERING_CHANGE
        elif delta_change < -MAX_STEERING_CHANGE:
            delta = self.prev_steering - MAX_STEERING_CHANGE
        self.prev_steering = delta

        # clamp steering
        delta = np.clip(delta, -MAX_STEER, MAX_STEER)

        # kinematic bicycle model
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / WB * math.tan(delta) * dt
        self.yaw = angle_mod(self.yaw)

# ---------------- Pure Pursuit Controller ----------------
def pure_pursuit_control(state, path, target_idx):
    # find lookahead point
    while target_idx < len(path):
        if distance((state.x, state.y), path[target_idx]) >= LOOKAHEAD:
            break
        target_idx += 1
    if target_idx >= len(path):
        target_idx = len(path)-1
    target_point = path[target_idx]

    # compute steering
    dx = target_point[0] - state.x
    dy = target_point[1] - state.y
    alpha = math.atan2(dy, dx) - state.yaw
    delta = math.atan2(2 * WB * math.sin(alpha)/LOOKAHEAD, 1.0)
    return delta, target_idx

# ---------------- PID CTE Controller ----------------
Kp, Ki, Kd = 1.0, 0.0, 0.0
pid_integral, pid_prev_error = 0.0, 0.0

def pid_cte_control(state, path, target_idx):
    global pid_integral, pid_prev_error
    # closest path point
    dxs = [state.x - p[0] for p in path]
    dys = [state.y - p[1] for p in path]
    dists = np.hypot(dxs, dys)
    min_idx = np.argmin(dists)
    target_point = path[min_idx]

    # heading based on previous step
    heading = state.yaw
    dx = target_point[0] - state.x
    dy = target_point[1] - state.y
    cte = -math.sin(heading)*dx + math.cos(heading)*dy

    # PID
    derivative = (cte - pid_prev_error)/dt
    pid_integral += cte * dt
    delta = Kp*cte + Ki*pid_integral + Kd*derivative
    pid_prev_error = cte
    return delta

# ---------------- Load CSV path ----------------
def load_csv_path(filename):
    path = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            x = float(row[2])  # x column
            y = float(row[3])  # y column
            path.append((x,y))
    return path

# ---------------- Simulation Loop ----------------
def simulate(path):
    # Initialize
    state_pp = VehicleState(x=path[0][0], y=path[0][1], yaw=0)
    state_pid = VehicleState(x=path[0][0], y=path[0][1], yaw=0)
    traj_pp, traj_pid = [], []
    target_idx_pp = 0

    for t in range(1000):
        # Pure Pursuit
        delta_pp, target_idx_pp = pure_pursuit_control(state_pp, path, target_idx_pp)
        state_pp.update(delta_pp)
        traj_pp.append((state_pp.x, state_pp.y))

        # PID
        delta_pid = pid_cte_control(state_pid, path, target_idx_pp)
        state_pid.update(delta_pid)
        traj_pid.append((state_pid.x, state_pid.y))

    # Plot
    path_x, path_y = zip(*path)
    pp_x, pp_y = zip(*traj_pp)
    pid_x, pid_y = zip(*traj_pid)

    plt.figure(figsize=(10,6))
    plt.plot(path_x, path_y, 'r-', label='Reference Path')
    plt.plot(pp_x, pp_y, 'b-', label='Pure Pursuit')
    plt.plot(pid_x, pid_y, 'g-', label='PID CTE')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()
