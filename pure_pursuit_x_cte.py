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
        return delta  # return actual steering used

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
Kp, Ki, Kd = 0.3, 0.0, 0.05  # tuned gains
pid_integral, pid_prev_error = 0.0, 0.0

def pid_cte_control(state, path):
    global pid_integral, pid_prev_error
    # closest path point
    dxs = [state.x - p[0] for p in path]
    dys = [state.y - p[1] for p in path]
    dists = np.hypot(dxs, dys)
    min_idx = np.argmin(dists)
    target_point = path[min_idx]

    # heading based on vehicle
    heading = state.yaw
    dx = target_point[0] - state.x
    dy = target_point[1] - state.y
    cte = -math.sin(heading)*dx + math.cos(heading)*dy

    # PID
    derivative = (cte - pid_prev_error)/dt
    pid_integral += cte * dt
    delta = Kp*cte + Ki*pid_integral + Kd*derivative
    pid_prev_error = cte
    return delta, cte

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
    # Initialize vehicle states
    state_pp = VehicleState(x=path[0][0], y=path[0][1], yaw=0)
    state_pid = VehicleState(x=path[0][0], y=path[0][1], yaw=0)

    traj_pp, traj_pid = [], []
    lat_err_pp, lat_err_pid = [], []
    steer_pp, steer_pid = [], []

    target_idx_pp = 0

    while target_idx_pp < len(path)-1:
        # --- Pure Pursuit ---
        delta_pp, target_idx_pp = pure_pursuit_control(state_pp, path, target_idx_pp)
        delta_pp_actual = state_pp.update(delta_pp)
        traj_pp.append((state_pp.x, state_pp.y))
        # lateral error
        dx = state_pp.x - path[target_idx_pp][0]
        dy = state_pp.y - path[target_idx_pp][1]
        lat_err_pp.append(math.hypot(dx, dy))
        steer_pp.append(delta_pp_actual)

        # --- PID CTE ---
        delta_pid, cte = pid_cte_control(state_pid, path)
        delta_pid_actual = state_pid.update(delta_pid)
        traj_pid.append((state_pid.x, state_pid.y))
        lat_err_pid.append(abs(cte))
        steer_pid.append(delta_pid_actual)

    # ---------------- Plots ----------------
    path_x, path_y = zip(*path)
    pp_x, pp_y = zip(*traj_pp)
    pid_x, pid_y = zip(*traj_pid)

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(path_x, path_y, 'r-', label='Reference Path')
    plt.plot(pp_x, pp_y, 'b-', label='Pure Pursuit')
    plt.plot(pid_x, pid_y, 'g-', label='PID CTE')
    plt.axis('equal')
    plt.grid(True)
    plt.title("Trajectory Comparison")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(lat_err_pp, 'b-', label='Pure Pursuit Error')
    plt.plot(lat_err_pid, 'g-', label='PID CTE Error')
    plt.grid(True)
    plt.xlabel("Time step")
    plt.ylabel("Lateral Error [m]")
    plt.title("Lateral Deviation")
    plt.legend()
    plt.show()

    # Steering angles plot
    plt.figure(figsize=(10,4))
    plt.plot(steer_pp, 'b-', label='Pure Pursuit Steering [rad]')
    plt.plot(steer_pid, 'g-', label='PID Steering [rad]')
    plt.grid(True)
    plt.xlabel("Time step")
    plt.ylabel("Steering angle [rad]")
    plt.title("Steering Commands")
    plt.legend()
    plt.show()

# ---------------- Main Entry ----------------
if __name__ == '__main__':
    path = load_csv_path("reference.csv")  # replace with your CSV path
    simulate(path)
