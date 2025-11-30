import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Define Parameters (Task 1a)
# ==========================================
# You need to choose A, B, a, b such that:
# - |x| <= 0.16, |y| <= 0.16
# - Starts and ends at (0,0)
# - a/b is rational
A = 0.11  # Amplitude X (must be <= 0.16)
B = 0.11  # Amplitude Y (must be <= 0.16)
a = 5.0   # Frequency X
b = 4.0   # Frequency Y

# Parametric end time (T_param).
# For sin(t), a full period is usually 2*pi (or multiples thereof)
T_param = 2 * np.pi

# Robot movement duration (tf) - Lab says max 15 seconds
t_f = 15.0
t_acc = 1.0 # Acceleration time (ramp up/down duration)

# Simulation Step size
dt = 1/500

# ==========================================
# 2. Calculate Arc Length & Constant Velocity c
# ==========================================
# Create temporary high-res time vector for arc length calc
t_raw = np.arange(0, T_param + dt, dt)

# Derivatives with respect to the parameter (tau)
# x(tau) = A sin(a*tau) -> x'(tau) = A*a*cos(a*tau)
# y(tau) = B sin(b*tau) -> y'(tau) = B*b*cos(b*tau)
dxdtau = A * a * np.cos(a * t_raw)
dydtau = B * b * np.cos(b * t_raw)

# Arc length integral: integral of sqrt(x'^2 + y'^2) dtau
integrand = np.sqrt(dxdtau**2 + dydtau**2)
L = np.trapz(integrand, t_raw) # Numerical integration using trapezoidal rule

# Calculate required average velocity c
c = L / t_f

print("-" * 30)
print(f"Parameters: A={A}, B={B}, a={a}, b={b}")
print(f"Total Arc Length (L): {L:.4f} meters")
print(f"Target Duration (tf): {t_f} seconds")
print(f"Required Average Velocity (c): {c:.4f} m/s")

if c > 0.25:
    print("WARNING: Velocity c exceeds 0.25 m/s limit!")
else:
    print("Velocity is within safety limits.")
print("-" * 30)

# ==========================================
# 3. Forward-Euler Integration for Alpha (Task 1b)
# ==========================================

def get_trapezoid_scale(t_current, tf, ta):
    """
    Returns the scaling factor g(t) for a trapezoidal velocity profile.
    Normalized such that average value over [0, tf] is 1.
    """
    if tf <= 2 * ta:
        raise ValueError("Acceleration time ta is too large for duration tf")

    # Peak height to ensure area under curve equals tf (so average is 1)
    # Area = Peak * (tf - ta) = tf  => Peak = tf / (tf - ta)
    peak = tf / (tf - ta)

    if t_current < ta:
        return peak * (t_current / ta)
    elif t_current < (tf - ta):
        return peak
    elif t_current < tf:
        return peak * ((tf - t_current) / ta)
    else:
        return 0.0

# Time vector for the actual robot motion
time_steps = np.arange(0, t_f + dt, dt)
N = len(time_steps)
alpha = np.zeros(N)

# Forward-Euler Loop
for k in range(N - 1):
    t_curr = time_steps[k]
    alpha_curr = alpha[k]

    # Calculate geometric derivative at current alpha
    # sqrt(x_d'(alpha)^2 + y_d'(alpha)^2)
    dx_dalpha = A * a * np.cos(a * alpha_curr)
    dy_dalpha = B * b * np.cos(b * alpha_curr)
    geom_deriv = np.sqrt(dx_dalpha**2 + dy_dalpha**2)

    # Trapezoidal profile scaling
    g_val = get_trapezoid_scale(t_curr, t_f, t_acc)

    # alpha_dot = c * g(t) / geometric_derivative
    if geom_deriv < 1e-6: geom_deriv = 1e-6 # Avoid division by zero
    alpha_dot = (c * g_val) / geom_deriv

    # Update alpha
    alpha[k + 1] = alpha_curr + alpha_dot * dt

# ==========================================
# 4. Generate Trajectory & Calculate Velocity (Task 1c)
# ==========================================
# Calculate final trajectory based on alpha
x_traj = A * np.sin(a * alpha)
y_traj = B * np.sin(b * alpha)

# Calculate velocity v[k] numerically using manual difference
# Formula: v[k] = sqrt( ((x[k]-x[k-1])/dt)^2 + ((y[k]-y[k-1])/dt)^2 )
dx = np.diff(x_traj)
dy = np.diff(y_traj)
vx = dx / dt
vy = dy / dt

# Total speed (magnitude of velocity vector)
speed_diff = np.sqrt(vx**2 + vy**2)

# np.diff returns array of size N-1. Prepend 0 to match time_steps size (start from rest)
velocity = np.insert(speed_diff, 0, 0.0)

# ==========================================
# 5. Plotting
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: The XY Trajectory
ax1.plot(x_traj, y_traj, label='Robot Path')
ax1.set_title(f"Task 1a: Desired Trajectory (L={L:.3f}m)")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
ax1.axhline(0, color='black', linewidth=0.5)
ax1.axvline(0, color='black', linewidth=0.5)
ax1.axis('equal')
ax1.grid(True)
# Workspace bounds
rect = plt.Rectangle((-0.16, -0.16), 0.32, 0.32, linewidth=1, edgecolor='r', facecolor='none', linestyle='--', label='Workspace')
ax1.add_patch(rect)
ax1.legend(loc='lower right')

# Plot 2: The Velocity Profile (Task 1c)
ax2.plot(time_steps, velocity, label='Actual Velocity')
ax2.set_title("Task 1c: Velocity Profile")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Velocity (m/s)")
ax2.grid(True)

# Add dashed lines as requested
ax2.axhline(0.25, color='red', linestyle='--', label='Limit (0.25 m/s)')
ax2.axhline(c, color='black', linestyle='--', label=f'Avg Speed c ({c:.3f} m/s)')
ax2.legend()

# Plot 3: Alpha(t) (Task 1b)
ax3.plot(time_steps, alpha, label=r'$\alpha(t)$', color='purple')
ax3.set_title("Task 1b: Time Scaling Function")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel(r"$\alpha$ (rad)")
ax3.grid(True)

# Dashed line at t = tf
ax3.axvline(t_f, color='black', linestyle='--', label=f'$t_f$ ({t_f}s)')
# Check if alpha reaches T_param
ax3.axhline(T_param, color='green', linestyle='--', label=f'Target T ({T_param:.2f})')
ax3.legend()

plt.tight_layout()
plt.show()
