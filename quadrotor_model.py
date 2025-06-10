import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Parameters – exactly the same symbols used in the Simulink model
# ------------------------------------------------------------------
m   = 0.5                     # mass [kg]
L   = 0.225                   # arm length [m]
k   = 0.01                    # thrust coeff  [N·s²/rad²]
b   = 0.001                   # drag-torque coeff [N·m·s²/rad²]
D   = np.diag([.01, .01, .01])      # linear drag [N·s/m] (inertial)
Ixx, Iyy, Izz = 3e-6, 3e-6, 1e-5    # inertia about body axes [kg·m²]
I   = np.diag([Ixx, Iyy, Izz])
g_i = np.array([0., 0., -9.81])     # gravity in inertial frame

# ------------------------------------------------------------------
# Helper functions — each one corresponds 1-to-1 to a MATLAB-Fcn block
# ------------------------------------------------------------------
def forces(Omega, k=k):
    """
    Simulink block  ‘forces’
    Input : Omega (4×1)  – propeller speeds [rad/s]
    Output: FB    (3×1)  – total thrust   [N] in BODY frame
    """
    T = k * np.sum(Omega**2)
    return np.array([0., 0., T])       # along +z_b


def torques(Omega, k=k, b=b, L=L):
    """
    Simulink block  ‘torques’
    Output: tauB (3×1) – body torques [N·m]  (φ, θ, ψ)
    """
    return np.array([
        L*k*(Omega[0]**2 - Omega[2]**2),                 # roll
        L*k*(Omega[1]**2 - Omega[3]**2),                 # pitch
        b*(Omega[0]**2 - Omega[1]**2 + Omega[2]**2 - Omega[3]**2)  # yaw
    ])


def Rzyx(phi, theta, psi):
    """
    Simulink block ‘Rzyx’  (body → inertial rotation matrix)
    """
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    return np.array([
        [ cth*cps,                     cth*sps,                    -sth],
        [ sph*sth*cps - cph*sps,       sph*sth*sps + cph*cps, sph*cth],
        [ cph*sth*cps + sph*sps,       cph*sth*sps - sph*cps, cph*cth]
    ])


def transl_dynamics(v, eul, FB):
    """
    Simulink block ‘transl_dynamics’
    Inputs : v   (3×1) inertial velocity
             eul (3×1) Euler angles [φ θ ψ]
             FB  (3×1) thrust in BODY frame
    Output : v̇   (3×1) acceleration in inertial frame
    """
    phi, theta, psi = eul
    FB_i = Rzyx(phi, theta, psi) @ FB          # body → inertial
    FD   = -D @ v                              # linear drag
    return (FB_i + FD)/m + g_i                 # Newton


def rot_dynamics(omega, tauB):
    """
    Simulink block ‘rot_dynamics’
    Input : omega (3×1) body angular velocity
            tauB  (3×1) body torques
    Output: ω̇     (3×1)
    """
    w_cross_Iw = np.cross(omega, I @ omega)
    return np.linalg.solve(I, tauB - w_cross_Iw)   # same as  I⁻¹(…)


def W_inv(phi, theta, psi):
    """
    Simulink block ‘W_inv’ (Euler-rate conversion matrix)
    Returns 3×3 matrix W such that  eul̇ = W * omega
    """
    sphi, cphi = np.sin(phi), np.cos(phi)
    tth, cth   = np.tan(theta), np.cos(theta)
    return np.array([
        [1, sphi*tth, cphi*tth],
        [0,     cphi,    -sphi],
        [0, sphi/cth, cphi/cth]
    ])


# ------------------------------------------------------------------
# Continuous-time state derivative  – equivalent to the Simulink net
# ------------------------------------------------------------------
def f(t, x, Omega):
    """
    x = [ p (3) , v (3) , eul (3) , omega (3) ]   ← 12 states
    """
    p     = x[0:3]        # position (unused inside f, but nice to keep)
    v     = x[3:6]
    eul   = x[6:9]
    omega = x[9:12]

    FB   = forces(Omega)
    tauB = torques(Omega)

    # blocks
    v_dot   = transl_dynamics(v, eul, FB)
    omega_dot = rot_dynamics(omega, tauB)
    eul_dot = W_inv(*eul) @ omega

    return np.concatenate([v,          # ṗ  = v
                           v_dot,      # v̇
                           eul_dot,    # eul̇
                           omega_dot]) # ω̇


# ------------------------------------------------------------------
# Simple RK4 integrator to mimic Simulink's fixed-step solver
# ------------------------------------------------------------------
def simulate(Omega, t_end=3.0, dt=1e-3):
    """returns time vector and state matrix"""
    steps = int(t_end/dt)
    t = np.linspace(0, t_end, steps+1)

    # initial conditions = the same ones in the assignment
    x = np.zeros(12)
    X = np.zeros((steps+1, 12))
    X[0] = x

    for k in range(steps):
        k1 = f(t[k],             x,               Omega)
        k2 = f(t[k]+dt/2.0,      x+dt/2*k1,       Omega)
        k3 = f(t[k]+dt/2.0,      x+dt/2*k2,       Omega)
        k4 = f(t[k]+dt,          x+dt   *k3,      Omega)
        x  = x + dt/6.0*(k1+2*k2+2*k3+k4)
        X[k+1] = x

    return t, X


# ------------------------------------------------------------------
# Run the three exercise cases and plot position & Euler angles
# ------------------------------------------------------------------
cases = {
    "all_off" : np.array([0.,   0.,   0.,   0.]),
    "roll"    : np.array([1e4,  0., 1e4,   0.]),
    "pitch"   : np.array([0., 1e4,   0., 1e4])
}

fig1, axs1 = plt.subplots(3, 1, figsize=(6, 8))
fig2, axs2 = plt.subplots(3, 1, figsize=(6, 8))

for idx, (name, Omega_case) in enumerate(cases.items()):
    t, X = simulate(Omega_case)

    p   = X[:, 0:3]
    eul = X[:, 6:9] * 180/np.pi   # deg for readability

    axs1[idx].plot(t, p)
    axs1[idx].set_title(f'Position – {name}')

    axs2[idx].plot(t, eul)
    axs2[idx].set_title(f'Euler angles – {name}')

for ax in list(axs1)+list(axs2):
    ax.grid(True)

plt.tight_layout()
plt.show()
