"""
6-DOF rigid body flight dynamics model - fixed wing aircraft
Longitudinal model: states = [u, w, q, theta, x, z]

State vector (6 states):
    u     : forward body velocity        (m/s)
    w     : vertical body velocity       (m/s)   positive = DOWNWARD
    q     : pitch rate                   (rad/s)
    theta : pitch angle                  (rad)
    x     : horizontal inertial position (m)
    z     : vertical inertial position   (m)     positive = DOWNWARD (NED)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from aerocoefficients import get_aero_coefficients

# -----------------------------------------------------------
# Aircraft parameters
# -----------------------------------------------------------
PARAMS = {
    'm': 1000.0,      # Mass of the aircraft (kg)
    'S': 16.0,        # Wing reference area (m²)
    'c_bar': 1.5,     # Mean aerodynamic chord (m)
    'b': 10.0,        # Wingspan (m)
    'Iyy': 4000.0,    # Moment of inertia for pitch (kg·m²)
    'rho': 1.225,     # Standard sea-level air density (kg/m³)
    'CD_parasite': 0.025, # Parasite drag coefficient (accounts for fuselage, gear, etc.)
    'g': 9.81,        # Gravitational acceleration (m/s²)
    'T': 1230.0,      # Engine thrust, assumed constant for this model (N)
}
initial_state = [
    64.0,            # u     : cruise speed (m/s)
    0.0,             # w     : no vertical velocity at start
    0.0,             # q     : no pitch rate at start
    np.radians(0.0), # theta : small nose-up pitch angle (rad)
    0.0,             # x     : start at origin
    -1000.0          # z     : 1000 m altitude (negative because z is DOWN in NED)
]


def compute_alpha(u, w):
    """
    Angle of attack: angle between chord line and oncoming airflow.
    Returns alpha in radians.
    """
    return np.arctan2(w, u)


def compute_airspeed(u, w):
    """
    Total airspeed magnitude from body velocity components.
    Ignores sideslip (v=0) for longitudinal model.
    Returns V in m/s.
    """
    return np.hypot(u, w)


def compute_dynamic_pressure(V, rho):
    """
    Returns q_bar in Pa (N/m^2).
    """
    return 0.5 * rho * V**2


def compute_forces(state, params):
    """
    Net forces on the aircraft in the body frame.

    state  : [u, w, q, theta, x, z]
    params : aircraft parameter dictionary

    Returns (Fx, Fz) in Newtons.

    Sign convention (NED, z positive downward):
        Fx positive = forward (accelerates aircraft)
        Fz positive = downward
        Lift acts UPWARD so it reduces Fz (negative contribution)
    """
    u, w, q, theta, x, z = state

    m   = params['m']
    g   = params['g']
    S   = params['S']
    rho = params['rho']
    T   = params['T']
    CD_parasite = params['CD_parasite']

    # --- Step 1: aerodynamic setup ---
    alpha  = compute_alpha(u, w)
    V      = compute_airspeed(u, w)
    q_bar  = compute_dynamic_pressure(V, rho)

    alpha_deg = np.degrees(alpha)
    CL, CD, Cm = get_aero_coefficients(alpha_deg)

    # --- Step 2: lift and drag magnitudes ---

    CD_airfoil = CD
    CD_total = CD_airfoil + CD_parasite
    
    L = q_bar * S * CL
    D = q_bar * S * CD_total

    # --- Step 3: resolve L and D into body axes ---
    # aero_x: drag pulls backward (negative), lift has a small forward
    #         component when alpha > 0
    # aero_z: lift acts upward = NEGATIVE z in NED convention
    aero_x =  L * np.sin(alpha) - D * np.cos(alpha)
    aero_z = -L * np.cos(alpha) - D * np.sin(alpha)

    # --- Step 4: weight components in body frame ---
    # When pitched up (theta > 0): weight pulls backward along body-x
    # Weight always has a downward (positive z) component along body-z
    W_x = -m * g * np.sin(theta)   # negative = opposes forward motion when nose up
    W_z =  m * g * np.cos(theta)   # positive = downward in NED

    # --- Step 5: sum all forces ---
    # It is a FIXED value from PARAMS — not computed to balance forces.
    # The simulation finds its own equilibrium naturally.
    Fx = aero_x + W_x + T
    Fz = aero_z + W_z

    return Fx, Fz


def compute_moment(state, params):
    """
    Returns M in N.m
    Positive M = nose up.
    For pitch stability: dCm/dalpha must be negative (checked in NACA data).
    """
    u, w, q, theta, x, z = state

    S     = params['S']
    c_bar = params['c_bar']
    rho   = params['rho']

    alpha    = compute_alpha(u, w)
    alpha_deg = np.degrees(alpha)
    V        = compute_airspeed(u, w)
    q_bar    = compute_dynamic_pressure(V, rho)

    _, _, Cm = get_aero_coefficients(alpha_deg)
    M_damping = q_bar * S * c_bar * Cmq *(q * c_bar / (2 * V))  # pitch damping moment
    M_aero = q_bar * S * c_bar * Cm
    M_total = M_aero + M_damping

    return M_total


def equations_of_motion(t, state, params):
    """
    The full longitudinal equations of motion.
    scipy calls this function at every timestep.
    """
    u, w, q, theta, x, z = state

    m   = params['m']
    Iyy = params['Iyy']

    Fx, Fz = compute_forces(state, params)
    M      = compute_moment(state, params)

    # Newton's 2nd law + Coriolis corrections
    du = Fx / m - q * w
    dw = Fz / m + q * u

    # Euler's rotation equation
    dq = M / Iyy

    # Kinematics: pitch angle rate = pitch rate
    dtheta = q

    # You must rotate by theta to convert between them.
    dx =  u * np.cos(theta) + w * np.sin(theta)
    dz = -u * np.sin(theta) + w * np.cos(theta)

    return [du, dw, dq, dtheta, dx, dz]


def run_simulation(t_span, initial_state, params, max_step=0.05):
    """
    Returns scipy OdeResult object.
        result.t      — time array
        result.y      — state array, shape (6, n_timesteps)
        result.y[0]   — u history
        result.y[5]   — z history  (altitude = -result.y[5])
     """
    result = solve_ivp(
        fun=equations_of_motion,
        t_span=t_span,
        y0=initial_state,
        args=(params,),       # passed as extra arguments to equations_of_motion
        method='RK45',
        max_step=max_step,
        dense_output=False,
    )
    return result
   

if __name__ == "__main__":
    # --- Run simulation ---
    print("=== Running simulation ===")
    t_span = (0, 60)
    result = run_simulation(t_span, initial_state, PARAMS)

    if result.success:
        print(f"Complete: {len(result.t)} timesteps over {t_span[1]} seconds")
        print(f"Initial altitude : {-initial_state[5]:.1f} m")
        print(f"Final altitude   : {-result.y[5, -1]:.1f} m")
        print(f"Initial speed    : {initial_state[0]:.1f} m/s")
        print(f"Final speed      : {result.y[0, -1]:.1f} m/s")
    else:
        print("Simulation failed:", result.message)
        exit()

    # --- Plot results ---
    altitude = -result.y[5]              # NED: altitude = -z
    airspeed = np.hypot(result.y[0], result.y[1])  # V = sqrt(u^2 + w^2)
    alpha_history = np.degrees(np.arctan2(result.y[1], result.y[0]))

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle('6-DOF Longitudinal Flight Dynamics — NACA 2412', fontsize=13)

    axes[0].plot(result.t, altitude)
    axes[0].set_ylabel('Altitude (m)')
    axes[0].set_xlabel('Time (s)')
    axes[0].grid(True)

    axes[1].plot(result.t, airspeed)
    axes[1].set_ylabel('Airspeed (m/s)')
    axes[1].set_xlabel('Time (s)')
    axes[1].grid(True)

    axes[2].plot(result.t, alpha_history)
    axes[2].set_ylabel('Angle of Attack (deg)')
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=150)
    plt.show()
    print("Plot saved to simulation_results.png")