import sys
sys.path.append('..')
import numpy as np
import control
from dynamics.quad_dynamics import quad_dynamics
from state_model.compute_models import compute_ss_model
import parameters.quadrotor_parameters as QUAD
import parameters.simulation_parameters as SIM
import parameters.sensor_parameters as SENSOR


# ---------------------------------------------------------------------------
# ------------------------ DESIRED TARGETS ----------------------------------
# ---------------------------------------------------------------------------

desired_spacing = 50    # 300
u_offset = 400          # 0
v_offset = -400         # 0
desired_targets = np.array([[SENSOR.pixel_width/2+desired_spacing + u_offset, SENSOR.pixel_width/2-desired_spacing + u_offset, SENSOR.pixel_width/2-desired_spacing + u_offset],
                            [SENSOR.pixel_height/2-desired_spacing + v_offset, SENSOR.pixel_height/2-desired_spacing + v_offset, SENSOR.pixel_height/2+desired_spacing + v_offset]])

K_p = 1/5  # 1/15

# --------------------------------------------------------------------------
# ------------------------ LQR PARAMETERS ----------------------------------
# --------------------------------------------------------------------------
quad = quad_dynamics(SIM.ts_simulation)

trim_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
trim_input = np.array([QUAD.mass*QUAD.gravity, 0, 0, 0])

A, B = compute_ss_model(quad, trim_state, trim_input)
C = np.eye(9)

u_max = 0.02                    # 0.5
v_max = 0.02                    # 0.5
w_max = 0.5                     # 0.5
phi_max = np.deg2rad(15)        # 15
theta_max = np.deg2rad(15)      # 15
psi_max = np.deg2rad(15)        # 15
p_max = np.deg2rad(15)          # 15
q_max = np.deg2rad(15)          # 15
r_max = np.deg2rad(15)          # 15

Q = np.diag((1.0/u_max**2,   1.0/v_max**2,     1.0/w_max**2,
             1.0/phi_max**2, 1.0/theta_max**2, 1.0/psi_max**2,
             1.0/p_max**2,   1.0/q_max**2,     1.0/r_max**2))

f_max = 50          # 50
tau_x_max = 0.01    # 0.02
tau_y_max = 0.01    # 0.02
tau_z_max = 0.01    # 0.02

R = np.diag((1.0/f_max**2, 1.0/tau_x_max**2, 1.0/tau_y_max**2, 1.0/tau_z_max**2))

# +f, -f, +tau_x, -tau_x, +tau_y, -tau_y, +tau_z, -tau_z
# limit_lqr = np.array([[200], [0], [25.0], [-25.0], [25.0], [-25.0], [25.0], [-25.0]])
limit_lqr = np.array([[np.inf], [0], [np.inf], [-np.inf], [np.inf], [-np.inf], [np.inf], [-np.inf]])
