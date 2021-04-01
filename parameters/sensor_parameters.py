import sys
sys.path.append('..')
import numpy as np

# -----------------------------------------------------------------------------
# ------------------------ CAMERA PARAMETERS ----------------------------------
# -----------------------------------------------------------------------------

f = 0.01    # focal length
sensor_width = 0.0127   # 1/2 inch
pixel_width = 1920
pixel_height = 1080
aspect_ratio = pixel_width/pixel_height
pixels_per_meter = pixel_width/sensor_width

# -----------------------------------------------------------------------------
# ------------------------ KALMAN PARAMETERS ----------------------------------
# -----------------------------------------------------------------------------

xhat_0 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T

# Process Covariance Terms
sig_w_u = 0.01
sig_w_v = 0.01
sig_w_w = 0.01
sig_w_phi = np.deg2rad(0.05)
sig_w_theta = np.deg2rad(0.05)
sig_w_psi = np.deg2rad(0.05)
sig_w_p = np.deg2rad(0.04)
sig_w_q = np.deg2rad(0.04)
sig_w_r = np.deg2rad(0.04)
Q_N = np.diag((sig_w_u**2, sig_w_v**2, sig_w_w**2, sig_w_phi**2, sig_w_theta**2, sig_w_psi**2, sig_w_p**2, sig_w_q**2, sig_w_r**2))

# Noise Covariance Terms
sig_n_u = 0.2
sig_n_v = 0.2
sig_n_w = 0.2
sig_n_phi = np.deg2rad(0.5)
sig_n_theta = np.deg2rad(0.5)
sig_n_psi = np.deg2rad(0.5)
sig_n_p = np.deg2rad(0.2)
sig_n_q = np.deg2rad(0.2)
sig_n_r = np.deg2rad(0.2)
R_N = np.diag((sig_n_u**2, sig_n_v**2, sig_n_w**2, sig_n_phi**2, sig_n_theta**2, sig_n_psi**2, sig_n_p**2, sig_n_q**2, sig_n_r**2))
