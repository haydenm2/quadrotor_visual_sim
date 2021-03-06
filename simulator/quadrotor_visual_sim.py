import sys
sys.path.append('..')
import numpy as np
import parameters.simulation_parameters as SIM
import time

from plotter.data_viewer import data_viewer
from plotter.input_viewer import input_viewer
from controller.controller import controller
from observer.observer import observer
from dynamics.quad_dynamics import quad_dynamics
from targets.targets import targets
from camera.camera import camera
from viewer.world_viewer import quad_viewer
from message_types import msg_state
import parameters.control_parameters as CON

targets = targets()

# initialize the visualization
quad_viewer = quad_viewer(targets)  # initialize the viewer
data_view = data_viewer()  # initialize view of data plots
# sensor_view = data_viewer()  # initialize view of data plots
input_view = input_viewer()

# initialize elements of the architecture
quad = quad_dynamics(SIM.ts_simulation)
ctrl = controller(SIM.ts_simulation)
obs = observer()
cam = camera(targets)
cam.update_state(quad.msg_true_state)
cam.show_image(0)
cam.plot_desired_targets(CON.desired_targets)

# initialize the simulation time
sim_time = SIM.start_time

# main simulation loop
print("Press Command-Q to exit...")
t0 = time.time()
inputs = CON.trim_input
while sim_time < SIM.end_time:
    #------------- reference signal -------------
    desired_state = msg_state
    desired_state.u = 0
    desired_state.v = 0
    desired_state.w = 0
    desired_state.phi = 0
    desired_state.theta = 0
    desired_state.psi = 0
    desired_state.p = 0
    desired_state.q = 0
    desired_state.r = 0

    #------------- observer -------------
    measurements = quad.sensor_measurements
    estimated_state = obs.update(measurements, inputs.flatten() - CON.trim_input)

    #------------- controller -------------
    pts = cam.get_target_pixels()
    pts_depth = cam.get_target_depths()
    inputs, commanded_state = ctrl.update(desired_state, estimated_state, pts, pts_depth)

    #------------- system -------------
    quad.update_state(inputs)  # propagate the MAV dynamics

    #-------------- camera --------------
    cam.update_state(quad.msg_true_state)
    cam.show_image(SIM.ts_simulation)

    #------------- update viewer -------------
    quad_viewer.update(quad.msg_true_state)  # plot path and MAV
    data_view.update(quad.msg_true_state,  # true states
                     estimated_state,
                     commanded_state,
                     SIM.ts_simulation)
    # sensor_view.update(measurements,  # sensor output for debugging
    #                  measurements,
    #                  measurements,
    #                  SIM.ts_simulation)
    input_view.update(inputs, CON.trim_input, SIM.ts_simulation)

    #------------- increment time -------------
    sim_time += SIM.ts_simulation
    if SIM.real_time:
        real_time = time.time() - t0
        if real_time < sim_time:
            time.sleep(sim_time - real_time)
