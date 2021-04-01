import numpy as np
import parameters.sensor_parameters as SENSOR
import parameters.simulation_parameters as SIM
from tools.tools import RotationVehicle2Body
import pyqtgraph as pg

class camera:
    def __init__(self, targets):
        self.f = SENSOR.f
        self.aspect = SENSOR.aspect_ratio
        self.targets = targets
        self.position = np.zeros(3)
        self.orientation = np.zeros(3)
        self.image = pg.plot()
        curve = self.image.plot()
        curve.getViewBox().invertY(True)
        self.image.setAspectLocked()
        self.image.setXRange(0, SENSOR.pixel_width, padding=0.1)
        self.image.setYRange(0, SENSOR.pixel_height, padding=0.1)
        edges = np.array([[0, SENSOR.pixel_width, SENSOR.pixel_width, 0, 0],
                          [0, 0, SENSOR.pixel_height, SENSOR.pixel_height, 0]])
        self.image.plot(edges[0], edges[1], pen=0)
        self.time = 0

    def update_state(self, state):
        self.position = np.array((state.pn, state.pe, state.pd))
        self.orientation = np.array((state.phi, state.theta, state.psi))

    def get_target_pixels(self):
        R_ib = RotationVehicle2Body(self.orientation.item(0), self.orientation.item(1), self.orientation.item(2))
        R_bc = np.array([[0, 1, 0],
                         [-1, 0, 0],
                         [0, 0, 1]])
        R_ic = R_bc @ R_ib
        p_t = self.targets.get_targets() - self.position.reshape(-1, 1)
        e_targets = R_ic @ p_t
        for i in range(len(p_t[0])):
            e_targets[:, i] = self.f/e_targets[2, i]*e_targets[:, i]*SENSOR.pixels_per_meter
        return e_targets[0:2, :] + np.array([[SENSOR.pixel_width/2], [SENSOR.pixel_height/2]])

    def plot_desired_targets(self, desired_targets):
        self.image.plot(desired_targets[0], desired_targets[1], pen=None, symbol='+', symbolBrush=(255, 0, 0, 255))

    def show_image(self, ts):
        self.time += ts
        if self.time > SIM.ts_plotting:
            self.time = 0
            p_t = self.get_target_pixels()
            self.image.plot(p_t[0], p_t[1], pen=None, symbol='o', symbolBrush=(0, 0, 255, 255))
