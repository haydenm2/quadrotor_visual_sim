import sys
import numpy as np
sys.path.append('..')
import parameters.control_parameters as CON
import parameters.sensor_parameters as SENSOR
from tools.tools import RotationVehicle2Body
from controller.lqr_controller import lqr_control
from message_types.msg_state import msg_state


class controller:
    def __init__(self, ts_control):
        self.control = lqr_control(CON.A, CON.B, CON.Q, CON.R, ts_control)
        self.commanded_state = msg_state()
        self.trim_input = CON.trim_input
        self.inputs = self.trim_input
        self.limit = CON.limit_lqr
        self.Kp = CON.K_p

    def update(self, cmd, state, pts, pts_depth):
        f = SENSOR.f
        u1 = pts[0, 0]/SENSOR.pixels_per_meter
        v1 = pts[1, 0]/SENSOR.pixels_per_meter
        u1_e = u1 - CON.desired_targets[0, 0]/SENSOR.pixels_per_meter
        v1_e = v1 - CON.desired_targets[1, 0]/SENSOR.pixels_per_meter
        z1 = pts_depth.item(0)
        Jp1 = self.image_jacobian(u1, v1, z1, f)
        u2 = pts[0, 1]/SENSOR.pixels_per_meter
        v2 = pts[1, 1]/SENSOR.pixels_per_meter
        u2_e = u2 - CON.desired_targets[0, 1]/SENSOR.pixels_per_meter
        v2_e = v2 - CON.desired_targets[1, 1]/SENSOR.pixels_per_meter
        z2 = pts_depth.item(1)
        Jp2 = self.image_jacobian(u2, v2, z2, f)
        u3 = pts[0, 2]/SENSOR.pixels_per_meter
        v3 = pts[1, 2]/SENSOR.pixels_per_meter
        u3_e = (u3 - CON.desired_targets[0, 2]/SENSOR.pixels_per_meter)
        v3_e = (v3 - CON.desired_targets[1, 2]/SENSOR.pixels_per_meter)
        z3 = pts_depth.item(2)
        Jp3 = self.image_jacobian(u3, v3, z3, f)
        J = np.vstack((Jp1, Jp2, Jp3))
        e = np.array([[u1_e, v1_e, u2_e, v2_e, u3_e, v3_e]]).T
        V_c = -self.Kp*np.linalg.inv(J)@e
        R_cb = np.array([[0, 1, 0],
                         [-1, 0, 0],
                         [0, 0, 1]]).T
        R_bi = RotationVehicle2Body(state.phi, state.theta, state.psi).T
        v_i = R_bi@R_cb@V_c[0:3, 0]
        w_b = R_cb@V_c[3:6, 0]*0

        cmd.u = v_i.item(0)
        cmd.v = v_i.item(1)
        cmd.w = v_i.item(2)
        cmd.p = w_b.item(0)
        cmd.q = w_b.item(1)
        cmd.r = w_b.item(2)

        x = np.array([[state.u - v_i.item(0), state.v - v_i.item(1), state.w - v_i.item(2), state.phi, state.theta, state.psi, state.p - w_b.item(0), state.q - w_b.item(1), state.r - w_b.item(2)]]).T  # using beta as an estimate for v
        u_tilde = self.control.update(x)
        u = u_tilde + self.trim_input.reshape(-1, 1)
        u_sat = self._saturate(u)
        self.inputs = u_sat.reshape(-1, 1)
        self.commanded_state = cmd
        return self.inputs, self.commanded_state

    def image_jacobian(self, u, v, z, f):
        return np.array([[-f/z, 0, u/z, u*v/f, -(f**2 + u**2)/f, v],
                        [0, -f/z, v/z, (f**2 + v**2)/f, -u*v/f, -u]])

    def _saturate(self, u):
        # saturate u at +- self.limit
        u_sat = u
        for i in range(len(u)):
            if u.item(i) >= self.limit.item(2 * i):
                u_sat[i, 0] = self.limit.item(2 * i)
            elif u.item(i) <= self.limit.item(2 * i + 1):
                u_sat[i, 0] = self.limit.item(2 * i + 1)
            else:
                u_sat[i, 0] = u.item(i)
        return u_sat
