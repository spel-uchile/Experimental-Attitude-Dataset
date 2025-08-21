"""
Created by:

@author: Elias Obreque
els.obrq@gmail.com
"""

import numpy as np

rtol = 1e-5
atol = 1e-08

# quaternion = [i, j, k, 1]


class Quaternions(object):
    def __init__(self, quaterion_ini):
        q_ = np.array([0.0, 0.0, 0.0, 1.0])
        if len(quaterion_ini) == 4:
            q_ = np.array(quaterion_ini)
        elif len(quaterion_ini) == 2:
            rot = quaterion_ini[1]
            # print(quaterion_ini[0])
            rot *= 0.5
            q_[3] = np.cos(rot)
            # print(q_)
            q_[0:3] = quaterion_ini[0] * np.sin(rot)
            # print(q_)
        elif len(quaterion_ini) == 1:
            # a non unit vector v is provided
            rot = np.linalg.norm(quaterion_ini) # scalar rotation rot = |v|
            if rot!=0:
                u = np.array(quaterion_ini)/rot # unit vector u = v / |v|
            else:
                u = np.array(quaterion_ini)     # u = v = 0 in this case
            q_[0:3] = u * np.sin(0.5*rot)
            q_[3] = np.cos(0.5*rot)
        else:
            raise Exception('[Quaternion] Bad initialization, arguments must be a vector of length 4, 2 (rot, '
                            'unit_dir) or one (dir)')
        self.q = q_

    def __call__(self, *args, **kwargs):
        return self.q

    def __mul__(self, r_quat):
        temp = np.zeros(4)
        left_quat = self.q
        right_quat = r_quat.q
        temp[0] = left_quat[3] * right_quat[0] - left_quat[2] * right_quat[1] + left_quat[1] * right_quat[2] + left_quat[0] * right_quat[3]
        temp[1] = left_quat[2] * right_quat[0] + left_quat[3] * right_quat[1] - left_quat[0] * right_quat[2] + left_quat[1] * right_quat[3]
        temp[2] = -left_quat[1] * right_quat[0] + left_quat[0] * right_quat[1] + left_quat[3] * right_quat[2] + left_quat[2] * right_quat[3]
        temp[3] = -left_quat[0] * right_quat[0] - left_quat[1] * right_quat[1] - left_quat[2] * right_quat[2] + left_quat[3] * right_quat[3]
        # temp2 = np.zeros(4)
        # temp2[3] = left_quat[3] * right_quat[3] - left_quat[:3].dot(right_quat[:3])
        # temp2[:3] = left_quat[3] * right_quat[:3] + right_quat[3] * left_quat[:3] +
        # np.cross(left_quat[:3], right_quat[:3])
        return Quaternions(temp)

    @staticmethod
    def get_from_two_v(u: np.array, v: np.array):
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)
        p_ = np.cross(u, v)
        if np.linalg.norm(p_) > 1e-9:
            p_ /= np.linalg.norm(p_)
        else:
            p_ = np.zeros(3)
        ang_ = u @ v
        ang_ = np.arccos(ang_)
        # p_ *= np.tan(np.arccos(ang_))
        q_ = np.array([*-np.sin(ang_ / 2) * p_, np.cos(ang_ / 2)])
        return Quaternions(q_)

    def setquaternion(self, setvalue):
        q_ = np.array([0.0, 0.0, 0.0, 1.0])
        if len(setvalue) == 4:
            q_ = np.array(setvalue)
        elif len(setvalue) == 2:
            rot = setvalue[1]
            rot *= 0.5
            q_[3] = np.cos(rot)
            q_[0:3] = setvalue[0] * np.sin(rot)
        elif len(setvalue) == 1:
            # a non unit vector v is provided
            rot = np.linalg.norm(setvalue)     # scalar rotation rot = |v|
            if rot!=0:
                u = np.array(setvalue)/rot     # unit vector u = v / |v|
            else:
                u = np.array(setvalue)     # u = v = 0 in this case
            q_[0:3] = u * np.sin(0.5*rot)
            q_[3] = np.cos(0.5*rot)
        else:
            raise Exception('[Quaternion] Bad value assignment, arguments must be a vector of length 4, 2'
                            ' (rot, unit_dir) or one (dir)')
        self.q = q_

    def normalize(self):
        div = np.linalg.norm(self.q)
        self.q = self.q / div

    def conjugate(self):
        return np.array([-self.q[0],
                         -self.q[1],
                         -self.q[2],
                         self.q[3]])

    def conjugate_class(self):
        return Quaternions(self.conjugate())

    def frame_conv(self, v, method=None):
        if method is None:
            q = Quaternions(self.q)
            qvqT = Quaternions(q.conjugate()) * Quaternions([v[0], v[1], v[2], 0]) * q
            ans = qvqT()[0:3]
        elif method == "q":
            q = Quaternions(self.q)
            qvqT = q * Quaternions([v[0], v[1], v[2], 0]) * Quaternions(q.conjugate())
            ans = qvqT()[0:3]
        elif method == 'dcm':
            dcm = self.todcm()
            ans = dcm.dot(v)
        else:
            raise Exception('[Quaternion] Unknown method for frame convertion')
        return ans

    def get_angle(self, error_flag=False):
        return np.arccos(self.q[3] if not error_flag else np.abs(self.q[3])) * 2.0

    def todcm(self):
        q1 = self.q[0]
        q2 = self.q[1]
        q3 = self.q[2]
        q4 = self.q[3]

        dcm = [[q1 ** 2 - q2 ** 2 - q3 ** 2 + q4 ** 2, 2 * (q1 * q2 + q3 * q4), 2 * (q1 * q3 - q2 * q4)],
               [2 * (q1 * q2 - q3 * q4), -q1 ** 2 + q2 ** 2 - q3 ** 2 + q4 ** 2, 2 * (q2 * q3 + q1 * q4)],
               [2 * (q1 * q3 + q2 * q4), 2 * (q2 * q3 - q1 * q4), -q1 ** 2 - q2 ** 2 + q3 ** 2 + q4 ** 2]]
        return np.array(dcm)

    def toeuler(self):
        """
        This function finds the angles of the classical Euler sequence
        R3(gamma)*R1(beta)*R3(alpha) from the direction cosine matrix.
        Q - direction cosine matrix
        alpha - first angle of the sequence (deg)
        beta - second angle of the sequence (deg)
        gamma - third angle of the sequence (deg)
        """
        Q = self.todcm()
        alpha = np.arctan2(Q[2, 0], -Q[2, 1])
        beta = np.arccos(Q[2, 2])
        gamma = np.arctan2(Q[0, 2], Q[1, 2])
        return alpha, beta, gamma

    def toypr(self):
        """
        This function finds the angles of the yaw-pitch-roll sequence
        R1(gamma)*R2(beta)*R3(alpha) from the direction cosine matrix.
        Q - direction cosine matrix
        yaw - yaw angle (deg)
        pitch - pitch angle (deg)
        roll - roll angle (deg)
        """
        Q = self.todcm()
        yaw = np.arctan2(Q[0, 1], Q[0, 0])
        pitch = np.arcsin(-Q[0, 2])
        roll = np.arctan2(Q[1, 2], Q[2, 2])

        if abs(yaw) >= np.pi:
            yaw = yaw - 2 * np.sign(yaw) * np.pi
        if abs(pitch) >= np.pi:
            pitch = pitch - 2 * np.sign(pitch) * np.pi
        if abs(roll) >= np.pi:
            roll = roll - 2 * np.sign(roll) * np.pi
        return yaw, pitch, roll

    @staticmethod
    def quat_from_ypr(yaw_, pitch_, roll_):
        q_yaw = Quaternions([np.array([0, 0, 1]), yaw_])
        q_pitch = Quaternions([np.array([0, 1, 0]), pitch_])
        q_roll = Quaternions([np.array([1, 0, 0]), roll_])
        q_ = q_yaw * q_pitch * q_roll  # TODO: documentar transformacion
        q_.normalize()
        return q_


if __name__ == '__main__':
    vb = np.array([1.0, 0, 1.0])
    vb /= np.linalg.norm(vb)
    vlvlh = np.array([0, 0, 1.0])

    rot = np.cross(vb, vlvlh)
    rot /= np.linalg.norm(rot)
    ang = np.arccos(np.dot(vlvlh, vb))
    q_lvlh2b = Quaternions([rot, ang])
    q_lvlh2b.normalize()

    q_b2lvlh = Quaternions(q_lvlh2b.conjugate())
    vlvlh_ = q_b2lvlh.frame_conv(vb)
    q1_ = Quaternions([np.array([0, 1, 0]), np.deg2rad(45)])
    q2_ = Quaternions([np.array([1, 0, 0]), np.deg2rad(90)])
    q3_ = Quaternions([np.array([0, 0, 1]), np.deg2rad(90)])
    q4_ = q3_ * q1_ * q2_
    print(q4_())
