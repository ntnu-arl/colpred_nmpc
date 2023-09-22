from math import pi
import numpy as np
from numpy import matrix as mat
from scipy.spatial.transform import Rotation as Rot
import casadi as cs
from casadi import vertcat, horzcat, diagcat, norm_2  # matrix
from casadi import sqrt, dot, cross, cos, sin, atan2, fabs, acos, asin  # math
from casadi import if_else as ie  # if/else conditional for piecewise linear function
import argparse


## input bounds functions
def symbol_fdot_bounds(omega, dw_min, dw_max, c_f):
    """Compute the casadi symbolic piecewise linear function to define the bounds on fdot.
    omega   -- value of omega for respective dw_min/max
    dw_min  -- minimum angular acceleration for the respective omega value
    dw_max  -- maximum angular acceleration for the respective omega value
    c_f     -- propeller force coefficient
    """
    coeff_dec = [(dw_min[i + 1] - dw_min[i]) / (omega[i + 1] - omega[i]) for i in range(6)]
    coeff_acc = [(dw_max[i + 1] - dw_max[i]) / (omega[i + 1] - omega[i]) for i in range(6)]

    f = cs.SX.sym('f')
    w = sqrt(f / c_f)
    dw_min = ie(w <= omega[0], dw_min[0],
                ie(w <= omega[1], dw_min[0] + (w - omega[0]) * coeff_dec[0],
                ie(w <= omega[2], dw_min[1] + (w - omega[1]) * coeff_dec[1],
                ie(w <= omega[3], dw_min[2] + (w - omega[2]) * coeff_dec[2],
                ie(w <= omega[4], dw_min[3] + (w - omega[3]) * coeff_dec[3],
                ie(w <= omega[5], dw_min[4] + (w - omega[4]) * coeff_dec[4],
                ie(w <= omega[6], dw_min[5] + (w - omega[5]) * coeff_dec[5],
                dw_min[6])))))))
    dw_max = ie(w <= omega[0], dw_max[0],
                ie(w <= omega[1], dw_max[0] + (w - omega[0]) * coeff_acc[0],
                ie(w <= omega[2], dw_max[1] + (w - omega[1]) * coeff_acc[1],
                ie(w <= omega[3], dw_max[2] + (w - omega[2]) * coeff_acc[2],
                ie(w <= omega[4], dw_max[3] + (w - omega[3]) * coeff_acc[3],
                ie(w <= omega[5], dw_max[4] + (w - omega[4]) * coeff_acc[4],
                ie(w <= omega[6], dw_max[5] + (w - omega[5]) * coeff_acc[5],
                dw_max[6])))))))
    df_min = cs.Function('df_min', [f], [2*sqrt(c_f)*sqrt(f)*dw_min])
    df_max = cs.Function('df_max', [f], [2*sqrt(c_f)*sqrt(f)*dw_max])
    return df_min, df_max


def symbol_wdot_bounds(omega, dw_min, dw_max):
    """Compute the casadi symbolic piecewise linear function to define the bounds on wdot.
    omega   -- value of omega for respective dw_min/max
    dw_min  -- minimum angular acceleration for the respective omega value
    dw_max  -- maximum angular acceleration for the respective omega value
    """
    coeff_dec = [(dw_min[i + 1] - dw_min[i]) / (omega[i + 1] - omega[i]) for i in range(6)]
    coeff_acc = [(dw_max[i + 1] - dw_max[i]) / (omega[i + 1] - omega[i]) for i in range(6)]
    w = cs.SX.sym('w')
    dw_min = ie(w <= omega[0], dw_min[0],
                ie(w <= omega[1], dw_min[0] + (w - omega[0]) * coeff_dec[0],
                ie(w <= omega[2], dw_min[1] + (w - omega[1]) * coeff_dec[1],
                ie(w <= omega[3], dw_min[2] + (w - omega[2]) * coeff_dec[2],
                ie(w <= omega[4], dw_min[3] + (w - omega[3]) * coeff_dec[3],
                ie(w <= omega[5], dw_min[4] + (w - omega[4]) * coeff_dec[4],
                ie(w <= omega[6], dw_min[5] + (w - omega[5]) * coeff_dec[5],
                dw_min[6])))))))
    dw_max = ie(w <= omega[0], dw_max[0],
                ie(w <= omega[1], dw_max[0] + (w - omega[0]) * coeff_acc[0],
                ie(w <= omega[2], dw_max[1] + (w - omega[1]) * coeff_acc[1],
                ie(w <= omega[3], dw_max[2] + (w - omega[2]) * coeff_acc[2],
                ie(w <= omega[4], dw_max[3] + (w - omega[3]) * coeff_acc[3],
                ie(w <= omega[5], dw_max[4] + (w - omega[4]) * coeff_acc[4],
                ie(w <= omega[6], dw_max[5] + (w - omega[5]) * coeff_acc[5],
                dw_max[6])))))))
    dw_min_f = cs.Function('dw_min', [w], [dw_min])
    dw_max_f = cs.Function('dw_max', [w], [dw_max])
    return dw_min_f, dw_max_f


## rotations (quat, euler, mat, angle-axis)
def quat2rot(q):
    """Compute the rotation matrix associated to a quaternion
    q       -- quaternion with scalar as first element [qw qx qy qz]
    """
    r11 = q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2
    r21 = 2 * (q[1] * q[2] + q[0] * q[3])
    r31 = 2 * (q[1] * q[3] - q[0] * q[2])
    r12 = 2 * (q[1] * q[2] - q[0] * q[3])
    r22 = q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2
    r32 = 2 * (q[2] * q[3] + q[0] * q[1])
    r13 = 2 * (q[1] * q[3] + q[0] * q[2])
    r23 = 2 * (q[2] * q[3] - q[0] * q[1])
    r33 = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2
    return horzcat(vertcat(r11, r21, r31), vertcat(r12, r22, r32), vertcat(r13, r23, r33))


def euler2rot(euler):
    """Compute the rotation matrix associated to euler angles.
    euler   -- euler angles [roll pitch yaw] (Z1Y2X3 convention, see https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix)
    """
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]
    r11 = cos(pitch) * cos(yaw)
    r21 = cos(pitch) * sin(yaw)
    r31 = -sin(pitch)
    r12 = sin(roll) * sin(pitch) * cos(yaw) - cos(roll) * sin(yaw)
    r22 = sin(roll) * sin(pitch) * sin(yaw) + cos(roll) * cos(yaw)
    r32 = sin(roll) * cos(pitch)
    r13 = cos(roll) * sin(pitch) * cos(yaw) + sin(roll) * sin(yaw)
    r23 = cos(roll) * sin(pitch) * sin(yaw) - sin(roll) * cos(yaw)
    r33 = cos(roll) * cos(pitch)
    return horzcat(vertcat(r11, r21, r31), vertcat(r12, r22, r32), vertcat(r13, r23, r33))


def quat2euler(q):
    """Compute the euler angles associated to a quaternion.
    q       -- quaternion with scalar as first element [qw qx qy qz]
    """
    roll = atan2(2 * (q[0]*q[1] + q[2]*q[3]), 1 - 2 * (q[1]*q[1] + q[2]*q[2]))
    pitch = asin(2 * (q[0]*q[2] - q[3]*q[1]))
    yaw = atan2(2 * (q[0]*q[3] + q[1]*q[2]), 1 - 2 * (q[2]*q[2] + q[3]*q[3]))
    # return np.array([roll, pitch, yaw])
    return vertcat(roll, pitch, yaw)


def euler2quat(euler):
    """Compute the quaternion associated to euler angles.
    euler   -- euler angles [roll pitch yaw] (Z1Y2X3 convention, see https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix)
    """
    cr = cos(euler[0] * 0.5)
    sr = sin(euler[0] * 0.5)
    cp = cos(euler[1] * 0.5)
    sp = sin(euler[1] * 0.5)
    cy = cos(euler[2] * 0.5)
    sy = sin(euler[2] * 0.5)

    return vertcat(
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    )


def invert(q):
    """Rerturn the inverse quaternion of q."""
    return (vertcat(q[0], -q[1], -q[2], -q[3])) / norm_2(q)


def hamilton_prod(q1, q2):
    """Return the Hamilton product of 2 quaternions q1*q2."""
    return vertcat(
        q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
        q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
        q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
        q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    )


def dist_geo_quat(q1, q2):
    """Computes the squared geodesic distance between two quaternions."""
    q2i = [q2[0], -q2[1], -q2[2], -q2[3]]  # conjugate (invert) quaternion
    q1q2_inv = hamilton_prod(q1,q2i)
    normv = norm_2(q1q2_inv[1:4])
    return ie(normv < 1e-6, 0, norm_2(2 * q1q2_inv[1:4] * atan2(normv, q1q2_inv[0]) / normv)**2)


def dist_quat(q1, q2):
    """Computes the angular distance between two quaternions."""
    q1n = q1/norm_2(q1)
    q2n = q2/norm_2(q2)
    return 1 - fabs(dot(q1n,q2n))


def deuler_avel_map(euler):
    """Computes the """
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]
    return vertcat(
            horzcat(1, sin(pitch)*sin(roll)/cos(pitch), sin(pitch)*cos(roll)),
            horzcat(0, cos(roll), -sin(pitch)),
            horzcat(0, sin(roll)/cos(pitch), cos(roll)/cos(pitch))
        )


## misc
def skew_mat(v):
    """Computes the skew matrix of a vector."""
    return vertcat(horzcat(    0, -v[2],  v[1]),
                   horzcat( v[2],     0, -v[0]),
                   horzcat(-v[1],  v[0],     0))


def print_mat(mat, f=None, prec=10):
    """Pretty print of matrix."""
    l, c = mat.shape
    for i in range(l):
        for j in range(c):
            print(round(mat[i, j], prec), end=';\n' if c == 1 or j // (c - 1) else ', ', file=f)


def rad(angle):
    """Convert an angle from degrees to radians."""
    return angle * pi / 180


## GTMR allocation matrices
def axis_rot(axis, angle):
    """Compute the rotation matrix around a given axis (x, y, or z), angle in rad."""
    if axis == 'x':
        return mat([[1, 0, 0], [0, cos(angle), -sin(angle)], [0, sin(angle), cos(angle)]])
    elif axis == 'y':
        return mat([[cos(angle), 0, sin(angle)], [0, 1, 0], [-sin(angle), 0, cos(angle)]])
    elif axis == 'z':
        return mat([[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]])


def GTMRP_props(n, l, alpha, beta, com=[0, 0, 0], alpha0=-1):
    """Compute position and orientation of propelers in a Generically Tilted Multi-Rotor Platform.
    n       -- Number of actuators
    l       -- Distance from propellers to CoM
    alpha   -- Alpha tilting angles in absolute value (rad)
    beta    -- Beta tilting angles in absolute value (rad)
    com     -- Position of the geometrical center of props wrt CoM
    alpha0  -- Sign of alpha tilting angle for the first propeller
    """
    R = [axis_rot('z', i * (pi / (n / 2))) @ axis_rot('x', alpha0 * (-1) ** i * alpha) @ axis_rot('y', beta) for i in range(n)]
    p = [l * axis_rot('z', i * (pi / (n / 2))) @ mat('1; 0; 0') + mat(com).T for i in range(n)]
    return p, R


def GTMRP_matrix(R, p, c_f, c_t, sign=1):
    """Compute allocdigitsation matrix for a Generically Tilted Multi-Rotor Platform.
    R       -- (3x3) orientation matrices of the n propellers
    p       -- (1x3) position vectors of the n propellers
    c_f     -- Propellers' force coefficient
    c_t     -- Propellers' torque coefficient
    sign    -- Rotation direction for 1st prop (-1:counter-clockwise, 1:clockwise)
    """
    r = range(len(R))
    Riz = [R[i] @ mat('0;0;1') for i in r]
    G_f = np.column_stack([Riz[i] for i in r])
    G_t = np.column_stack([np.cross(p[i], Riz[i], 0, 0, 0) + c_t[i] / c_f[i] * sign * (-1) ** i * Riz[i] for i in r])
    return G_f, G_t


def alloc(n, l, alpha, beta, c_f, c_t, com=[0, 0, 0], sign=1, alpha0=-1):
    """Compute the force and torque allocation matrices.
    n       -- Number of actuators
    l       -- Distance from propellers to CoM
    aplha   -- Alpha tilting angles in absolute value (rad)
    beta    -- Beta tilting angles in absolute value (rad)
    c_f     -- Propellers' force coefficient
    c_t     -- Propellers' torque coefficient
    com     -- 3D position offset between geometrical center of props and CoM
    sign    -- Rotation direction for 1st prop (-1:counter-clockwise, 1:clockwise)
    alpha0  -- Sign of tilting for 1st prop (1 or -1)
    """
    if isinstance(c_f, float):
        c_f = [c_f for i in range(n)]
    if isinstance(c_t, float):
        c_t = [c_t for i in range(n)]
    p, R = GTMRP_props(n, l, rad(alpha), rad(beta), com, alpha0)
    GF, GT = GTMRP_matrix(R, p, c_f, c_t, sign)

    return GF, GT


## main
if __name__ == '__main__':
    """Main function used to generate the allocation matrix as a txt file."""

    parser = argparse.ArgumentParser(description="Generate allocation matrix or propeller 6D pose for a Generically Tilted Multi-Rotor Platform.")
    parser.add_argument(dest='mode', help="mat or pose", type=str)
    parser.add_argument('-f', dest='type', help="Multiply output by c_f or not", action='store_true')
    parser.add_argument('-p', dest='path', help="Output path.", default='./output.txt', type=str)
    parser.add_argument('-n', dest='n', help="Number of propellers", default=4, type=int)
    parser.add_argument('-l', dest='l', help="Size of arm", default=0.23, type=float)
    parser.add_argument('-a', dest='alpha', help="Alpha tilting angle (deg)", default=0, type=float)
    parser.add_argument('-b', dest='beta', help="Beta tilting angle (deg)", default=0, type=float)
    parser.add_argument('-cf', dest='c_f', help="Force coefficient", default=5.9e-4, type=float)
    parser.add_argument('-ct', dest='c_t', help="Torque coefficient", default=1e-5, type=float)
    parser.add_argument('-s', dest='sign', help="Rotation direction for 1st prop (-1:counter-clockwise, 1:clockwise)", default=1, type=int)
    parser.add_argument('-a0', dest='alpha0', help="Sign of tilting for 1st prop (1 or -1)", default=-1, type=int)
    parser.add_argument('-cx', dest='com_x', help="X position offset between geometrical center of props and CoM", default=0, type=float)
    parser.add_argument('-cy', dest='com_y', help="Y position offset between geometrical center of props and CoM", default=0, type=float)
    parser.add_argument('-cz', dest='com_z', help="Z position offset between geometrical center of props and CoM", default=0, type=float)
    parser.add_argument('-r', dest='round', help="Number of significative digits", default=5, type=int)
    args = parser.parse_args()

    com = [args.com_x, args.com_y, args.com_z]

    p, R = GTMRP_props(args.n, args.l, rad(args.alpha), rad(args.beta), com, args.alpha0)

    if args.mode == 'mat':
        GF, GT = GTMRP_matrix(R, p, args.c_f, args.c_t, args.sign)

        G = np.concatenate((GF, GT))

        if args.type:
            G = G * args.c_f
            args.round += 4

        with open(args.path, 'w') as f:
            print_mat(G, f, args.round)

    elif args.mode == 'pose':
        R = mat(Rot.from_matrix(R).as_euler('xyz'))
        with open(args.path, 'w') as f:
            for i in range(args.n):
                print("prop%i:" % (i + 1), file=f)
                print_mat(p[i].T, f, args.round)
                print_mat(R[i], f, args.round)
                print('', file=f)
