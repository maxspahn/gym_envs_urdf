# pylint: disable-all
import numpy as np
import casadi as ca
from casadi import cos, sin
import sys


def makeTF(p=np.array([0.0, 0.0, 0.0]), axis="z", angle=0, name="T"):
    T = np.zeros(shape=(4, 4))
    # T = ca.SX.sym(name, (4, 4))
    T[3, 3] = 1.0
    if axis == "z":
        T[1, 1] = cos(angle)
        T[2, 2] = cos(angle)
        T[1, 2] = -sin(angle)
        T[2, 1] = sin(angle)
    T[0:3, 3] = p
    return T


def euler2TF(rpy_j, name="T"):
    T_x = axisAngle2TF(rpy_j[0], np.array([0.0, 0.0, 1.0]), name)
    T_y = axisAngle2TF(rpy_j[1], np.array([0.0, 1.0, 0.0]), name)
    T_z = axisAngle2TF(rpy_j[2], np.array([1.0, 0.0, 0.0]), name)
    return ca.mtimes(T_z, ca.mtimes(T_y, T_x))


def makeTrans(p):
    R = np.identity(3)
    b = np.zeros(shape=(1, 4))
    b[0, 3] = 1.0
    T = ca.vertcat(ca.horzcat(R, p), b)
    return T


def axisAngle2TF(angle, axis, name):
    T = ca.SX.sym(name, (4, 4))
    T[3, :] = 0.0
    T[:, 3] = 0.0
    T[3, 3] = 1.0
    x = axis[0]
    y = axis[1]
    z = axis[2]
    T[0, 0] = 1 + (1 - ca.cos(angle)) * (x * x - 1)
    T[0, 1] = -z * ca.sin(angle) + (1 - ca.cos(angle)) * x * y
    T[0, 2] = y * ca.sin(angle) + (1 - ca.cos(angle)) * x * z
    T[1, 0] = z * ca.sin(angle) + (1 - ca.cos(angle)) * x * y
    T[1, 1] = 1 + (1 - ca.cos(angle)) * (y * y - 1)
    T[1, 2] = -x * ca.sin(angle) + (1 - ca.cos(angle)) * y * z
    T[2, 0] = -y * ca.sin(angle) + (1 - ca.cos(angle)) * x * z
    T[2, 1] = x * ca.sin(angle) + (1 - ca.cos(angle)) * y * z
    T[2, 2] = 1 + (1 - ca.cos(angle)) * (z * z - 1)
    return T


def manual(q, qdot, l, m, I, g):
    v_2 = (
        l[0] ** 2 * qdot[0] ** 2
        + 0.25 * l[1] ** 2 * (qdot[0] + qdot[1]) ** 2
        + l[0] * l[1] * np.cos(q[1]) * qdot[0] * (qdot[0] + qdot[1])
    )
    v_1 = 0.25 * l[0] ** 2 * qdot[0] ** 2
    w_1 = qdot[0]
    w_2 = qdot[0] + qdot[1]
    """
    print('v_1 :', np.sqrt(v_1))
    print('v_2 :', np.sqrt(v_2))
    print('w_1 :', w_1)
    print('w_2 :', w_2)
    """
    T = (
        0.5 * v_1 * m[0]
        + 0.5 * v_2 * m[1]
        + 0.5 * I * w_1 ** 2
        + 0.5 * I * w_2 ** 2
    )
    V = m[0] * g * l[0] * 0.5 * np.sin(q[0]) + m[1] * g * (
        l[0] * np.sin(q[0]) + 0.5 * l[1] * np.sin(q[0] + q[1])
    )
    return T - V


def create3DDynamics(n):

    # parameters and variables
    q = ca.SX.sym("q", n)
    qdot = ca.SX.sym("qdot", n)
    qddot = ca.SX.sym("qddot", n)
    g = ca.SX.sym("g", 1)
    g_vec = ca.vertcat(np.zeros((2, 1)), g)
    I_j = ca.SX.sym("I_j", (9, n))
    com = ca.SX.sym("com", (3, n))
    off_j = ca.SX.sym("off_j", (16, n))
    off_ee = ca.SX.sym("off_ee", (16, n))
    rpy_j = ca.SX.sym("rpy_j", (3, n))
    m = ca.SX.sym("m_0", n)
    axis_j = ca.SX.sym("axis_j", (3, n))
    fk = ca.SX.sym("ee", (3, n))
    k = ca.SX.sym("k", n)

    # inputs
    tau = ca.SX.sym("tau", n)

    # Example for two link
    m_ex = np.array([1.0, 1.0])
    l_ex = np.array([1.0, 1.0])
    I_jex = np.zeros((9, 2))
    I_jex[:, 0] = np.identity(3).flatten() * 0.0858
    I_jex[:, 1] = np.identity(3).flatten()
    com_ex = np.array([[0.0, 0.0], [0.0, 0.0], [l_ex[0] / 2.0, l_ex[1] / 2.0]])
    off_jex = np.zeros((16, 2))
    off_jex[:, 0] = np.identity(4).flatten()
    off_jex[:, 1] = np.identity(4).flatten()
    off_jex[14, 1] = l_ex[1]
    off_eeex = np.zeros((16, 2))
    off_eeex[:, 0] = np.identity(4).flatten()
    off_eeex[:, 1] = np.identity(4).flatten()
    off_eeex[14, 0] = l_ex[1]
    off_eeex[14, 1] = l_ex[1]
    axis_jex = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    g_ex = np.array([-10])
    q_ex = np.array([0.0, 0.0])
    qdot_ex = np.array([0.0, 1.0])
    tau_ex = np.array([1.0, 0.1])

    T = 0.0
    V = 0.0

    # correction simulation in z axis
    # T_cor = axisAngle2TF(0.0 * np.pi/2.0, np.array([1.0, 0.0, 0.0]), name="T_cor")
    # T_0ib = ca.mtimes(T_cor, makeTrans(np.zeros((3, 1))))
    T_0ib = makeTrans(np.zeros((3, 1)))
    for i in range(n):
        # transform between sucessive joints T_i1bi = i-1'_T_i
        # T_i1bi = ca.mtimes(makeTrans(x_j[:, i]), euler2TF(rpy_j[:, i]))
        # T_i1bi = ca.mtimes(ca.reshape(off_j[:, i], 4, 4), T_cor)
        # T_i1bi = ca.mtimes(T_cor, ca.reshape(off_j[:, i], 4, 4))
        T_i1bi = ca.reshape(off_j[:, i], 4, 4)
        """
        test = ca.Function("test", [q, qdot, com, m, I_j, g, axis_j, off_j], [T_i1bi])
        print("test : ", test(q_ex, qdot_ex, com_ex, m_ex, I_ex, g_ex, axis_jex, off_jex))
        """
        # rotation due to joint T_i1bi = i_T_ib
        T_iib = axisAngle2TF(q[i], axis_j[:, i], name="T_iib")
        # Update running transform
        T_i1bib = ca.mtimes(T_i1bi, T_iib)
        T_0ib = ca.mtimes(T_0ib, T_i1bib)
        # transform to center of mass T_ibic = i'_T_ic
        T_ibic = makeTrans(com[:, i])
        print(T_ibic)
        T_0ic = ca.mtimes(T_0ib, T_ibic)
        r_0ic = T_0ic[0:3, 3]
        # transfor to end effector for forward kinematics T_ibiee = i'_T_iee
        # T_ibiee = ca.mtimes(ca.reshape(off_ee[:, i], 4, 4), T_cor)
        T_ibiee = ca.reshape(off_ee[:, i], 4, 4)
        T_0iee = ca.mtimes(T_0ib, T_ibiee)
        fk[0:3, i] = T_0iee[0:3, 3]

        # trans velocities
        J_0ic = ca.jacobian(r_0ic, q)
        v_0ic = ca.mtimes(J_0ic, qdot)

        # angular values
        w_i = axis_j[:, i] * qdot[i]
        for j in range(i):
            w_j = axis_j[:, j] * qdot[j]
            R_j_i = axisAngle2TF(q[j], axis_j[:, j], "R_j_i")[:3, :3]
            w_i += ca.mtimes(ca.transpose(R_j_i), w_j)

        # lagrangian
        T_trans = 0.5 * m[i] * ca.dot(v_0ic, v_0ic)
        I = ca.reshape(I_j[:, i], 3, 3)
        T_rot = 0.5 * ca.dot(w_i, ca.mtimes(I, w_i))
        T += T_trans + T_rot
        V += 1 * m[i] * ca.dot(g_vec, r_0ic)
        """
        test = ca.Function("test", [q, qdot, com, m, I_j, g, axis_j, off_j], [V])
        print(
            "test : ",
            test(
                q_ex[0],
                qdot_ex[0],
                com_ex[:, 0],
                m_ex[0],
                I_jex[:, 0],
                g_ex,
                axis_jex[:, 0],
                off_jex[:, 0],
            ),
        )
        """

    L = T - V

    dL_dq = ca.gradient(L, q)
    dL_dqdot = ca.gradient(L, qdot)

    d2L_dq2 = ca.jacobian(dL_dq, q)
    d2L_dqdqdot = ca.jacobian(dL_dq, qdot)
    d2L_dqdot2 = ca.jacobian(dL_dqdot, qdot)

    M = d2L_dqdot2
    F = d2L_dqdqdot
    f = dL_dq

    K = ca.diag(k)

    tau_forward = (
        ca.mtimes(M, qddot) + ca.mtimes(F, qdot) + ca.mtimes(K, qdot) - f
    )

    # equation of motion
    # M * q_ddot + F q_dot - f = tau
    # augmented for first order system
    tau_aug = ca.vertcat(np.zeros((n, 1)), tau)
    f_aug = ca.vertcat(np.zeros((n, 1)), f)
    F_aug = ca.horzcat(np.zeros((2 * n, n)), ca.vertcat(-1 * np.identity(n), F))
    M_aug = ca.vertcat(
        ca.horzcat(np.identity(n), np.zeros((n, n))),
        ca.horzcat(np.zeros((n, n)), M),
    )
    """
    print('---')
    print(f_aug)
    print('---')
    print(F_aug)
    print('---')
    print(M_aug)
    print('---')
    """

    x = ca.vertcat(q, qdot)

    rhs = tau_aug + f_aug - ca.mtimes(F_aug, x)
    xdot = ca.solve(M_aug, rhs)

    """
    test_eval =fk(q_ex, qdot_ex, axis_jex, off_jex, off_eeex)
    print(np.transpose(np.round(test_eval)))
    print(test_eval)

    # test system for two link
    test_eval = dynamics(q_ex, qdot_ex, com_ex, m_ex, I_jex, g_ex, axis_jex, off_jex, tau_ex)
    print(test_eval)

    """
    M_fun = ca.Function(
        "M", [q, qdot, com, m, I_j, g, axis_j, off_j], [M, T, V, r_0ic]
    )
    dynamics = ca.Function(
        "dynamics", [q, qdot, com, m, I_j, g, axis_j, off_j, k, tau], [xdot]
    )
    fk = ca.Function("fk", [q, qdot, axis_j, off_j, off_ee], [fk])
    tau_fun = ca.Function(
        "tau", [q, qdot, com, m, I_j, g, axis_j, off_j, k, qddot], [tau_forward]
    )
    return dynamics, fk, tau_fun, M_fun


def main():
    if len(sys.argv) == 1:
        n = 1
    else:
        n = int(sys.argv[1])
    print("Creating dynamics for urdf with " + str(n) + " joints")
    dynamics, fk, _, _ = create3DDynamics(n)
    # save system
    dynamics_name = "./dynamics_" + str(n) + "_link.casadi"
    fk_name = "./fk_" + str(n) + "_link.casadi"
    dynamics.save(dynamics_name)
    fk.save(fk_name)


if __name__ == "__main__":
    main()
