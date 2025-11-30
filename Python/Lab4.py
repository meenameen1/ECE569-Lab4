import matplotlib.pyplot as plt
import numpy as np
import math

'''
*** BASIC HELPER FUNCTIONS ***
'''

def ECE569_NearZero(z):
    """Determines whether a scalar is small enough to be treated as zero"""
    return abs(z) < 1e-6

def ECE569_Normalize(V):
    """ECE569_Normalizes a vector"""
    return V / np.linalg.norm(V)

'''
*** CHAPTER 3: RIGID-BODY MOTIONS ***
'''

def ECE569_RotInv(R):
    """Inverts a rotation matrix"""
    return np.array(R).T

def ECE569_VecToso3(omg):
    """Converts a 3-vector to an so(3) representation"""
    return np.array([[0,      -omg[2],  omg[1]],
                     [omg[2],       0, -omg[0]],
                     [-omg[1], omg[0],       0]])

def ECE569_so3ToVec(so3mat):
    """Converts an so(3) representation to a 3-vector"""
    return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])

def ECE569_AxisAng3(expc3):
    """Converts a 3-vector of exponential coordinates for rotation into axis-angle form"""
    return (ECE569_Normalize(expc3), np.linalg.norm(expc3))

def ECE569_MatrixExp3(so3mat):
    """Computes the matrix exponential of a matrix in so(3)"""
    omgtheta = ECE569_so3ToVec(so3mat)
    if ECE569_NearZero(np.linalg.norm(omgtheta)):
        return np.eye(3)
    else:
        theta = ECE569_AxisAng3(omgtheta)[1]
        omgmat = so3mat / theta
        return np.eye(3) + np.sin(theta) * omgmat \
               + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)

def ECE569_MatrixLog3(R):
    """Computes the matrix logarithm of a rotation matrix"""
    acosinput = (np.trace(R) - 1) / 2.0
    if acosinput >= 1:
        return np.zeros((3, 3))
    elif acosinput <= -1:
        if not ECE569_NearZero(1 + R[2][2]):
            omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
                  * np.array([R[0][2], R[1][2], 1 + R[2][2]])
        elif not ECE569_NearZero(1 + R[1][1]):
            omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) \
                  * np.array([R[0][1], 1 + R[1][1], R[2][1]])
        else:
            omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) \
                  * np.array([1 + R[0][0], R[1][0], R[2][0]])
        return ECE569_VecToso3(np.pi * omg)
    else:
        theta = np.arccos(acosinput)
        return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)

def ECE569_RpToTrans(R, p):
    """Converts a rotation matrix and a position vector into homogeneous transformation matrix"""
    return np.r_[np.c_[R, p], [[0, 0, 0, 1]]]

def ECE569_TransToRp(T):
    """Converts a homogeneous transformation matrix into a rotation matrix and position vector"""
    T = np.array(T)
    return T[0: 3, 0: 3], T[0: 3, 3]

def ECE569_TransInv(T):
    """Inverts a homogeneous transformation matrix"""
    R, p = ECE569_TransToRp(T)
    Rt = np.array(R).T
    # T^-1 = [R^T, -R^T*p; 0, 1]
    return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]

def ECE569_VecTose3(V):
    """Converts a spatial velocity vector into a 4x4 matrix in se3"""
    # V = [omega, v]
    omg = V[0:3]
    v = V[3:6]
    return np.r_[np.c_[ECE569_VecToso3(omg), v], [[0, 0, 0, 0]]]

def ECE569_se3ToVec(se3mat):
    """Converts an se3 matrix into a spatial velocity vector"""
    return np.r_[ECE569_so3ToVec(se3mat[0: 3, 0: 3]), se3mat[0: 3, 3]]

def ECE569_Adjoint(T):
    """Computes the adjoint representation of a homogeneous transformation matrix"""
    R, p = ECE569_TransToRp(T)
    p_so3 = ECE569_VecToso3(p)
    return np.r_[np.c_[R, np.zeros((3, 3))],
                 np.c_[np.dot(p_so3, R), R]]

def ECE569_MatrixExp6(se3mat):
    """Computes the matrix exponential of an se3 representation"""
    se3mat = np.array(se3mat)
    omgtheta = ECE569_so3ToVec(se3mat[0: 3, 0: 3])

    if ECE569_NearZero(np.linalg.norm(omgtheta)):
        # Case: pure translation
        return np.r_[np.c_[np.eye(3), se3mat[0: 3, 3]], [[0, 0, 0, 1]]]
    else:
        # Case: rotation and translation
        theta = ECE569_AxisAng3(omgtheta)[1]
        omgmat = se3mat[0: 3, 0: 3] / theta
        R = ECE569_MatrixExp3(se3mat[0: 3, 0: 3])
        p = np.dot(np.eye(3) * theta + (1 - np.cos(theta)) * omgmat +
                   (theta - np.sin(theta)) * np.dot(omgmat, omgmat),
                   se3mat[0: 3, 3] / theta)
        return np.r_[np.c_[R, p], [[0, 0, 0, 1]]]

def ECE569_MatrixLog6(T):
    """Computes the matrix logarithm of a homogeneous transformation matrix"""
    R, p = ECE569_TransToRp(T)
    omgmat = ECE569_MatrixLog3(R)

    if np.array_equal(omgmat, np.zeros((3, 3))):
        res = np.zeros((4, 4))
        res[0:3, 3] = p
        return res
    else:
        theta = np.arccos((np.trace(R) - 1) / 2.0)
        eye3 = np.eye(3)
        # 1/tan(x) is cot(x)
        coef = (1.0 / theta**2) * (1.0 - (theta / 2.0) * (1.0 / np.tan(theta / 2.0)))
        G_inv = eye3 - 0.5 * omgmat + coef * np.dot(omgmat, omgmat)
        v_theta = np.dot(G_inv, p)
        res = np.zeros((4, 4))
        res[0:3, 0:3] = omgmat
        res[0:3, 3] = v_theta
        return res

'''
*** CHAPTER 4: FORWARD KINEMATICS ***
'''

def ECE569_FKinBody(M, Blist, thetalist):
    """Computes forward kinematics in the body frame"""
    T = np.array(M)
    for i in range(len(thetalist)):
        # T = M * e^[B1]theta1 * ... * e^[Bn]thetan
        T = np.dot(T, ECE569_MatrixExp6(ECE569_VecTose3(Blist[:, i] * thetalist[i])))
    return T

def ECE569_FKinSpace(M, Slist, thetalist):
    """Computes forward kinematics in the space frame"""
    T = np.array(M)
    # T = e^[S1]theta1 * ... * e^[Sn]thetan * M
    # Iterate backwards to build the product from right to left or iterate forward and post-multiply
    # Standard forward iteration: R = e^S1 * e^S2 ... * M
    T_chain = np.eye(4)
    for i in range(len(thetalist)):
        T_chain = np.dot(T_chain, ECE569_MatrixExp6(ECE569_VecTose3(Slist[:, i] * thetalist[i])))

    return np.dot(T_chain, T)

'''
*** CHAPTER 5: VELOCITY KINEMATICS AND STATICS***
'''

def ECE569_JacobianBody(Blist, thetalist):
    """Computes the body Jacobian for an open chain robot"""
    Jb = np.array(Blist).copy().astype(float)
    T = np.eye(4)
    # J_n is just B_n
    # J_i is Ad_{e^{-B_n theta_n}...e^{-B_{i+1} theta_{i+1}}} (B_i)
    for i in range(len(thetalist) - 2, -1, -1):
        # Update transform T by multiplying e^{-B_{i+1} * theta_{i+1}}
        T = np.dot(T, ECE569_MatrixExp6(ECE569_VecTose3(-1 * Blist[:, i+1] * thetalist[i+1])))
        Jb[:, i] = np.dot(ECE569_Adjoint(T), Blist[:, i])
    return Jb

'''
*** CHAPTER 6: INVERSE KINEMATICS ***
'''

def ECE569_IKinBody(Blist, M, T, thetalist0, eomg, ev):
    """Computes inverse kinematics in the body frame"""
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20

    # Calculate current Tsb
    Tsb = ECE569_FKinBody(M, Blist, thetalist)
    # Calculate error twist Vb = Log(Tsb^-1 * Tsd)
    Vb = ECE569_se3ToVec(ECE569_MatrixLog6(np.dot(ECE569_TransInv(Tsb), T)))

    err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg \
          or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev

    while err and i < maxiterations:
        # Calculate Body Jacobian
        Jb = ECE569_JacobianBody(Blist, thetalist)
        # Update thetalist using pseudoinverse
        thetalist = thetalist + np.dot(np.linalg.pinv(Jb), Vb)

        i += 1
        # Recalculate error
        Tsb = ECE569_FKinBody(M, Blist, thetalist)
        Vb = ECE569_se3ToVec(ECE569_MatrixLog6(np.dot(ECE569_TransInv(Tsb), T)))
        err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg \
              or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev

    return (thetalist, not err)

# the ECE569_normalized trapezoid function
def g(t, T, ta):
    if t < 0 or t > T:
        return 0
    if t < ta:
        return (T/(T-ta)) * t/ta
    elif t > T - ta:
        return (T/(T-ta)) * (T - t)/ta
    else:
        return (T/(T-ta))

def trapezoid(t, T, ta):
    return g(t, T, ta)

def main():

    ### Step 1: Trajectory Generation

    A = 0.11
    B = 0.11
    a = 5.0
    b = 4.0

    T_param = 2*np.pi
    t_smooth = np.linspace(0, T_param, 500)
    xd = A*np.sin(a*t_smooth)
    yd = B*np.sin(b*t_smooth)

    # calculate the arc length
    d = 0
    for i in range(1, len(t_smooth)):
        d += np.sqrt((xd[i] - xd[i-1])**2 + (yd[i] - yd[i-1])**2)

    tfinal = 15.0
    ta = 1.0
    # calculate average velocity
    c = d/tfinal
    print(f"Arc Length: {d}, Average Velocity: {c}")

    # forward euler to calculate alpha
    dt = 0.002
    t = np.arange(0, tfinal, dt)
    alpha = np.zeros(len(t))

    for i in range(1, len(t)):
        t_prev = t[i-1]
        alpha_prev = alpha[i-1]
        dx_dalpha = A * a * np.cos(a * alpha_prev)
        dy_dalpha = B * b * np.cos(b * alpha_prev)

        deriv = np.sqrt(dx_dalpha**2 + dy_dalpha**2)
        if deriv < 1e-6: deriv = 1e-6

        gt = trapezoid(t_prev, tfinal, ta)
        alpha_dot = (c * gt) / deriv

        # Euler update
        alpha[i] = alpha[i-1] + alpha_dot * dt

    # plot alpha vs t
    plt.figure()
    plt.plot(t, alpha,'b-',label='alpha')
    plt.plot(t, np.ones(len(t))*T_param, 'k--',label='T (period)')
    plt.xlabel('t')
    plt.ylabel('alpha')
    plt.title('alpha vs t')
    plt.legend()
    plt.grid()
    plt.show()

    # rescale our trajectory with alpha
    x = A*np.sin(a*alpha)
    y = B*np.sin(b*alpha)

    # calculate velocity
    xdot = np.diff(x)/dt
    ydot = np.diff(y)/dt
    v = np.sqrt(xdot**2 + ydot**2)

    # plot velocity vs t
    plt.figure()
    plt.plot(t[1:], v, 'b-',label='velocity')
    plt.plot(t[1:], np.ones(len(t[1:]))*c, 'k--',label='average velocity')
    plt.plot(t[1:], np.ones(len(t[1:]))*0.25, 'r--',label='velocity limit')
    plt.xlabel('t')
    plt.ylabel('velocity')
    plt.title('velocity vs t')
    plt.legend()
    plt.grid()
    plt.show()

    ### Step 2: Forward Kinematics
    L1 = 0.2435
    L2 = 0.2132
    W1 = 0.1311
    W2 = 0.0921
    H1 = 0.1519
    H2 = 0.0854

    M = np.array([[-1, 0, 0, L1 + L2],
                  [0, 0, 1, W1 + W2],
                  [0, 1, 0, H1 - H2],
                  [0, 0, 0, 1]])

    S1 = np.array([0, 0, 1, 0, 0, 0])
    S2 = np.array([0, 1, 0, -H1, 0, 0])
    S3 = np.array([0, 1, 0, -H1, 0, L1])
    S4 = np.array([0, 1, 0, -H1, 0, L1 + L2])
    S5 = np.array([0, 0, -1, -W1, L1+L2, 0])
    S6 = np.array([0, 1, 0, H2-H1, 0, L1+L2])
    S = np.array([S1, S2, S3, S4, S5, S6]).T

    B1 = np.linalg.inv(ECE569_Adjoint(M))@S1
    B2 = np.linalg.inv(ECE569_Adjoint(M))@S2
    B3 = np.linalg.inv(ECE569_Adjoint(M))@S3
    B4 = np.linalg.inv(ECE569_Adjoint(M))@S4
    B5 = np.linalg.inv(ECE569_Adjoint(M))@S5
    B6 = np.linalg.inv(ECE569_Adjoint(M))@S6
    B = np.array([B1, B2, B3, B4, B5, B6]).T

    theta0 = np.array([-1.6800, -1.4018, -1.8127, -2.9937, -0.8857, -0.0696])

    # perform forward kinematics
    T0_space = ECE569_FKinSpace(M, S, theta0)
    print(f'T0_space:\n{T0_space}')
    T0_body = ECE569_FKinBody(M, B, theta0)
    print(f'T0_body:\n{T0_body}')
    T0_diff = T0_space - T0_body
    print(f'T0_diff:\n{T0_diff}')

    # Set T0 to the calculated home position (Tsb(0))
    T0 = T0_space

    Tsd = np.zeros((4, 4, len(t)))
    for i in range(len(t)):
        Td = np.eye(4)
        Td[0, 3] = x[i]
        Td[1, 3] = y[i]
        Td[2, 3] = 0.0

        Tsd[:, :, i] = np.dot(T0, Td)

    xs = Tsd[0, 3, :]
    ys = Tsd[1, 3, :]
    zs = Tsd[2, 3, :]
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(xs, ys, zs, 'b-',label='p(t)')
    ax.plot(xs[0], ys[0], zs[0], 'go',label='start')
    ax.plot(xs[-1], ys[-1], zs[-1], 'rx',label='end')
    ax.set_title('Task 2d: Trajectory in s frame')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.legend()
    plt.show()

    ### Step 3: Inverse Kinematics

    thetaAll = np.zeros((6, len(t)))

    initialguess = theta0
    eomg = 1e-4
    ev = 1e-4

    print("Solving Inverse Kinematics...")
    thetaSol, success = ECE569_IKinBody(B, M, Tsd[:,:,0], initialguess, eomg, ev)
    if not success:
        raise Exception(f'Failed to find a solution at index {0}')
    thetaAll[:, 0] = thetaSol

    for i in range(1, len(t)):

        initialguess = thetaAll[:, i-1]
        thetaSol, success = ECE569_IKinBody(B, M, Tsd[:,:,i], initialguess, eomg, ev)

        if not success:
            print(f"Warning: Failed to converge at step {i}")

            thetaSol = thetaAll[:, i-1]

        thetaAll[:, i] = thetaSol

    dj = np.diff(thetaAll, axis=1)
    plt.figure()
    plt.plot(t[1:], dj[0], 'b-',label='joint 1')
    plt.plot(t[1:], dj[1], 'g-',label='joint 2')
    plt.plot(t[1:], dj[2], 'r-',label='joint 3')
    plt.plot(t[1:], dj[3], 'c-',label='joint 4')
    plt.plot(t[1:], dj[4], 'm-',label='joint 5')
    plt.plot(t[1:], dj[5], 'y-',label='joint 6')
    plt.xlabel('t (seconds)')
    plt.ylabel('first order difference')
    plt.title('Task 3.3: Joint angles first order difference')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.show()

    actual_Tsd = np.zeros((4, 4, len(t)))
    for i in range(len(t)):
        actual_Tsd[:,:,i] = ECE569_FKinBody(M, B, thetaAll[:, i])

    xs = actual_Tsd[0, 3, :]
    ys = actual_Tsd[1, 3, :]
    zs = actual_Tsd[2, 3, :]
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(xs, ys, zs, 'b-',label='p(t)')
    ax.plot(xs[0], ys[0], zs[0], 'go',label='start')
    ax.plot(xs[-1], ys[-1], zs[-1], 'rx',label='end')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title('Task 3.4: Verified Trajectory in s frame')
    ax.legend()
    plt.show()

    body_dets = np.zeros(len(t))
    for i in range(len(t)):
        Jb = ECE569_JacobianBody(B, thetaAll[:, i])
        body_dets[i] = np.linalg.det(Jb)

    plt.figure()
    plt.plot(t, body_dets, '-')
    plt.xlabel('t (seconds)')
    plt.ylabel('det of J_B')
    plt.title('Task 3.5: Manipulability (Det(Jb))')
    plt.grid()
    plt.tight_layout()
    plt.show()

    led = np.ones_like(t)
    data = np.column_stack((t, thetaAll.T, led))

    filename = 'aabubakr.csv'
    np.savetxt(filename, data, delimiter=',')
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    main()
