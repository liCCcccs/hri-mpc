from libc.math cimport sin, cos

# Define a struct to hold the parameters
cdef struct Parameters:
    double m1
    double m2
    double m3
    double m4
    double g
    double I_G1z
    double I_G2z
    double I_G3z
    double I_G4z
    double l1
    double l2
    double l3
    double l4
    double lc1
    double la1
    double lb1
    double lc2
    double lc3
    double lc4
    double la4
    double la2
    double K_AFz
    double K_AFx
    double K_AMy
    double K_BFz
    double K_BFx
    double K_BMy


# Create an instance of the Parameters struct
cdef Parameters params

def func(Parameters params, double h_q2, double r_d2, double r_d3, double r_q4, r_q5, double h_dq2, double r_dd2, double r_dd3, double r_dq4, double r_dq5, double tau1, double tau2, double tau3, double tau4):

    m1 = params.m1
    m2 = params.m2
    m3 = params.m3
    m4 = params.m4
    g = params.g

    I_G1z = params.I_G1z
    I_G2z = params.I_G2z
    I_G3z = params.I_G3z
    I_G4z = params.I_G4z

    l1 = params.l1
    l2 = params.l2
    l3 = params.l3
    l4 = params.l4

    lc1 = params.lc1
    la1 = params.la1
    lb1 = params.lb1
    lc2 = params.lc2

    lc3 = params.lc3
    lc4 = params.lc4
    la4 = params.la4
    la2 = params.la2

    K_AFz = params.K_AFz
    K_AFx = params.K_AFx
    K_AMy = params.K_AMy

    K_BFz = params.K_BFz
    K_BFx = params.K_BFx
    K_BMy = params.K_BMy

    s_h_q2 = sin(h_q2)
    s_r_d2 = sin(r_d2)
    s_r_d3 = sin(r_d3)
    s_r_q4 = sin(r_q4)
    s_r_q5 = sin(r_q5)

    c_h_q2 = cos(h_q2)
    c_r_d2 = cos(r_d2)
    c_r_d3 = cos(r_d3)
    c_r_q4 = cos(r_q4)
    c_r_q5 = cos(r_q5)

    s_h_dq2 = sin(h_dq2)
    s_r_dd2 = sin(r_dd2)
    s_r_dd3 = sin(r_dd3)
    s_r_dq4 = sin(r_dq4)
    s_r_dq5 = sin(r_dq5)

    c_h_dq2 = cos(h_dq2)
    c_r_dd2 = cos(r_dd2)
    c_r_dd3 = cos(r_dd3)
    c_r_dq4 = cos(r_dq4)
    c_r_dq5 = cos(r_dq5)

    s2_h_q2 = sin(2*h_q2)
    s2_r_d2 = sin(2*r_d2)
    s2_r_d3 = sin(2*r_d3)
    s2_r_q4 = sin(2*r_q4)
    s2_r_q5 = sin(2*r_q5)

    c2_h_q2 = cos(2*h_q2)
    c2_r_d2 = cos(2*r_d2)
    c2_r_d3 = cos(2*r_d3)
    c2_r_q4 = cos(2*r_q4)
    c2_r_q5 = cos(2*r_q5)

    h_ddq2 = -(K_BMy*h_q2 - tau2 - K_BMy*r_q4 - K_BMy*r_q5 - K_BFz*la4*lc2*sin(r_q4 - h_q2 + r_q5) + K_BFz*lc2*r_d2*c_h_q2 - K_BFz*l1*lc2*s_h_q2 + K_BFz*la1*lc2*s_h_q2 + K_BFz*lc2*r_d3*s_h_q2 + g*lc2*m2*s_h_q2 + K_BFz*l3*lc2*sin(h_q2 - r_q4))/(m2*lc2**2 + I_G2z)

    return h_ddq2