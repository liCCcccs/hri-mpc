from libc.math cimport sin, cos, pi

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
    double D_AFx
    double D_AFz
    double D_AMy
    double D_BFz
    double D_BFx
    double D_BMy


# Create an instance of the Parameters struct
cdef Parameters params

def func(Parameters params, double q1, double dq1, double h_q2, double h_dq2, double r_d2, double r_dd2, double r_d3, double r_dd3, double r_q4, double r_dq4, double r_q5, double r_dq5, double tau1, double tau2, double tau3, double tau4):

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

    D_AFz = params.D_AFz
    D_AFx = params.D_AFx
    D_AMy = params.D_AMy
    D_BFz = params.D_BFz
    D_BFx = params.D_BFx
    D_BMy = params.D_BMy

    f_intBi = D_BFx*(r_dd2*cos(h_q2) + r_dd3*cos(h_q2) - h_dq2*l1*sin(h_q2) + r_d3*r_dq4*cos(h_q2) + r_d3*r_dq5*cos(h_q2) + la1*r_dq4*sin(h_q2) + la1*r_dq5*sin(h_q2) + r_d2*r_dq4*sin(h_q2) + r_d2*r_dq5*sin(h_q2) + l3*r_dq5*sin(h_q2 - r_q4)) - K_BFx*((la2*(cos(h_q2)*cos(q1 - pi/2) - sin(h_q2)*sin(q1 - pi/2)) + l1*cos(q1 - pi/2))*(cos(h_q2)*cos(q1 - pi/2) - sin(h_q2)*sin(q1 - pi/2)) + (la2*(cos(h_q2)*sin(q1 - pi/2) + sin(h_q2)*cos(q1 - pi/2)) + l1*sin(q1 - pi/2))*(cos(h_q2)*sin(q1 - pi/2) + sin(h_q2)*cos(q1 - pi/2)) + (cos(h_q2)*sin(q1 - pi/2) + sin(h_q2)*cos(q1 - pi/2))*(l3*(cos(q1)*cos(r_q4) - sin(q1)*sin(r_q4)) + cos(q1)*(la1 + r_d2) + r_d3*sin(q1) + la4*(cos(r_q5)*(cos(q1)*cos(r_q4) - sin(q1)*sin(r_q4)) - sin(r_q5)*(cos(q1)*sin(r_q4) + cos(r_q4)*sin(q1)))) - (cos(h_q2)*cos(q1 - pi/2) - sin(h_q2)*sin(q1 - pi/2))*(l3*(cos(q1)*sin(r_q4) + cos(r_q4)*sin(q1)) + sin(q1)*(la1 + r_d2) - r_d3*cos(q1) + la4*(cos(r_q5)*(cos(q1)*sin(r_q4) + cos(r_q4)*sin(q1)) + sin(r_q5)*(cos(q1)*cos(r_q4) - sin(q1)*sin(r_q4)))));
    f_intBj = - D_BFz*(r_dd2*sin(h_q2) + r_dd3*sin(h_q2) - la1*r_dq4*cos(h_q2) - la1*r_dq5*cos(h_q2) - r_d2*r_dq4*cos(h_q2) - r_d2*r_dq5*cos(h_q2) + r_d3*r_dq4*sin(h_q2) + r_d3*r_dq5*sin(h_q2) - l3*r_dq5*cos(h_q2 - r_q4) + h_dq2*l1*cos(h_q2)) - K_BFz*((la2*(cos(h_q2)*sin(q1 - pi/2) + sin(h_q2)*cos(q1 - pi/2)) + l1*sin(q1 - pi/2))*(cos(h_q2)*cos(q1 - pi/2) - sin(h_q2)*sin(q1 - pi/2)) - (la2*(cos(h_q2)*cos(q1 - pi/2) - sin(h_q2)*sin(q1 - pi/2)) + l1*cos(q1 - pi/2))*(cos(h_q2)*sin(q1 - pi/2) + sin(h_q2)*cos(q1 - pi/2)) + (cos(h_q2)*sin(q1 - pi/2) + sin(h_q2)*cos(q1 - pi/2))*(l3*(cos(q1)*sin(r_q4) + cos(r_q4)*sin(q1)) + sin(q1)*(la1 + r_d2) - r_d3*cos(q1) + la4*(cos(r_q5)*(cos(q1)*sin(r_q4) + cos(r_q4)*sin(q1)) + sin(r_q5)*(cos(q1)*cos(r_q4) - sin(q1)*sin(r_q4)))) + (cos(h_q2)*cos(q1 - pi/2) - sin(h_q2)*sin(q1 - pi/2))*(l3*(cos(q1)*cos(r_q4) - sin(q1)*sin(r_q4)) + cos(q1)*(la1 + r_d2) + r_d3*sin(q1) + la4*(cos(r_q5)*(cos(q1)*cos(r_q4) - sin(q1)*sin(r_q4)) - sin(r_q5)*(cos(q1)*sin(r_q4) + cos(r_q4)*sin(q1)))));
    m4gi = m4*g*cos(q1+r_q4+r_q5);
    m4gj = -m4*g*sin(q1+r_q4+r_q5);
    tau_intBz  = K_BMy*(r_q4 + r_q5 - h_q2) + D_BMy*(r_dq4 + r_dq5 - h_dq2);
    f_intAi = K_AFx * r_d2 + D_AFx * r_dd2;
    f_intAj = - K_AFz * r_d3 - D_AFz * r_dd3;
    m3gi  = m3*g*cos(q1+r_q4);
    m3gj  = -m3*g*sin(q1+r_q4);
    tau_intAz  = K_AMy*r_q4 + D_AMy*r_dq4;
    m2gj  = -m2*g*sin(q1+h_q2);

    a1 = - tau_intBz - lc2*(f_intBj + m2gj);
    a2 = m2*lc2**2 + I_G2z;
    b1 = l3*lc4*m4*sin(r_q5)*r_dq4**2 - l3*la4*m4*sin(r_q5)*r_dq5**2 + tau_intAz + tau_intBz + f_intAj*lc3 + f_intBj*lc4 - lc3*m3gj - lc4*m4gj + f_intBj*l3*cos(r_q5) - l3*m4gj*cos(r_q5) + f_intBi*l3*sin(r_q5) - l3*m4gi*sin(r_q5);
    b2 = - (l3*m4*sin(r_q4) + lc3*m3*sin(r_q4) + lc4*m4*sin(r_q4 + r_q5));
    b3 = - (l3*m4*cos(r_q4) + lc3*m3*cos(r_q4) + lc4*m4*cos(r_q4 + r_q5));
    b4 = m4*l3**2 + lc4*m4*cos(r_q5)*l3 + m3*lc3**2 + 2*I_G4z;
    b5 = I_G4z + la4*lc4*m4 + l3*la4*m4*cos(r_q5);
    c1 = l3*lc4*m4*sin(r_q5)*r_dq4**2 + tau_intBz + f_intBj*lc4 - lc4*m4gj;
    c2 = - lc4*m4*sin(r_q4 + r_q5);
    c3 = - lc4*m4*cos(r_q4 + r_q5);
    c4 = I_G4z + l3*lc4*m4*cos(r_q5);
    c5 = I_G4z + la4*lc4*m4;
    d1 =  + f_intAi - m3gi + f_intBi*cos(r_q5) - m4gi*cos(r_q5) - f_intBj*sin(r_q5) + m4gj*sin(r_q5) - l3*m4*r_dq4**2 - lc3*m3*r_dq4**2 - la4*m4*r_dq5**2*cos(r_q5);
    d2 = m3*cos(r_q4) + m4*cos(r_q4);
    d3 = - m3*sin(r_q4) - m4*sin(r_q4);
    d5 = -la4*m4*sin(r_q5);
    e1 = - la4*m4*sin(r_q5)*r_dq5**2 + f_intAj - m3gj + f_intBj*cos(r_q5) - m4gj*cos(r_q5) + f_intBi*sin(r_q5) - m4gi*sin(r_q5);
    e2 = -(m3*sin(r_q4) + m4*sin(r_q4));
    e3 = -(m3*cos(r_q4) + m4*cos(r_q4));
    e4 = l3*m4 + lc3*m3;
    e5 = la4*m4*cos(r_q5);

    ddq1 = 0;
    h_ddq2 = -(a1 - tau2)/a2;
    r_ddd2 = -(b1*c3*d5*e4 + b1*c4*d3*e5 - b1*c4*d5*e3 - b1*c5*d3*e4 - b3*c1*d5*e4 - b3*c4*d1*e5 + b3*c4*d5*e1 + b3*c5*d1*e4 - b4*c1*d3*e5 + b4*c1*d5*e3 + b4*c3*d1*e5 - b4*c3*d5*e1 - b4*c5*d1*e3 + b4*c5*d3*e1 + b5*c1*d3*e4 - b5*c3*d1*e4 + b5*c4*d1*e3 - b5*c4*d3*e1 + b3*d5*e4*tau4 + b4*d3*e5*tau4 - b4*d5*e3*tau4 - b5*d3*e4*tau4)/(b2*c3*d5*e4 + b2*c4*d3*e5 - b2*c4*d5*e3 - b2*c5*d3*e4 - b3*c2*d5*e4 - b3*c4*d2*e5 + b3*c4*d5*e2 + b3*c5*d2*e4 - b4*c2*d3*e5 + b4*c2*d5*e3 + b4*c3*d2*e5 - b4*c3*d5*e2 - b4*c5*d2*e3 + b4*c5*d3*e2 + b5*c2*d3*e4 - b5*c3*d2*e4 + b5*c4*d2*e3 - b5*c4*d3*e2);
    r_ddd3 = (b1*c2*d5*e4 + b1*c4*d2*e5 - b1*c4*d5*e2 - b1*c5*d2*e4 - b2*c1*d5*e4 - b2*c4*d1*e5 + b2*c4*d5*e1 + b2*c5*d1*e4 - b4*c1*d2*e5 + b4*c1*d5*e2 + b4*c2*d1*e5 - b4*c2*d5*e1 - b4*c5*d1*e2 + b4*c5*d2*e1 + b5*c1*d2*e4 - b5*c2*d1*e4 + b5*c4*d1*e2 - b5*c4*d2*e1 + b2*d5*e4*tau4 + b4*d2*e5*tau4 - b4*d5*e2*tau4 - b5*d2*e4*tau4)/(b2*c3*d5*e4 + b2*c4*d3*e5 - b2*c4*d5*e3 - b2*c5*d3*e4 - b3*c2*d5*e4 - b3*c4*d2*e5 + b3*c4*d5*e2 + b3*c5*d2*e4 - b4*c2*d3*e5 + b4*c2*d5*e3 + b4*c3*d2*e5 - b4*c3*d5*e2 - b4*c5*d2*e3 + b4*c5*d3*e2 + b5*c2*d3*e4 - b5*c3*d2*e4 + b5*c4*d2*e3 - b5*c4*d3*e2);
    r_ddq4 = (b1*c2*d3*e5 - b1*c2*d5*e3 - b1*c3*d2*e5 + b1*c3*d5*e2 + b1*c5*d2*e3 - b1*c5*d3*e2 - b2*c1*d3*e5 + b2*c1*d5*e3 + b2*c3*d1*e5 - b2*c3*d5*e1 - b2*c5*d1*e3 + b2*c5*d3*e1 + b3*c1*d2*e5 - b3*c1*d5*e2 - b3*c2*d1*e5 + b3*c2*d5*e1 + b3*c5*d1*e2 - b3*c5*d2*e1 - b5*c1*d2*e3 + b5*c1*d3*e2 + b5*c2*d1*e3 - b5*c2*d3*e1 - b5*c3*d1*e2 + b5*c3*d2*e1 + b2*d3*e5*tau4 - b2*d5*e3*tau4 - b3*d2*e5*tau4 + b3*d5*e2*tau4 + b5*d2*e3*tau4 - b5*d3*e2*tau4)/(b2*c3*d5*e4 + b2*c4*d3*e5 - b2*c4*d5*e3 - b2*c5*d3*e4 - b3*c2*d5*e4 - b3*c4*d2*e5 + b3*c4*d5*e2 + b3*c5*d2*e4 - b4*c2*d3*e5 + b4*c2*d5*e3 + b4*c3*d2*e5 - b4*c3*d5*e2 - b4*c5*d2*e3 + b4*c5*d3*e2 + b5*c2*d3*e4 - b5*c3*d2*e4 + b5*c4*d2*e3 - b5*c4*d3*e2);
    r_ddq5 = -(b1*c2*d3*e4 - b1*c3*d2*e4 + b1*c4*d2*e3 - b1*c4*d3*e2 - b2*c1*d3*e4 + b2*c3*d1*e4 - b2*c4*d1*e3 + b2*c4*d3*e1 + b3*c1*d2*e4 - b3*c2*d1*e4 + b3*c4*d1*e2 - b3*c4*d2*e1 - b4*c1*d2*e3 + b4*c1*d3*e2 + b4*c2*d1*e3 - b4*c2*d3*e1 - b4*c3*d1*e2 + b4*c3*d2*e1 + b2*d3*e4*tau4 - b3*d2*e4*tau4 + b4*d2*e3*tau4 - b4*d3*e2*tau4)/(b2*c3*d5*e4 + b2*c4*d3*e5 - b2*c4*d5*e3 - b2*c5*d3*e4 - b3*c2*d5*e4 - b3*c4*d2*e5 + b3*c4*d5*e2 + b3*c5*d2*e4 - b4*c2*d3*e5 + b4*c2*d5*e3 + b4*c3*d2*e5 - b4*c3*d5*e2 - b4*c5*d2*e3 + b4*c5*d3*e2 + b5*c2*d3*e4 - b5*c3*d2*e4 + b5*c4*d2*e3 - b5*c4*d3*e2);

    return dq1, ddq1, h_dq2, h_ddq2, r_dd2, r_ddd2, r_dd3, r_ddd3, r_dq4, r_ddq4, r_dq5, r_ddq5