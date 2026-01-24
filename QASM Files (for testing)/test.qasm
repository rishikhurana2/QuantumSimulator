OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
x q[0];
h q[1];
x q[2];
h q[2];
x q[1];
h q[2];
cx q[1],q[2];
t q[0];
tdg q[1];

