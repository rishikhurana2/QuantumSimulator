You can run the simulator on the file using 

python3 Simulator.py {QASM FILENAME}

Where QASM FILENAME is any .qasm file that you wish to run the simulator on. Note that the accepted QASM files can only contain Hadamard gates, NOT gates, Controlled NOT gates, T gates, 
and the conjugate transpose of T gates. An example format with a few of the accepted gates is below:

OPENQASM 2.0;  
include "qelib1.inc";  
qreg q[16];  
creg c[16];  
cx q[12],q[15];  
h q[12];  
t q[14];  
t q[13];  
t q[12];  
cx q[13],q[14];  
cx q[12],q[13];  
cx q[14],q[12];  
tdg q[13];  
cx q[14],q[13];  
tdg q[14];  
tdg q[13];  
