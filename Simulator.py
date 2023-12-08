import math
import numpy as np
from numpy import linalg as LA
import time
import argparse
import sys

class Simulator:
    def __init__(self, qasm_file):
        #File name 
        self.qasm_file = qasm_file

        #matrix representation of the gates supported by the current language -- tuples are for complex number representation
        self.I = np.array([ [1 + 0j, 0 + 0j], 
                            [0 + 0j, 1 + 0j]  ])
        
        self.H = np.array([ [ 1/(math.sqrt(2)) + 0j, 1/(math.sqrt(2)) + 0j], 
                            [ 1/(math.sqrt(2)) + 0j, -1/(math.sqrt(2)) + 0j]  ])
        
        self.X = np.array([  [ 0 + 0j, 1 + 0j ], 
                             [ 1 + 0j, 0 + 0j]  ])
        
        self.CX_forward = np.array([  [1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j], 
                                      [0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j], 
                                      [0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j], 
                                      [0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j]  ])
        
        self.CX_backward = np.array([ [1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],                                     
                                      [0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j],
                                      [0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j],
                                      [0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j]  ])
        
        self.T = np.array([ [1 + 0j, 0 + 0j],
                            [0 + 0j, 1/(math.sqrt(2)) + 1j/(math.sqrt(2))]  ])
    
        self.Tdg = np.array([ [1 + 0j, 0 + 0j],
                              [0 + 0j, 1/(math.sqrt(2)) - 1j/(math.sqrt(2))]  ])
        #self.qubits_arr is an array whose index represents the state and whose coefficient represents the probability of measuring that state
        self.num_qubits = -1
        self.qubits_arr = []

    def find_effective_num_qubits(self):
        #Pre-processes the qasm file to find the effective number of qubits
        with open(self.qasm_file) as qasmLines:
            line_num = 1
            effective_num_qubits = -1
            for lines in qasmLines:
                if (line_num >= 5):
                    for c in lines:
                        try:
                            if (int(c) > effective_num_qubits):
                                effective_num_qubits = int(c)
                        except:
                            continue
                line_num += 1
        self.num_qubits = effective_num_qubits + 1
    
    def set_qubits(self):
        #Output: sets the qubits_arr and classical_bits_arr from the given number of qubits

        #initial state of the system is all zeros ==> |0> x |0> x |0> x |0> x .... x |0>  = (1, 0, 0, 0, ..., 0):
        self.qubits_arr.append(1 + 0j)
        for i in range(1, 2**(self.num_qubits)):
            self.qubits_arr.append(0 + 0j)   

    def process_line(self, line):
        grab_gate = True
        grab_registers = False
        registers = ""
        gate = ""
        for c in line:
            if (grab_registers):
                registers += c
            if (c == " "):
                grab_gate = False
                grab_registers = True
            if (grab_gate):
                gate += c
        registers = registers.split(",")
        regs_in_ints = []
        for regs in registers:
            for c in regs:
                try: #append the integer index of the register if we have encountered that character
                    regs_in_ints.append(int(c)) 
                except: #continue to the next character otherwise
                    continue
        return [gate, [regs_in_ints[i] for i in range(len(regs_in_ints))]]

    def parse_file(self):
        #Output: parses the file and applies the matrices for each line
        with open(self.qasm_file) as qasmLines:
            line_num = 1
            for line in qasmLines:
                if (line_num >= 5): #waiting to reach line 5 because that is when the instructions start                    
                    [gate, registers] = self.process_line(line)
                    if (len(registers) == 1):
                        unitary = self.str_to_gate(gate) #grab the single-qubit unitary specified by the string
                        self.apply_single_qubit_gate(unitary, registers[0]) #apply
                    elif(len(registers) > 1):
                        self.apply_CX(registers[0], registers[1]) #CX is the only two qubit gate
                line_num += 1
    
    def str_to_gate(self, gate_str):
        #Input: string representing a one-qubit gate
        #Output: The unitary represented by the string (returns nothing if not one of the below unitaries)
        if (gate_str == "x"):
            return self.X
        if (gate_str == "h"):
            return self.H
        if (gate_str == "t"):
            return self.T
        if (gate_str == "tdg"):
            return self.Tdg

    def tensor_product(self, A, B):
        #Input:  matrix A and matrix B (square matricies)
        #Output: tensor product of A and B
        tensor_product = np.zeros((len(A)*len(B), len(A[0])*len(B[0])), dtype=np.complex_)
        for i in range(len(A)):
            for j in range(len(A[0])):
                a = A[i][j]
                a_x_B = a*B
                for m in range(len(B)):
                    for n in range(len(B[0])):
                        tensor_product[i*len(B) + m][j*len(B[0]) + n] = a_x_B[m][n]
        return tensor_product
    

    def apply_single_qubit_gate(self, gate, reg):
        #Input: A single qubit gate and a qubit register
        #Output: The state of the system with the gate applied to the qubit
        GATE = [[1 + 0j]]
        for i in range(reg):
            GATE = self.tensor_product(GATE, self.I)
        GATE = self.tensor_product(GATE, gate)
        for i in range(reg + 1, self.num_qubits):
            GATE = self.tensor_product(GATE, self.I)
        self.qubits_arr = np.dot(GATE, self.qubits_arr)
    
    def SWAP(self, reg1, reg2):
        #Input: reg1 and reg2 that MUST be next to each other
        #Output: reg1 and reg2 swapped
        self.apply_next_too_each_other_CX(reg1, reg2)
        self.apply_next_too_each_other_CX(reg2, reg1)
        self.apply_next_too_each_other_CX(reg1, reg2)
    
    def apply_next_too_each_other_CX(self, reg1, reg2):
        #Input: reg1 and reg2 right next to each other
        #Output: CX applied to reg1 and reg2, where reg1 is the control bit and reg2 is the target bit
        if (reg1 < reg2):
            CNOT = [[1 + 0j]]
            for i in range(reg1):
                CNOT = self.tensor_product(CNOT, self.I)
            CNOT = self.tensor_product(CNOT, self.CX_forward)
            for i in range(reg2 + 1, self.num_qubits):
                CNOT = self.tensor_product(CNOT, self.I)
            self.qubits_arr = np.dot(CNOT, self.qubits_arr)
        if (reg2 < reg1):
            CNOT = [[1 + 0j]]
            for i in range(reg2):
                CNOT = self.tensor_product(CNOT, self.I)
            CNOT = self.tensor_product(CNOT, self.CX_backward)
            for i in range(reg1 + 1, self.num_qubits):
                CNOT = self.tensor_product(CNOT, self.I)
            self.qubits_arr = np.dot(CNOT, self.qubits_arr)            

    def apply_CX(self, reg1, reg2): 
        #generalized version of apply_next_too_each_other_CX
        #Input: reg1 as control bit, reg2 as target bit
        #Output: CNOT(reg1, reg2)
        r1 = reg1
        r2 = reg2
        swapped_R1 = False
        while (abs(r1 - r2) > 1):
            if (r1 < r2):
                self.SWAP(r1, r1 + 1)
                swapped_R1 = True
                r1 = r1 + 1
            if (r2 < r1):
                self.SWAP(r2, r2 + 1)
                r2 = r2 + 1
        if (abs(r1 - r2) == 1):
            self.apply_next_too_each_other_CX(r1, r2)
        
        while (abs(r1 - r2) != abs(reg1 - reg2)):
            if (swapped_R1):
                self.SWAP(r1 - 1, r1)
                r1 = r1 - 1
            else:
                self.SWAP(r2 - 1, r2)  
                r2 = r2 - 1      

    def run(self):
        #runs the simulator
        self.find_effective_num_qubits() #finds the number of qubits used by the program
        self.set_qubits() #sets the initial state of the qubits
        self.parse_file() #parses the file and applies gates to the qubits
        for i in range(len(self.qubits_arr)): #prints an output state and the probability of measuring it
            str_bin = bin(i)
            zeros = ""
            for j in range(self.num_qubits - (len(str_bin) - 2)):
                zeros += '0'
            print(str_bin[2:] + zeros + ":", abs(self.qubits_arr[i])**2) #probability is the norm of the coeefficient squared

def main():
    parser = argparse.ArgumentParser(prog='Quantum Simulator', description='Simulates a QASM file and outputs the state vector')
    parser.add_argument('QASM_filename')
    args = parser.parse_args()
    print("Testing", args.QASM_filename)
    start_time = time.time()
    s = Simulator(args.QASM_filename)
    s.run()
    print("Took", time.time() - start_time)

if __name__ == "__main__":
    main()