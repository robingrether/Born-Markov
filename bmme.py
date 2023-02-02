import numpy
import scipy
from scipy import linalg, special, integrate, sparse
from scipy.integrate import solve_ivp
import types
import gc
import time

#hbar = 0.65821195695091   # reduced Planck constant in units eV*fs
#_hbar = 1 / hbar          # inverse reduced Planck constant in units 1/(eV*fs)
#k_B = 8.6173332621452e-5  # Boltzmann constant in units eV/K

hbar = 1
_hbar = 1
k_B = 1
    

def Hc(A):
    return (A.T).conj()
    
def commute(A, B):
    return A @ B - B @ A

def anticommute(A, B):
    return A @ B + B @ A

# N: number of different fermionic operators
#
# resulting operators are (2**N, 2**N) matrices
def generate_fermionic_ops(N):
    jks = []
    bits = []
    for bit in range(2**N):
        jk = []
        for i in range(N):
            if bit & (2**i):
                jk.append(i)
        jks.append(numpy.array(jk, dtype=numpy.int32))
        bits.append(bit)
    
    d_ops = []
    d_dags = []
    for i in range(N):
        d_dag = numpy.zeros((2**N, 2**N))
        for jk, bit in zip(jks, bits):
            #print(jk, bit)
            if bit & (2**i):
                continue
            sigma, = numpy.where(jk > i)
            #print(sigma)
            d_dag[bit + 2**i, bit] = (-1)**len(sigma)
        d_ops.append(d_dag.T)
        d_dags.append(d_dag)
    
    return d_ops, d_dags
            

# N: number of different bosonic operators       
# M: number of states per bosonic subspace (can be a single integer or list of size N)    
#
# resulting operators are (N**M, N**M) matrices or (M[0]+M[1]+...+M[N-1], M[0]+M[1]+...+M[N-1]) matrices 
def generate_bosonic_ops(N, M):
    if not hasattr(M, "__getitem__"):
        M = numpy.ones(N, dtype=numpy.int32) * M
        
    if N > 1:
        a_ops = []
        for i in range(N):
            a_op = numpy.diag(numpy.sqrt(numpy.arange(1, M[i])), k=1)
            for j in range(i):
                a_ops[j] = numpy.kron(numpy.identity(M[i]), a_ops[j])
            if i > 0:
                a_op = numpy.kron(a_op, numpy.identity(numpy.sum(M[:i])))
            a_ops.append(a_op)

        a_dags = []
        for a_op in a_ops:
            a_dags.append(a_op.T)

        return a_ops, a_dags
    if N == 1:
        a_op = numpy.diag(numpy.sqrt(numpy.arange(1, M[0])), k=1)
        
        return [a_op], [a_op.T]


# Nf: number of different fermionic operators
# Nb: number of different bosonic operators
# Mb: number of states per bosonic subspace (can be a single integer or list of size Nb)   
#
# resulting matrices are kroneckered in order: fermionic x bosonic
def generate_ferm_bos_ops(Nf, Nb, Mb):
    d_ops, d_dags = generate_fermionic_ops(Nf)
    a_ops, a_dags = generate_bosonic_ops(Nb, Mb)
    
    nf = d_ops[0].shape[0]
    nb = a_ops[0].shape[0]
    
    for i in range(Nf):
        d_ops[i] = numpy.kron(d_ops[i], numpy.identity(nb))
        d_dags[i] = numpy.kron(d_dags[i], numpy.identity(nb))
        
    for i in range(Nb):
        a_ops[i] = numpy.kron(numpy.identity(nf), a_ops[i])
        a_dags[i] = numpy.kron(numpy.identity(nf), a_dags[i])
        
    return (d_ops, d_dags), (a_ops, a_dags)


# arr: square matrix/array to take partial trace of
# n: dimension of subspace that is to be traced over
# first: is the subspace to be traced over the first or second argument of the kronecker multiplication?
#
# calculates the partial trace of a square matrix; returned array has shape arr.shape // n
def partial_trace(arr, n, first=True):
    if first:
        m = arr.shape[0] // n
        pt = numpy.zeros((m,m), dtype=arr.dtype)
        for i in range(m):
            for j in range(m):
                pt[i,j] = numpy.sum(arr[i::m,j::m])
        return pt
    else:
        m = arr.shape[0] // n
        pt = numpy.zeros((m,m), dtype=arr.dtype)
        for i in range(n):
            for j in range(n):
                pt += arr[i::n,j::n]
        return pt


def create_sparse_matrix(arg1, dtype=numpy.complex128):
    return sparse.csr_array(arg1, dtype=dtype)


class BornMarkovSolver:

    def __init__(self, H_S, d_ops, a_ops, Gammas, chemical_potentials, temperatures, diagonalize=numpy.linalg.eig, include_digamma=True):
        number_of_leads = len(chemical_potentials)
        
        if len(Gammas.shape) == 2:
            Gammas = numpy.transpose(numpy.array([Gammas] * number_of_leads), axes=(1,2,0))
            
        number_of_fermions = len(d_ops)
        dimension = H_S.shape[0]
        dimension_of_fermions = number_of_fermions * 2
        dimension_of_bosons = dimension - dimension_of_fermions
        
        #time1 = time.time()
        
        # diagonalize system Hamiltonian
        w, V = diagonalize(H_S)
        # sort values for chopping
        # TODO
        W = numpy.diag(w)
        V_dag = Hc(V)
          
        
        #time2 = time.time()
        
        # create frequencies and subspace projected operators
        frequencies = numpy.tile(w,(dimension,1)).T - numpy.tile(w,(dimension,1))
        frequency = []
        d_op_omega = [[] for i in range(number_of_fermions)]
        #d_dag_omega = [[] for i in range(number_of_fermions)]
        
        first = True
        for n in range(dimension):
            for m in range(dimension):
                freq = frequencies[n,m]
                
                if not first:
                    diff = numpy.abs(numpy.array(frequency) - freq)
                    minarg = numpy.argmin(diff)
                
                projector_n = numpy.outer(V[:,n], Hc(V[:,n]))
                projector_m = numpy.outer(V[:,m], Hc(V[:,m]))
                
                if (not first) and (diff[minarg] < 1e-6):
                    for i in range(number_of_fermions):
                        d_op_omega[i][minarg] += create_sparse_matrix(projector_n @ V_dag @ d_ops[i] @ V @ projector_m)
                        #d_dag_omega[i][minarg] += projector_m @ V_dag @ Hc(d_ops[i]) @ V @ projector_n
                    
                else:
                    local_d_op_omega = []
                    any_nonzero = False
                    for i in range(number_of_fermions):
                        current_d_op_omega = create_sparse_matrix(projector_n @ V_dag @ d_ops[i] @ V @ projector_m)
                        if current_d_op_omega.count_nonzero():
                            any_nonzero = True
                        local_d_op_omega.append(current_d_op_omega)
                        
                    if any_nonzero:
                        frequency.append(freq)
                        first = False
                        
                        for i in range(number_of_fermions):
                            d_op_omega[i].append(local_d_op_omega[i])
                            #d_dag_omega[i].append(projector_m @ V_dag @ Hc(d_ops[i]) @ V @ projector_n)
                
        # somehow merge subspaces together -> create projected d_ops/d_dags
        
        #time3 = time.time()
        
        self.Gamma = Gammas
        self.number_of_leads = number_of_leads
        self.number_of_fermions = number_of_fermions
        self.dimension = dimension
        self.V = V
        self.V_dag = V_dag
        self.H_S = create_sparse_matrix(W)
        self.frequency = frequency
        self.d_op_omega = d_op_omega
        #self.d_dag_omega = d_dag_omega
        self.chemical_potential = chemical_potentials
        self.temperature = temperatures
        self.include_digamma = include_digamma
        
        #time4 = time.time()
        self.construct_liouvillian()
        #time5 = time.time()
        
        #print(f"Diagonalize: {(time2 - time1) * 1e6} µs")
        #print(f"Frequencies: {(time3 - time2) * 1e6} µs")
        #print(f"Liouvillian: {(time5 - time4) * 1e6} µs")
        
    def construct_liouvillian(self):
        #time1 = time.time()
        unity = sparse.identity(self.dimension, dtype=numpy.complex128)
        part1 = sparse.kron(unity, self.H_S) - sparse.kron(self.H_S.transpose(), unity)
        part2 = create_sparse_matrix((self.dimension**2, self.dimension**2))
        A = create_sparse_matrix((self.dimension, self.dimension))
        D = create_sparse_matrix((self.dimension, self.dimension))
        E = create_sparse_matrix((self.dimension, self.dimension))
        H = create_sparse_matrix((self.dimension, self.dimension))
        for i in range(self.number_of_fermions):
            for j in range(self.number_of_fermions):
                for k in range(len(self.frequency)):
                    #A = numpy.kron(unity, self.d_op_omega[i][k] @ Hc(self.d_op_omega[j][k])) * self.gamma1(i, j, self.frequency[k])
                    #B = numpy.kron(numpy.transpose(Hc(self.d_op_omega[j][k])), self.d_op_omega[i][k]) * self.gamma2(i, j, self.frequency[k])
                    #C = numpy.kron(numpy.transpose(self.d_op_omega[i][k]), Hc(self.d_op_omega[j][k])) * self.gamma1(i, j, self.frequency[k])
                    #D = numpy.kron(numpy.transpose(Hc(self.d_op_omega[j][k]) @ self.d_op_omega[i][k]), unity) * self.gamma2(i, j, self.frequency[k])
                    #E = numpy.kron(unity, Hc(self.d_op_omega[i][k]) @ self.d_op_omega[j][k]) * numpy.conjugate(self.gamma2(i, j, self.frequency[k]))
                    #F = numpy.kron(numpy.transpose(self.d_op_omega[j][k]), Hc(self.d_op_omega[i][k])) * numpy.conjugate(self.gamma1(i, j, self.frequency[k]))
                    #G = numpy.kron(numpy.transpose(Hc(self.d_op_omega[i][k])), self.d_op_omega[j][k]) * numpy.conjugate(self.gamma2(i, j, self.frequency[k]))
                    #H = numpy.kron(numpy.transpose(self.d_op_omega[j][k] @ Hc(self.d_op_omega[i][k])), unity) * numpy.conjugate(self.gamma1(i, j, self.frequency[k]))
                    #part2 += A - B - C + D + E - F - G + H
                    
                    # optimized:
                    A += (self.d_op_omega[i][k] @ self.d_op_omega[j][k].getH()) * self.gamma1(i, j, self.frequency[k])
                    B = sparse.kron(self.d_op_omega[j][k].conjugate(), self.d_op_omega[i][k] * self.gamma2(i, j, self.frequency[k]))
                    C = sparse.kron(self.d_op_omega[i][k].transpose(), self.d_op_omega[j][k].getH() * self.gamma1(i, j, self.frequency[k]))
                    D += (self.d_op_omega[j][k].getH() @ self.d_op_omega[i][k]).transpose() * self.gamma2(i, j, self.frequency[k])
                    E += (self.d_op_omega[i][k].getH() @ self.d_op_omega[j][k]) * numpy.conjugate(self.gamma2(i, j, self.frequency[k]))
                    F = sparse.kron(self.d_op_omega[j][k].transpose(), self.d_op_omega[i][k].getH() * numpy.conjugate(self.gamma1(i, j, self.frequency[k])))
                    G = sparse.kron(self.d_op_omega[i][k].conjugate(), self.d_op_omega[j][k] * numpy.conjugate(self.gamma2(i, j, self.frequency[k])))
                    H += (self.d_op_omega[j][k] @ self.d_op_omega[i][k].getH()).transpose() * numpy.conjugate(self.gamma1(i, j, self.frequency[k]))
                    part2 -= (B + C + F + G)
        part2 += sparse.kron(unity, A + E) + sparse.kron(D + H, unity)
        self.sparse_L = create_sparse_matrix(-1j * _hbar * part1 - _hbar * part2)
        #time2 = time.time()
        #print(f"Liouvillian: {(time2 - time1) * 1e6} µs")
    
    def gamma1(self, i, j, energy):
        leads = numpy.arange(self.number_of_leads, dtype=numpy.int32)
        val = numpy.sum(0.5 * self.Gamma[i,j] * self.fermi(leads, energy))
        if self.include_digamma:
            val += numpy.sum(0.5j * self.Gamma[i,j] * self.digamma(leads, energy))
        return val
    
    def gamma2(self, i, j, energy):
        leads = numpy.arange(self.number_of_leads, dtype=numpy.int32)
        val = numpy.sum(0.5 * self.Gamma[i,j] * (1 - self.fermi(leads, energy)))
        if self.include_digamma:
            val -= numpy.sum(0.5j * self.Gamma[i,j] * self.digamma(leads, energy))
        return val
    
    def gamma1_K(self, i, j, lead, energy):
        val = 0.5 * self.Gamma[i,j,lead] * self.fermi(lead, energy)
        if self.include_digamma:
            val += 0.5j * self.Gamma[i,j,lead] * self.digamma(lead, energy)
        return val
        
    def gamma2_K(self, i, j, lead, energy):
        val = 0.5 * self.Gamma[i,j,lead] * (1 - self.fermi(lead, energy))
        if self.include_digamma:
            val -= 0.5j * self.Gamma[i,j,lead] * self.digamma(lead, energy)
        return val
    
    def fermi(self, lead, energy):
        return 1 / (numpy.exp((energy - self.chemical_potential[lead]) / (k_B * self.temperature[lead])) + 1)
        
    def digamma(self, lead, energy):
        if not self.include_digamma:
            return 0
        return special.digamma(0.5 + 1j * (energy - self.chemical_potential[lead])/(2 * numpy.pi * k_B * self.temperature[lead])).real / numpy.pi
        
    def find_steady_state(self, ignore_coherences=False, additional_coherences=[]):
        #time1 = time.time()
        if ignore_coherences:
            selection = list(numpy.arange(0, self.dimension**2, self.dimension+1))
            for (row, col) in additional_coherences:
                selection.append(row + self.dimension * col)
            selection = numpy.array(selection, dtype=numpy.int32)
            
            L = (self.sparse_L[selection][:,selection]).toarray(order='C')
            
            # trace selection
            L[0][:self.dimension] += 1.
        
        else:
            L = self.sparse_L.toarray(order='C')
            
            # trace selection
            L[0][0::self.dimension+1] += 1.
        
        b = numpy.zeros(L.shape[-1], dtype=numpy.complex128)
        b[0] = 1.
        
        r = linalg.solve(L, b)
        
        del L, b
        gc.collect()
        
        if ignore_coherences:
            rho_ss = numpy.diag(r[:self.dimension])
            for i, (row, col) in enumerate(additional_coherences):
                rho_ss[row, col] = r[self.dimension + i]
            
            #time2 = time.time()
            #print(f"Steadystate: {(time2 - time1) * 1e6} µs")
            return rho_ss
        
        else:
            rho_ss = numpy.reshape(r, (self.dimension, self.dimension), order='F')
            
            #time2 = time.time()
            #print(f"Steadystate: {(time2 - time1) * 1e6} µs")
            return rho_ss
    
    def get_current(self, rho, lead):
        val = 0
        for i in range(self.number_of_fermions):
            for j in range(self.number_of_fermions):
                for k in range(len(self.frequency)):
                    part1 = (self.d_op_omega[j][k].getH() @ rho @ self.d_op_omega[i][k]).trace() * self.gamma1_K(i, j, lead, self.frequency[k])
                    part2 = (rho @ self.d_op_omega[j][k].getH() @ self.d_op_omega[i][k]).trace() * self.gamma2_K(i, j, lead, self.frequency[k])
                    val += part1 - part2 - numpy.conjugate(part2) + numpy.conjugate(part1)
        return -0.2434 * val
        
    def get_currents(self, rho):
        currents = numpy.zeros(self.number_of_leads, dtype=numpy.complex128)
        for lead in range(self.number_of_leads):
            currents[lead] = self.get_current(rho, lead)
        return currents
        

def create_holstein_solver_via_diagonalization(e_0, omega, lamda, N, Gamma, mu_L, mu_R, T_L, T_R, include_digamma=True):
    d_op = numpy.array([[0, 1], [0, 0]])
    d_dag = numpy.transpose(d_op)
    d_op = numpy.kron(d_op, numpy.identity(N))
    d_dag = numpy.kron(d_dag, numpy.identity(N))
    
    a_op = numpy.diag(numpy.sqrt(numpy.arange(1, N)), k=1)
    a_dag = numpy.transpose(a_op)
    a_op = numpy.kron(numpy.identity(2), a_op)
    a_dag = numpy.kron(numpy.identity(2), a_dag)
    
    H_s = e_0 * (d_dag @ d_op)
    H_s += omega * (a_dag @ a_op)
    H_s += lamda * ((a_dag + a_op) @ d_dag @ d_op)
    
    #print(H_s)
    
    return BornMarkovSolver(H_s, [d_op], [a_op], numpy.array([[Gamma]]), numpy.array([mu_L, mu_R]), numpy.array([T_L, T_R]), include_digamma=include_digamma)
    
    
def create_holstein_solver_via_polaron_transformation(e_0, omega, lamda, N, Gamma, mu_L, mu_R, T_L, T_R, include_digamma=True):
    def fc_minus(n, m):
        mu = -lamda / omega
        summe = 0
        mini = numpy.minimum(n, m)
        abso = numpy.abs(n-m)
        for i in range(mini+1):
            summe += (-1 * (mu**2))**i / (numpy.math.factorial(abso + i) * numpy.math.factorial(mini - i) * numpy.math.factorial(i))
        if n > m:
            alt = mu**abso
        elif n == m:
            alt = 1
        else:
            alt = (-mu)**abso
        return numpy.exp(-0.5 * (mu**2)) * summe * numpy.sqrt(float(numpy.math.factorial(n) * numpy.math.factorial(m))) * alt
    
    d_op = numpy.array([[0, 1], [0, 0]])
    d_dag = numpy.transpose(d_op)
    d_op = numpy.kron(d_op, numpy.identity(N))
    d_dag = numpy.kron(d_dag, numpy.identity(N))
    
    a_op = numpy.diag(numpy.sqrt(numpy.arange(1, N)), k=1)
    a_dag = numpy.transpose(a_op)
    a_op = numpy.kron(numpy.identity(2), a_op)
    a_dag = numpy.kron(numpy.identity(2), a_dag)
    
    H_s = (e_0 - lamda**2 / omega) * (d_dag @ d_op)
    H_s += omega * (a_dag @ a_op)
    
    transform = numpy.zeros((N, N), dtype=numpy.complex128)
    for n in range(N):
        for m in range(N):
            transform[n,m] = fc_minus(n, m)
    
    d_transformed = d_op @ numpy.kron(numpy.identity(2), transform)
    
    #print(H_s)
    
    return BornMarkovSolver(H_s, [d_transformed], [a_op], numpy.array([[Gamma]]), numpy.array([mu_L, mu_R]), numpy.array([T_L, T_R]), include_digamma=include_digamma)

