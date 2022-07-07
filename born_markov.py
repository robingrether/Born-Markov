import numpy
from scipy import linalg, special, integrate
from scipy.integrate import solve_ivp
import types

hbar = 0.65821195695091   # reduced Planck constant in units eV*fs
_hbar = 1 / hbar          # inverse reduced Planck constant in units 1/(eV*fs)
k_B = 8.6173332621452e-5  # Boltzmann constant in units eV/K

#hbar = 1
#_hbar = 1
#k_B = 1
    

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
    

class BornMarkovSolver:
    
    # time in fs
    # energies in eV
    # temperature in K
    # current in mA
    
    def __init__(self, H_S, d_ops, D1, D2, D1L, D2L, D1R, D2R):
        self.N = len(d_ops)
        self.H_S = H_S
        self.d_ops = d_ops
        self.D1 = D1
        self.D2 = D2
        self.D1L = D1L
        self.D2L = D2L
        self.D1R = D1R
        self.D2R = D2R
        
        n = self.H_S.shape[0]
        unity = numpy.identity(n, dtype=numpy.complex128)
        part0 = numpy.kron(unity, self.H_S) - numpy.kron(numpy.transpose(self.H_S), unity)
        part1 = numpy.zeros((n**2, n**2), dtype=numpy.complex128)
        for i in range(self.N):
            for j in range(self.N):
                #print(self.d_ops[i])
                #print(self.D1[i][j])
                A = numpy.kron(unity, self.d_ops[i] @ self.D1[i][j])
                B = numpy.kron(numpy.transpose(self.D2[i][j]), self.d_ops[i])
                C = numpy.kron(numpy.transpose(self.d_ops[i]), self.D1[i][j])
                D = numpy.kron(numpy.transpose(self.D2[i][j] @ self.d_ops[i]), unity)
                E = numpy.kron(unity, Hc(self.d_ops[i]) @ Hc(self.D2[i][j]))
                F = numpy.kron(numpy.transpose(Hc(self.D1[i][j])), Hc(self.d_ops[i]))
                G = numpy.kron(numpy.transpose(Hc(self.d_ops[i])), Hc(self.D2[i][j]))
                H = numpy.kron(numpy.transpose(Hc(self.D1[i][j]) @ Hc(self.d_ops[i])), unity)
                part1 += A - B - C + D + E - F - G + H
                #print(D)    
        self.Liouvillian = -1j * _hbar * part0 - _hbar * part1
        
    def get_ddt_rho(self, rho):
        part0 = self.H_S @ rho - rho @ self.H_S
        part1 = numpy.zeros(rho.shape, dtype=numpy.complex128)
        for i in range(self.N):
            for j in range(self.N):
                part1 += self.d_ops[i] @ self.D1[i][j] @ rho
                part1 -= self.d_ops[i] @ rho @ self.D2[i][j]
                part1 -= self.D1[i][j] @ rho @ self.d_ops[i]
                part1 += rho @ self.D2[i][j] @ self.d_ops[i]
                
        return -1j * _hbar * part0 - _hbar * part1 - _hbar * numpy.conjugate(numpy.transpose(part1))
    
    def get_current(self, rho):
        part1 = numpy.zeros(rho.shape, dtype=numpy.complex128)
        for i in range(self.N):
            for j in range(self.N):
                part1 -= self.d_ops[i] @ self.D1L[i][j] @ rho
                part1 += self.d_ops[i] @ rho @ self.D2L[i][j]
        
        tr1 = numpy.trace(part1) 
    
        part2 = numpy.zeros(rho.shape, dtype=numpy.complex128)
        for i in range(self.N):
            for j in range(self.N):
                part2 -= self.d_ops[i] @ self.D1R[i][j] @ rho
                part2 += self.d_ops[i] @ rho @ self.D2R[i][j]
        
        tr2 = numpy.trace(part2)
        
        return numpy.array([0.2434 * (tr1 + numpy.conjugate(tr1)), 0.2434 * (tr2 + numpy.conjugate(tr2))])
    
    def propagate(self, rho_0, t, consumer=None, dt=1e-3):
        rho = numpy.array(rho_0, dtype=numpy.complex128)
        if consumer:
            for i in range(int(t/dt)):
                ddt_rho = self.get_ddt_rho(rho)
                consumer(i, rho, ddt_rho)
                rho += ddt_rho * dt
            consumer(int(t/dt), rho, self.get_ddt_rho(rho))
        else:
            for i in range(int(t/dt)):
                rho += self.get_ddt_rho(rho) * dt
                
        return rho
    
    def propagate_RK(self, rho_0, t, dt_max=1e-1):
        rho = numpy.array(rho_0, dtype=numpy.complex128)
        shape = rho.shape
        stuff = solve_ivp(lambda t, rho: self.get_ddt_rho(rho.reshape(shape)).flatten(), (0, t), rho.flatten(), method="RK45", t_eval=(0, t), max_step=dt_max)
        return stuff.y[:,-1].reshape(shape)
    
    def find_steady_state(self):
        L = self.Liouvillian
        
        null_space = linalg.null_space(L)
        
        #u, s, vh = numpy.linalg.svd(L, full_matrices=True)
        #M, N = u.shape[0], vh.shape[1]
        #rcond = numpy.finfo(s.dtype).eps * max(M, N)
        #tol = numpy.amax(s) * rcond
        #num = numpy.sum(s > tol, dtype=int)
        #null_space = vh[num:,:].T.conj()
        
        if null_space.shape[-1] != 1:
            print("ALARM!!! ", null_space.shape[-1], "/", L.shape[-1], sep="")
            raise ValueError
        
        rho_ss = numpy.reshape(null_space[:,0], self.H_S.shape, order='F')
        rho_ss /= numpy.sum(numpy.diag(rho_ss))
        return rho_ss, numpy.copy(L)

    

    def calc_rho(self, rho_0, t, dt=1e-3):
        N = int(t/dt)
        rho_t = numpy.zeros((N+1, rho_0.shape[0], rho_0.shape[1]), dtype=numpy.complex128)
        
        def consume(i, rho, ddt_rho):
            rho_t[i] = rho
        
        self.propagate(rho_0, t, consumer=consume, dt=dt)
        
        return rho_t
    
    def calc_current(self, rho_0, t, dt=1e-3):
        N = int(t/dt)
        current = numpy.zeros((N+1,2))
        
        def consume(i, rho, ddt_rho):
            current[i] = self.get_current(rho)
        
        self.propagate(rho_0, t, consumer=consume, dt=dt)
        
        return current
    
	
def general_solver(H_s_tot, d_ops, a_ops, Gammas, mu_L, mu_R, T_L, T_R, diagonalize=numpy.linalg.eig, include_digamma=True):
    def f_L(E):
        return 1 / (numpy.exp((E - mu_L)/(k_B * T_L)) + 1)
    
    def f_R(E):
        return 1 / (numpy.exp((E - mu_R)/(k_B * T_R)) + 1)
    
    def psi_L(E):
        if include_digamma:
            return special.digamma(0.5 + 1j * (E - mu_L)/(2 * numpy.pi * k_B * T_L)).real / numpy.pi
        else:
            return 0
    
    def psi_R(E):
        if include_digamma:
            return special.digamma(0.5 + 1j * (E - mu_R)/(2 * numpy.pi * k_B * T_R)).real / numpy.pi
        else:
            return 0
    
    n = len(d_ops)
    N = H_s_tot.shape[0]
    Nf = n * 2
    Nb = N - Nf
    
    # create and diagonalize total system Hamiltonian
    #is_diag = (H_s_tot == 0)
    #numpy.fill_diagonal(is_diag, True)
    #if is_diag.all():
    #    w = numpy.diag(H_s_tot)
    #    V = V_dag = numpy.identity(N)
    #    W = H_s_tot
    #else:
    w, V = diagonalize(H_s_tot)
    W = numpy.diag(w)
    V_dag = Hc(V)
    
    # transform fermionic operators
    d_ops_T = []
    d_dag_T = []
    for d_op in d_ops:
        d_ops_T.append(V_dag @ d_op @ V)
        d_dag_T.append(V_dag @ Hc(d_op) @ V)
        
    # create correlation integrated operators
    D1 = []
    D1_L = []
    D1_R = []
    D2 = []
    D2_L = []
    D2_R = []
    freqs = numpy.tile(w,(N,1)).T - numpy.tile(w,(N,1))  # matrix w_n - w_m
    gL = f_L(freqs) + 1j * psi_L(freqs)
    gR = f_R(freqs) + 1j * psi_R(freqs)
    for i in range(n):
        D1.append([])
        D1_L.append([])
        D1_R.append([])
        D2.append([])
        D2_L.append([])
        D2_R.append([])
        for j in range(n):
            if Gammas[i][j] != 0:
                
                # create D1_ij
                D1[i].append(0.5 * Gammas[i][j] * (gL + gR) * d_dag_T[j])
                
                # create D1_L_ij
                D1_L[i].append(0.5 * Gammas[i][j] * gL * d_dag_T[j])
                
                # create D1_R_ij
                D1_R[i].append(0.5 * Gammas[i][j] * gR * d_dag_T[j])
                
                # create D2_ij
                D2[i].append(0.5 * Gammas[i][j] * (2 - gL - gR) * d_dag_T[j])
                
                # create D2_L_ij
                D2_L[i].append(0.5 * Gammas[i][j] * (1 - gL) * d_dag_T[j])
                
                # create D2_R_ij
                D2_R[i].append(0.5 * Gammas[i][j] * (1 - gR) * d_dag_T[j])
            
            else:
                zero_op = numpy.zeros((N,N))
                D1[i].append(zero_op)
                D1_L[i].append(zero_op)
                D1_R[i].append(zero_op)
                D2[i].append(zero_op)
                D2_L[i].append(zero_op)
                D2_R[i].append(zero_op)
    
    solver = BornMarkovSolver(W, d_ops_T, D1, D2, D1_L, D2_L, D1_R, D2_R)
    solver.V = V
    solver.V_dag = V_dag
    return solver
            

def create_holstein_solver_via_diagonalization(e_0, omega, lamda, N, Gamma, mu_L, mu_R, T_L, T_R):
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
    
    return general_solver(H_s, [d_op], [a_op], [[Gamma]], mu_L, mu_R, T_L, T_R)
    
    
    

def create_single_level_solver(e_0, Gamma, mu_L, mu_R, T_L, T_R):
    f_L = 1 / (numpy.exp((e_0 - mu_L)/(k_B * T_L)) + 1)
    f_R = 1 / (numpy.exp((e_0 - mu_R)/(k_B * T_R)) + 1)
    psi_L = special.digamma(0.5 + 1j * (e_0 - mu_L)/(2 * numpy.pi * k_B * T_L)).real / numpy.pi
    psi_R = special.digamma(0.5 + 1j * (e_0 - mu_R)/(2 * numpy.pi * k_B * T_R)).real / numpy.pi
    gamma_1 = (Gamma/2) * ((f_L + f_R) + 1j * (psi_L + psi_R))
    gamma_2 = (Gamma/2) * ((2 - f_L - f_R) - 1j * (psi_L + psi_R))
    gamma_1_L = (Gamma/2) * (f_L + 1j * psi_L)
    gamma_2_L = (Gamma/2) * ((1 - f_L) - 1j * psi_L)
    gamma_1_R = (Gamma/2) * (f_R + 1j * psi_R)
    gamma_2_R = (Gamma/2) * ((1 - f_R) - 1j * psi_R)
    
    ket_0 = numpy.array([1,0])
    ket_1 = numpy.array([0,1])
    
    d = numpy.tensordot(ket_0, ket_1, axes=0)
    d_dag = numpy.tensordot(ket_1, ket_0, axes=0)
    
    return BornMarkovSolver(numpy.diag([0, e_0]), [d], [[d_dag*gamma_1]], [[d_dag*gamma_2]], 
                            [[d_dag*gamma_1_L]], [[d_dag*gamma_2_L]],
                           [[d_dag*gamma_1_R]], [[d_dag*gamma_2_R]])

def create_anderson_solver(e_up, e_down, U, Gamma, mu_L, mu_R, T_L, T_R):
    def f_L(E):
        return 1 / (numpy.exp((E - mu_L)/(k_B * T_L)) + 1)
    
    def f_R(E):
        return 1 / (numpy.exp((E - mu_R)/(k_B * T_R)) + 1)
    
    def psi_L(E):
        return special.digamma(0.5 + 1j * (E - mu_L)/(2 * numpy.pi * k_B * T_L)).real / numpy.pi
    
    def psi_R(E):
        return special.digamma(0.5 + 1j * (E - mu_R)/(2 * numpy.pi * k_B * T_R)).real / numpy.pi
    
    gamma_1 = (Gamma/2) * ((f_L(e_up) + f_R(e_up)) + 1j * (psi_L(e_up) + psi_R(e_up)))
    gamma_2 = (Gamma/2) * ((f_L(e_up+U) + f_R(e_up+U)) + 1j * (psi_L(e_up+U) + psi_R(e_up+U)))
    gamma_3 = (Gamma/2) * ((2 - f_L(e_up) - f_R(e_up)) - 1j * (psi_L(e_up) + psi_R(e_up)))
    gamma_4 = (Gamma/2) * ((2 - f_L(e_up+U) - f_R(e_up+U)) - 1j * (psi_L(e_up+U) + psi_R(e_up+U)))
    gamma_9 = (Gamma/2) * ((f_L(e_down) + f_R(e_down)) + 1j * (psi_L(e_down) + psi_R(e_down)))
    gamma_10 = (Gamma/2) * ((f_L(e_down+U) + f_R(e_down+U)) + 1j * (psi_L(e_down+U) + psi_R(e_down+U)))
    gamma_11 = (Gamma/2) * ((2 - f_L(e_down) - f_R(e_down)) - 1j * (psi_L(e_down) + psi_R(e_down)))
    gamma_12 = (Gamma/2) * ((2 - f_L(e_down+U) - f_R(e_down+U)) - 1j * (psi_L(e_down+U) + psi_R(e_down+U)))
    
    gamma_1L = (Gamma/2) * (f_L(e_up) + 1j * psi_L(e_up))
    gamma_2L = (Gamma/2) * (f_L(e_up+U) + 1j * psi_L(e_up+U))
    gamma_3L = (Gamma/2) * ((1 - f_L(e_up)) - 1j * psi_L(e_up))
    gamma_4L = (Gamma/2) * ((1 - f_L(e_up+U)) - 1j * psi_L(e_up+U))
    gamma_9L = (Gamma/2) * (f_L(e_down) + 1j * psi_L(e_down))
    gamma_10L = (Gamma/2) * (f_L(e_down+U) + 1j * psi_L(e_down+U))
    gamma_11L = (Gamma/2) * ((1 - f_L(e_down)) - 1j * psi_L(e_down))
    gamma_12L = (Gamma/2) * ((1 - f_L(e_down+U)) - 1j * psi_L(e_down+U))
    
    gamma_1R = (Gamma/2) * (f_R(e_up) + 1j * psi_R(e_up))
    gamma_2R = (Gamma/2) * (f_R(e_up+U) + 1j * psi_R(e_up+U))
    gamma_3R = (Gamma/2) * ((1 - f_R(e_up)) - 1j * psi_R(e_up))
    gamma_4R = (Gamma/2) * ((1 - f_R(e_up+U)) - 1j * psi_R(e_up+U))
    gamma_9R = (Gamma/2) * (f_R(e_down) + 1j * psi_R(e_down))
    gamma_10R = (Gamma/2) * (f_R(e_down+U) + 1j * psi_R(e_down+U))
    gamma_11R = (Gamma/2) * ((1 - f_R(e_down)) - 1j * psi_R(e_down))
    gamma_12R = (Gamma/2) * ((1 - f_R(e_down+U)) - 1j * psi_R(e_down+U))
    
    ket_0 = numpy.array([1,0,0,0])
    ket_up = numpy.array([0,1,0,0])
    ket_down = numpy.array([0,0,1,0])
    ket_2 = numpy.array([0,0,0,1])
    
    d_up = numpy.tensordot(ket_0, ket_up, axes=0) - numpy.tensordot(ket_down, ket_2, axes=0)
    d_down = numpy.tensordot(ket_0, ket_down, axes=0) + numpy.tensordot(ket_up, ket_2, axes=0)
    d_ops = [d_up, d_down]
    
    gamma_d_dags1 = [[gamma_1*numpy.tensordot(ket_up, ket_0, axes=0)
                     - gamma_2*numpy.tensordot(ket_2, ket_down, axes=0), numpy.zeros((4,4))],
                    [numpy.zeros((4,4)), gamma_9*numpy.tensordot(ket_down, ket_0, axes=0)
                     + gamma_10*numpy.tensordot(ket_2, ket_up, axes=0)]]
    
    gamma_d_dags2 = [[gamma_3*numpy.tensordot(ket_up, ket_0, axes=0)
                     - gamma_4*numpy.tensordot(ket_2, ket_down, axes=0), numpy.zeros((4,4))],
                    [numpy.zeros((4,4)), gamma_11*numpy.tensordot(ket_down, ket_0, axes=0)
                     + gamma_12*numpy.tensordot(ket_2, ket_up, axes=0)]]
    
    gamma_d_dags1L = [[gamma_1L*numpy.tensordot(ket_up, ket_0, axes=0)
                     - gamma_2L*numpy.tensordot(ket_2, ket_down, axes=0), numpy.zeros((4,4))],
                    [numpy.zeros((4,4)), gamma_9L*numpy.tensordot(ket_down, ket_0, axes=0)
                     + gamma_10L*numpy.tensordot(ket_2, ket_up, axes=0)]]
    
    gamma_d_dags2L = [[gamma_3L*numpy.tensordot(ket_up, ket_0, axes=0)
                     - gamma_4L*numpy.tensordot(ket_2, ket_down, axes=0), numpy.zeros((4,4))],
                    [numpy.zeros((4,4)), gamma_11L*numpy.tensordot(ket_down, ket_0, axes=0)
                     + gamma_12L*numpy.tensordot(ket_2, ket_up, axes=0)]]
    
    gamma_d_dags1R = [[gamma_1R*numpy.tensordot(ket_up, ket_0, axes=0)
                     - gamma_2R*numpy.tensordot(ket_2, ket_down, axes=0), numpy.zeros((4,4))],
                    [numpy.zeros((4,4)), gamma_9R*numpy.tensordot(ket_down, ket_0, axes=0)
                     + gamma_10R*numpy.tensordot(ket_2, ket_up, axes=0)]]
    
    gamma_d_dags2R = [[gamma_3R*numpy.tensordot(ket_up, ket_0, axes=0)
                     - gamma_4R*numpy.tensordot(ket_2, ket_down, axes=0), numpy.zeros((4,4))],
                    [numpy.zeros((4,4)), gamma_11R*numpy.tensordot(ket_down, ket_0, axes=0)
                     + gamma_12R*numpy.tensordot(ket_2, ket_up, axes=0)]]
    
    return BornMarkovSolver(numpy.diag([0, e_up, e_down, e_up+e_down+U]), d_ops, gamma_d_dags1,
                           gamma_d_dags2, gamma_d_dags1L, gamma_d_dags2L,
                           gamma_d_dags1R, gamma_d_dags2R)

def create_holstein_solver(e_0, omega, lamda, N, Gamma, mu_L, mu_R, T_L, T_R, use_exp=False):
    def f_L(n, m):
        return 1 / (numpy.exp((e_0 - lamda**2/omega + (n-m)*omega - mu_L)/(k_B * T_L)) + 1)
    
    def f_R(n, m):
        return 1 / (numpy.exp((e_0 - lamda**2/omega + (n-m)*omega - mu_R)/(k_B * T_R)) + 1)
    
    def psi_L(n, m):
        return special.digamma(0.5 + 1j * (e_0 - lamda**2/omega + (n-m)*omega - mu_L)/(2 * numpy.pi * k_B * T_L)).real / numpy.pi
    
    def psi_R(n, m):
        return special.digamma(0.5 + 1j * (e_0 - lamda**2/omega + (n-m)*omega - mu_R)/(2 * numpy.pi * k_B * T_R)).real / numpy.pi
        
    def fc_plus(n, m):
        mu = lamda / omega
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
    
    
    if not use_exp:
        FC_plus = numpy.zeros((N, N), dtype=numpy.float64)
        for n in range(N):
            for m in range(N):
                FC_plus[n,m] = fc_plus(n,m)

        FC_minus = numpy.zeros((N, N), dtype=numpy.float64)
        for n in range(N):
            for m in range(N):
                FC_minus[n,m] = fc_minus(n,m)
    
    d_op = numpy.array([[0, 1], [0, 0]])
    d_dag = numpy.transpose(d_op)
    d_op = numpy.kron(d_op, numpy.identity(N)) #numpy.kron(numpy.identity(N), d_op)
    d_dag = numpy.kron(d_dag, numpy.identity(N)) #numpy.kron(numpy.identity(N), d_dag)
    
    if use_exp:
        a_op = numpy.diag(numpy.sqrt(numpy.arange(1, N)), k=1)
        a_dag = numpy.transpose(a_op)
        FC_plus = linalg.expm((lamda/omega) * (a_dag - a_op))
        FC_minus = linalg.expm((lamda/omega) * (a_op - a_dag))
    
    Gamma_1L = numpy.zeros((N, N), dtype=numpy.complex128)
    for n in range(N):
        for m in range(N):
            Gamma_1L[n,m] = FC_plus[n,m] * (f_L(n, m) + 1j * psi_L(n, m))
    Gamma_1L = (Gamma/2) * numpy.kron(numpy.identity(2), Gamma_1L) #numpy.kron(Gamma_1L, numpy.identity(2))
    
    Gamma_1R = numpy.zeros((N, N), dtype=numpy.complex128)
    for n in range(N):
        for m in range(N):
            Gamma_1R[n,m] = FC_plus[n,m] * (f_R(n, m) + 1j * psi_R(n, m))
    Gamma_1R = (Gamma/2) * numpy.kron(numpy.identity(2), Gamma_1R) #numpy.kron(Gamma_1R, numpy.identity(2))
    
    Gamma_1tot = numpy.zeros((N, N), dtype=numpy.complex128)
    for n in range(N):
        for m in range(N):
            Gamma_1tot[n,m] = FC_plus[n,m] * (f_L(n, m) + f_R(n, m) + 1j * psi_L(n, m) + 1j * psi_R(n, m))
    Gamma_1tot = (Gamma/2) * numpy.kron(numpy.identity(2), Gamma_1tot) #numpy.kron(Gamma_1tot, numpy.identity(2))
    
    Gamma_2L = numpy.zeros((N, N), dtype=numpy.complex128)
    for n in range(N):
        for m in range(N):
            Gamma_2L[n,m] = FC_plus[n,m] * ((1-f_L(n, m)) - 1j * psi_L(n, m))
    Gamma_2L = (Gamma/2) * numpy.kron(numpy.identity(2), Gamma_2L) #numpy.kron(Gamma_2L, numpy.identity(2))
    
    Gamma_2R = numpy.zeros((N, N), dtype=numpy.complex128)
    for n in range(N):
        for m in range(N):
            Gamma_2R[n,m] = FC_plus[n,m] * ((1-f_R(n, m)) - 1j * psi_R(n, m))
    Gamma_2R = (Gamma/2) * numpy.kron(numpy.identity(2), Gamma_2R) #numpy.kron(Gamma_2R, numpy.identity(2))
    
    Gamma_2tot = numpy.zeros((N, N), dtype=numpy.complex128)
    for n in range(N):
        for m in range(N):
            Gamma_2tot[n,m] = FC_plus[n,m] * (2 - f_L(n, m) - f_R(n, m) - 1j * psi_L(n, m) - 1j * psi_R(n, m))
    Gamma_2tot = (Gamma/2) * numpy.kron(numpy.identity(2), Gamma_2tot) #numpy.kron(Gamma_2tot, numpy.identity(2))
    
    #FC_plus = numpy.kron(FC_plus, numpy.identity(2))
    FC_minus = numpy.kron(numpy.identity(2), FC_minus) #numpy.kron(FC_minus, numpy.identity(2))
    
    H_S = numpy.diag(numpy.array([i*omega for i in range(N)] * 2)) + numpy.kron(numpy.diag([0,1]), numpy.diag([e_0 - lamda**2/omega] * N))
    
    return BornMarkovSolver(H_S, [d_op @ FC_minus], [[d_dag @ Gamma_1tot]], [[d_dag @ Gamma_2tot]],
                           [[d_dag @ Gamma_1L]], [[d_dag @ Gamma_2L]], [[d_dag @ Gamma_1R]], [[d_dag @ Gamma_2R]])


def create_anderson_hopping_solver(e_1, e_2, U, t, Gamma, mu_L, mu_R, T_L, T_R):
    ket_0 = numpy.array([1,0,0,0])
    ket_1 = numpy.array([0,1,0,0])
    ket_2 = numpy.array([0,0,1,0])
    ket_12 = numpy.array([0,0,0,1])

    d_1 = numpy.tensordot(ket_0, ket_1, axes=0) - numpy.tensordot(ket_2, ket_12, axes=0)
    d_2 = numpy.tensordot(ket_0, ket_2, axes=0) + numpy.tensordot(ket_1, ket_12, axes=0)
    d_ops = [d_1, d_2]

    H_S = e_1 * Hc(d_1) @ d_1 + e_2 * Hc(d_2) @ d_2 + U * Hc(d_1) @ d_1 @ Hc(d_2) @ d_2 - t * (Hc(d_1) @ d_2 + Hc(d_2) @ d_1)
    
    return general_solver(H_S, [d_1, d_2], [], Gamma * numpy.ones((2,2)), mu_L, mu_R, T_L, T_R)
   
  
def calc_langevin_quantities(H_s_func, ddx_H_s_func, x, d_ops, a_ops, Gammas, mu_L, mu_R, T_L, T_R, use_pinv=False, diagonalize=numpy.linalg.eig, include_digamma=True):
    xs = len(x)
    
    mean_force = numpy.zeros(xs, dtype=numpy.complex128)
    friction = numpy.zeros((xs, xs), dtype=numpy.complex128)
    correlation = numpy.zeros((xs, xs), dtype=numpy.complex128)
    
    solver = general_solver(H_s_func(x), d_ops, a_ops, Gammas, mu_L, mu_R, T_L, T_R, diagonalize=diagonalize, include_digamma=include_digamma)
    rho_ss, L = solver.find_steady_state()
    V, V_dag = solver.V, solver.V_dag
    if use_pinv:
        L_inv = linalg.pinv(L)
    
    N = 2
    dx = 1e-4
    integral_lim = numpy.inf
    
    for nu in range(xs):
        ddx_H_s_nu = V_dag @ ddx_H_s_func(nu, x) @ V
        
        mean_force[nu] = -numpy.trace(ddx_H_s_nu @ rho_ss)
        
        vec_dx = numpy.zeros(xs)
        vec_dx[nu] = dx
        
        rhos = []
        for i in (list(range(-N, 0)) + list(range(1, N+1))):
            H_temp = H_s_func(x + i*vec_dx)
            #solver = create_single_level_solver(H_temp[1,1]-H_temp[0,0], Gammas[0][0], mu_L, mu_R, T_L, T_R)
            solver2 = general_solver(H_temp, d_ops, a_ops, Gammas, mu_L, mu_R, T_L, T_R, diagonalize=diagonalize, include_digamma=include_digamma)
            #print(solver.H_S)
            rho, L2 = solver2.find_steady_state()
            V2, V2_dag = solver2.V, solver2.V_dag
            rhos.append(V2 @ rho @ V2_dag)
        if N == 2:
            coeff = [1/12, -2/3, 2/3, -1/12]
        elif N == 1:
            coeff = [-1/2, 1/2]
        ddx_rho = numpy.zeros(rhos[0].shape, dtype=numpy.complex128)
        for rho, c in zip(rhos, coeff):
            ddx_rho += rho * c
        ddx_rho /= dx
        ddx_rho = V_dag @ ddx_rho @ V
        
        for alpha in range(xs):
            
            ddx_H_s_alpha = V_dag @ ddx_H_s_func(alpha, x) @ V
            
            if use_pinv:
                friction[alpha, nu] = numpy.trace(ddx_H_s_alpha @ ((L_inv @ ddx_rho.flatten(order='F')).reshape(ddx_rho.shape, order='F')))
                correlation[alpha, nu] = -0.5 * numpy.trace(ddx_H_s_alpha @ ((L_inv @ (numpy.kron(numpy.identity(rho_ss.shape[0]), ddx_H_s_nu) + numpy.kron(numpy.transpose(ddx_H_s_nu), numpy.identity(rho_ss.shape[0])) + 2 * mean_force[nu] * numpy.identity(L.shape[0])) @ rho_ss.flatten(order='F')).reshape(rho_ss.shape, order='F')))
            else:
                friction[alpha, nu] = integrate.quad(lambda lamda: numpy.trace(ddx_H_s_alpha @ ((-linalg.expm(L * lamda) @ ddx_rho.flatten(order='F')).reshape(ddx_rho.shape, order='F')), dtype=numpy.float64), 0, integral_lim)[0]
                correlation[alpha, nu] = -0.5 * integrate.quad(lambda lamda: numpy.trace(ddx_H_s_alpha @ ((-linalg.expm(L * lamda) @ (numpy.kron(numpy.identity(rho_ss.shape[0]), ddx_H_s_nu) + numpy.kron(numpy.transpose(ddx_H_s_nu), numpy.identity(rho_ss.shape[0])) + 2 * mean_force[nu] * numpy.identity(L.shape[0])) @ rho_ss.flatten(order='F')).reshape(rho_ss.shape, order='F')), dtype=numpy.float64), 0, integral_lim)[0]
            
    return mean_force, friction, correlation