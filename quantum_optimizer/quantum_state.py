import numpy as np
import qutip as qt

class QuantumState():
    def __init__(
        self,
        N_blocks=None,
        Nq=2,
        Nc=30,
        T2=None,  # Dephasing time
        initial_state=None,
        final_state=None, 
        **kwargs,
    ):
        self.N_blocks = N_blocks
        self.Nq = Nq
        self.Nc = Nc
        self.T2 = T2  # Qubit dephasing time
        self.initial_state = initial_state
        self.final_state = final_state
        self.parameters = {
            "Nq": Nq,
            "Nc": Nc,
        }
        self.parameters.update(kwargs)
        self._operators()
    
    # Initialize all important operators for quantum manipulation 
    def _operators(self):
        self.g = qt.fock(self.Nq, 0)  # |g>
        self.e = qt.fock(self.Nq, 1)  # |e>
        self.eg = qt.tensor(self.e * self.g.dag(), qt.qeye(self.Nc))  # |e><g| operator
        self.sx = qt.tensor(qt.sigmax(), qt.qeye(self.Nc))  # sigmax()
        self.sy = qt.tensor(qt.sigmay(), qt.qeye(self.Nc))  # sigmay()
        self.sz = qt.tensor(qt.sigmaz(), qt.qeye(self.Nc))  # sigmaz()
        self.a = qt.tensor(qt.qeye(self.Nq), qt.destroy(self.Nc))  # Boson annihilation operator 
        self.q = qt.tensor(qt.destroy(self.Nq), qt.qeye(self.Nc))  # Qubit annihilation operator
        
    # Displacement Operator
    def D(self, alpha):
        return (alpha * self.a.dag() - np.conjugate(alpha) * self.a).expm()

    # Unselective Qubit rotation
    def R(self, phi, theta):
        return (-1j * theta / 2 * (self.sx * np.cos(phi) + self.sy * np.sin(phi))).expm()

    # ECD Gate
    def ECD(self, beta):
        return self.D(beta / 2) * self.eg + self.D(-beta / 2) * self.eg.dag()

    # Qubit Dephasing
    def _apply_dephasing(self, state, dt=0.1):
        """
        Applies qubit dephasing to the state using a Lindblad operator.
        Args:
            state (qutip.Qobj): Quantum state.
            dt (float): Time step for the dephasing.
        Returns:
            qutip.Qobj: Updated quantum state with qubit dephasing applied.
        """
        if self.T2 is None:
            return state  # If T2 is not defined, skip dephasing

        gamma_phi = 1 / self.T2  # Dephasing rate
        lindblad_op = np.sqrt(gamma_phi) * self.sz  # Dephasing operator acts only on the qubit
        result = qt.mesolve(0, state, tlist=[0, dt], c_ops=[lindblad_op])
        return result.states[-1]

    # Conditional displacement block (state transfer)
    def gate_block(self, state, beta, phi, theta, dt=0.1):
        """
        Evolves the state under a single gate block and applies qubit dephasing.
        Args:
            state (qutip.Qobj): Quantum state.
            beta (complex): Displacement amplitude.
            phi (float): Rotation axis angle.
            theta (float): Rotation angle.
            dt (float): Time step for dephasing.
        Returns:
            qutip.Qobj: Updated quantum state.
        """
        # Apply gate operations
        state = self.R(phi, theta) * state
        state = self.ECD(beta) * state
        return state
    
    def gate_sequence(self, betas, phis, thetas, step, dt=0.1):
        """
        Evolves the state under a sequence of gates with qubit dephasing.
        Args:
            betas, phis, thetas: Gate parameters for each step.
            step (int): Number of gate steps to apply.
            dt (float): Time step for dephasing after each gate.
        Returns:
            qutip.Qobj: Final quantum state.
        """
        state = self.initial_state
        for i in range(step):
            state = self.gate_block(state, betas[i], phis[i], thetas[i], dt)
        return state 
    
    def fidelity(self, state):
        return qt.fidelity(state, self.final_state)
    
    def inverse_fidelity(self, state):
        return 1 - qt.fidelity(state, self.final_state)

def gkp_state(N, Delta_q, logical):
        """
        Generate an approximate GKP state in the Fock basis.
        
        Args:
            N (int): Number of Fock states in the Hilbert space.
            Delta_q (float): Width of the Gaussian envelope in position space.
            logical (str): Logical state, either '0' or '1'.
        
        Returns:
            Qobj: GKP state as a QuTiP quantum object.
        """
        # Grid in position space
        q_grid = np.linspace(-10, 10, N)
        peaks = 2 * np.sqrt(np.pi) * np.arange(-N//2, N//2)
        
        # Initialize state
        state = np.zeros(N, dtype=complex)
        
        # Add Gaussian peaks
        for p in peaks:
            if logical == '0':
                state += np.exp(-0.5 * (q_grid - p)**2 / Delta_q**2)
            elif logical == '1':
                state += np.exp(-0.5 * (q_grid - (p + np.sqrt(np.pi)))**2 / Delta_q**2)
        
        # Normalize the state
        state /= np.linalg.norm(state)
        return qt.Qobj(state)
gkp_zero = gkp_state(N=100, Delta_q=0.1, logical='0')