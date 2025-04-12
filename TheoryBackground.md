# Theory Background

# 1. Wigner Function

### Definition

The [**Wigner function**](https://en.wikipedia.org/wiki/Wigner_quasiprobability_distribution) is a mathematical tool that can help us visualize and analyze quantum states in [phase space](https://en.wikipedia.org/wiki/Phase_space).  However, unlike a classical [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution), the Wigner function can take **negative values**, which reveal quantum interference effects and non-classicality.

For a single-mode state with density matrix $\rho$, the Wigner function $W_{\rho}$ in  can be written as:

$$
W_{\rho}(x,p) = \frac{1}{\pi\hbar}\int_{-\infty}^{\infty}\bra{x+y}\rho \ket{x-y}e^{2ipy/\hbar}dy
$$

This formula may look intimidating, so let’s **break it down step by step**:

1. $x$ and $p$: These represent the position and momentum of the system, just like in classical mechanics. The Wigner function assigns a value to every possible combination of $(x,p)$, similar to a probability distribution.
2. $\bra{x+y}\rho\ket{x-y}$ : This is the **density matrix in the position basis**, but instead of being evaluated at a single point, it is averaged over two points, $x+y$ and $x-y$. This captures the **coherence** of the system, which is responsible for interference effects.
3. The integral $\int e^{-2ipy/\hbar}dy$ : This is a [**Fourier transform**](https://en.wikipedia.org/wiki/Fourier_transform), which translates information from position space to momentum space. It allows us to see how different position states interfere with each other in the momentum domain.
4. $\frac{1}{\pi\hbar}$: normalization factor that ensures the Wigner function integrates correctly over phase space.

### What Does the Wigner Function Represent?

In classical mechanics, the state of a system is fully described by a by a **point** in phase space. However, in quantum mechanics, Heisenberg’s uncertainty principle prevents us from simultaneously knowing both position and momentum with perfect precision. The Wigner function provides a way to **visualize quantum states in phase space**, but it is not a true probability distribution because it can take **negative values**, which have no classical interpretation.

- Positive regions of $W_\rho(x,p)$ indicate where the quantum state is more likely to be found.
- **Negative regions** indicate quantum interference effects.
- **Zero regions** mean that the state has no contribution at those phase-space coordinates.

The presence of negative values in $W_\rho(x,p)$ show the **non-classical behavior** of the system. Classical states (like thermal states or coherent states of light) have strictly non-negative Wigner functions, whereas quantum states with superposition effects, such as **Schrödinger cat states** or **Fock states**, exhibit negative regions.

### Key properties of the Winger function:

The Wigner function satisfies:

1. **Normalization:**

$$
\int W_\rho(x,p) dxdp = 1
$$

This ensures that the total "weight" of the distribution sums to 1, like a classical probability distribution.

1. $W_\rho(x,p)$ is a real-valued function.

### The Displacement and Parity Operators

To understand the alternative form of the Wigner function, we introduce two key operators:

- The **displacement operator $D(\alpha)$  is defined as:**

$$
D(\alpha) = \exp(\alpha\hat{a}^\dagger - \alpha\hat{a})
$$

where $\alpha \in \mathbb{C}$ and $\hat{a}$ is the annihilation operator. This operator shifts the phase space origin by $\alpha$; that is, $D(\alpha)\ket{0} = \ket{\alpha}$ produces a coherent state centered at $\alpha$

- The **parity operator** $P$ is given by:

$$
P = e^{i\pi \hat{a}^\dagger \hat{a}}
$$

In the Fock basis $\{\ket{n}\}$, this acts as:

$$
P\ket{n} = (-1)^n\ket{n}
$$

That is, it assigns +1 to even photon number states, and –1 to odd ones. It’s a reflection symmetry in phase space, flipping the sign of both quadratures.

### The Displaced-Parity Expectation Formula

Using these operators, we can write the Wigner function in a completely different — and very powerful — way:

$$
W(\alpha) = \frac{2}{\pi} \text{Tr}\left[ D(\alpha) P D^\dagger(\alpha )  \rho \right]
$$

This means: displace the state $\rho$  by $-\alpha$, then measure the expectation value of the parity operator. Multiply the result by $2/\pi$ to get the Wigner function at that point.

---

# 2.  Density Matrix

The [density matrix](https://en.wikipedia.org/wiki/Density_matrix) $\rho$ is a fundamental object in quantum mechanics that describes the state of a quantum system. Unlike the [wave function](https://en.wikipedia.org/wiki/Wave_function) $\ket{\psi}$, which fully describes **pure states**, the density matrix provides a more general description, encompassing both [**pure** and **mixed** states.](https://en.wikipedia.org/wiki/Quantum_state)

### What is a Density Matrix?

The density matrix is defined as:

$$
\rho = \sum_ip_i\ket{\psi_i}\bra{\psi_i}
$$

where:

- $\ket{\psi_i}$ are orthogonal quantum states
- $p_i$ are probabilities that the system is in state $\ket{\psi_i}$ (such that $\sum_ip_i = 1$).

The density matrix allows us to describe **statistical mixtures** of quantum states, which is crucial when dealing with realistic quantum systems affected by decoherence or incomplete knowledge.

### Key Properties of the Density Matrix

1. **Hermitian: $\rho^{\dagger} = \rho$**
2. **Positive Semi-Definite**: All eigenvalues are non-negative ($\lambda_i \geq 0)$.
3. **Trace is 1: $\text{Tr}(\rho) =1$,** ensuring proper normalization.

### Pure vs. Mixed States

- A **pure state** satisfies $\rho ^2 = \rho$, meaning it has only one nonzero eigenvalue (equal to 1). Example: A qubit in the state $\ket{\psi} = \alpha\ket{0} + \beta\ket{1}$ has a density matrix $\rho = \ket{\psi}\bra{\psi}$.
- A **mixed state** describes probabilistic combinations of different pure states and satisfies $\rho^2 \neq \rho$. Example:

### Density Matrix Reconstruction

If one measures the Wigner function $W_\rho(x,p)$ across a sufficient range of $(x,p)$ points with enough resolution, it is possible to “invert” the transform to retrieve the density matrix $\rho$. Conceptually, the process is similar to tomographic inversion. Numerically, we often define:

$$
\tilde{\rho} = \text{fit}(W(x,p))
$$

where $\text{fit}$ is an algorithm (e.g., using [maximum likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) or [least-squares](https://en.wikipedia.org/wiki/Least_squares) approaches) that reconstructs $\rho$ by fitting model Wigner functions to data.

### Fidelity

[Fidelity](https://en.wikipedia.org/wiki/Fidelity_of_quantum_states) is a measure of **how similar two quantum states are.** It tells us how much one quantum state **overlaps** with another, similar to how you might compare two probability distributions or two vectors in a geometric space. The fidelity between two density matrices $\rho$ and $\sigma$ is defined as:

$$
F(\rho,\sigma) = \left(\text{Tr} \sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}\right)^2
$$

Intuitively, fidelity quantifies how much two quantum states "agree" with each other. If $F(\rho,\sigma) = 1$ the states are **identical**, meaning no difference can be detected. If $F(\rho,\sigma) = 0$, the states are **completely different** and can be perfectly distinguished. Intermediate values indicate partial similarity, where lower fidelity implies greater deviation between the two states.

In the context of quantum tomography and denoising, fidelity is a crucial metric for evaluating how well an estimated state $\tilde{\rho}$ approximates the true state $\rho$. A higher fidelity means the reconstruction or denoising process has preserved the essential quantum properties of the original state. This measure appears in our loss functions for denoising (see below).

---

# **3. Wigner Tomography: A Convex‑Optimization Approach**

### From Wigner Function to Probability

Although the Wigner function is not a true probability distribution (since it can be negative), the parity measurement **is a physical measurement** with outcomes $\pm1$. So, instead of working directly with $W(\alpha)$,  we use **measurement operators** that give probabilities.

We define the operator:

$$
E_\alpha = \frac{1}{2}(I + D(\alpha)PD^\dagger(\alpha))
$$

Then, using Born's rule, we compute the **probability of obtaining even parity** after displacing the state by $\alpha$:

$$
p_\alpha = \text{Tr}(E_\alpha\rho)
$$

This maps directly to the Wigner function via:

$$
W(\alpha) = \frac{2}{\pi} (2p_\alpha -1 ) \Longleftrightarrow p_\alpha  = \frac{1}{2}(1+\frac{\pi}{2}W(\alpha))
$$

This is extremely important: **if we know $W(\alpha)$, we can compute the corresponding measurement probability**, and vice versa.

### What This Means Practically

From this point on, we switch our perspective:

- Instead of thinking of $W(\alpha)$ as a “given function”, we now treat the **measured parity probabilities** $p_\alpha$ as **data points.**
- Each point $\alpha$ corresponds to a measurement operator $E_\alpha$ and we seek to find a density matrix $\rho$ such that:

$$
p_\alpha = \text{Tr}(E_\alpha \rho)
$$

This allows us to reconstruct $\rho$ from measured values of the Wigner function or from actual measurement outcomes.

### Convex Optimization for State Reconstruction

Once we have a set of **parity measurement outcomes $p_{\alpha_k}$,  each corresponding to a phase-space point $\alpha_k$** the goal is to reconstruct the **density matrix $\rho$ hat best explains the data.**

Suppose we are given a finite number of measurement results $\\{p_{\alpha_k}\\}_{k=1}^n$, we want to find the density matrix $\rho$ that minimizes the squared error between the predicted probabilities and the observed ones:

$$
\mathcal{L(\rho)} = \sum_{k=1}^n|\text{Tr}(E_{\alpha_k}\rho) - p_{\alpha_k}|^2
$$

This objective is convex in $\rho$ and we can impose the following physical constraints:

$$
\rho \succeq 0 (\text{positive semidefinite}),  \text{Tr}(\rho) = 1
$$

# References

1. Réglade, U., Bocquet, A., Gautier, R., Cohen, J., Marquet, A.,
Albertinale, E., ... & Leghtas, Z. (2024). Quantum control of a cat
qubit with bit-flip times exceeding ten seconds. *Nature*, *629*(8013), 778-783.
2. Strandberg, I. (2022). Simple,
reliable, and noise-resilient continuous-variable quantum state
tomography with convex optimization. *Physical Review Applied*, *18*(4), 044041.
3. FitzGerald, G., & Yeadon, W. (2025). QSTToolkit: A Python Library for Deep Learning Powered Quantum State Tomography. *arXiv preprint arXiv:2503.14422*.
4. Guillaud, J., Cohen, J., & Mirrahimi, M. (2023). Quantum computation with cat qubits. *SciPost Physics Lecture Notes*, 072.
5. Chamberland, C., Noh, K., Arrangoiz-Arriola, P., Campbell, E. T., Hann, 
C. T., Iverson, J., ... & Brandão, F. G. (2022). Building a 
fault-tolerant quantum computer using concatenated cat codes. *PRX Quantum*, *3*(1), 010329.
6. [`dynamiqs` website](https://www.dynamiqs.org/stable/)
