import numpy as np

class NBodySystem:
    def __init__(
        self,
        N: int,
        dimension: int = 3,
        G: float = 1.0,
        R: float = 1.0,
        mass_range: tuple = (0.5, 2.0),
        force_law=None,
        seed: int | None = None,   # Optional seed
    ):
        """
        N-body system in arbitrary dimensions with flexible force law.

        Parameters
        ----------
        N : int
            Number of bodies
        dimension : int
            Spatial dimension
        G : float
            Generalized gravitational constant
        R : float
            Scale for random initial positions
        mass_range : tuple
            (min_mass, max_mass) for random mass generation
        force_law : callable
            Function f(r, G, m1, m2) -> force magnitude
            If None, defaults to inverse-square law: G * m1 * m2 / r^2
        """

        if seed is not None:
            np.random.seed(seed)

        self.N = N
        self.D = dimension
        self.positions = 2*R*(np.random.rand(N, dimension) - 0.5)
        self.velocities = np.zeros((N, dimension))
        self.masses = np.random.uniform(mass_range[0], mass_range[1], size=(N,1))
        self.G = G

        if force_law is None:
            # Default inverse-square law
            self.force_law = lambda r, G, m1, m2: G * m1 * m2 / (r**2 + 1e-8)
        else:
            self.force_law = force_law

    def step(self, dt):
        forces = np.zeros_like(self.positions)
        for i in range(self.N):
            for j in range(i+1, self.N):
                r_vec = self.positions[j] - self.positions[i]
                dist = np.linalg.norm(r_vec) + 1e-8
                f_mag = self.force_law(dist, self.G, self.masses[i], self.masses[j])
                f_vec = f_mag * r_vec / dist  # vector in direction of separation
                forces[i] += f_vec
                forces[j] -= f_vec  # Newton's 3rd law

        self.velocities += forces / self.masses * dt
        self.positions += self.velocities * dt
