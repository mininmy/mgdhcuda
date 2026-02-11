from dataclasses import dataclass
import cupy as cp


class PhiDescriptor:
    __slots__ = ("kind", "s", "k")

    def __init__(self, kind, s=None, k=None):
        self.kind = kind          # "const" | "linear" | "product"
        self.s = s
        self.k = k

    def __hash__(self):
        return hash((self.kind, self.s, self.k))

    def __eq__(self, other):
        return (
            isinstance(other, PhiDescriptor)
            and self.kind == other.kind
            and self.s == other.s
            and self.k == other.k
        )

    def __repr__(self):
        return f"PhiDescriptor({self.kind}, s={self.s}, k={self.k})"


@dataclass
class PhiData:
    phi: cp.ndarray            # (N,)
    dt: cp.ndarray             # (N,)
    grad: list                 # list of (N,)
    laplace: cp.ndarray        # (N,)

class PhiCache:
    def __init__(self):
        self._cache = {}

    def get(self, desc):
        return self._cache.get(desc)

    def put(self, desc, phi_data):
        self._cache[desc] = phi_data

    def clear(self):
        self._cache.clear()


def build_phi_data(desc, u, du_dt, grad_u, lap_u):
    """
    Build φ_i and its derivatives on GPU.
    All inputs are lists of cp.ndarray.
    """

    if desc.kind == "const":
        N = u[0].shape[0]
        return PhiData(
            phi=cp.ones(N),
            dt=cp.zeros(N),
            grad=[cp.zeros(N) for _ in grad_u[0]],
            laplace=cp.zeros(N),
        )

    if desc.kind == "linear":
        s = desc.s
        return PhiData(
            phi=u[s],
            dt=du_dt[s],
            grad=grad_u[s],
            laplace=lap_u[s],
        )

    if desc.kind == "product":
        s, k = desc.s, desc.k

        phi = u[s] * u[k]
        dt = du_dt[s] * u[k] + u[s] * du_dt[k]

        grad = [
            grad_u[s][j] * u[k] + u[s] * grad_u[k][j]
            for j in range(len(grad_u[s]))
        ]

        laplace = (
            lap_u[s] * u[k]
            + u[s] * lap_u[k]
            + 2 * sum(grad_u[s][j] * grad_u[k][j]
                      for j in range(len(grad)))
        )

        return PhiData(phi, dt, grad, laplace)

    raise ValueError(f"Unknown φ kind: {desc.kind}")

def generate_phi_descriptors(n_u: int) -> list[PhiDescriptor]:
    descs = []

    # 1. constant
    descs.append(PhiDescriptor(kind="const"))

    # 2. linear terms
    for s in range(n_u):
        descs.append(PhiDescriptor(kind="linear", s=s))

    # 3. quadratic products (s ≤ k to avoid duplicates)
    for s in range(n_u):
        for k in range(s, n_u):
            descs.append(PhiDescriptor(kind="product", s=s, k=k))

    return descs
