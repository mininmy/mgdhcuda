import cupy as cp
import numpy as np
from dataclasses import dataclass


# ============================================================
# Residual block container
# ============================================================

@dataclass
class ResidualBlocks:
    data: list                  # list[cp.ndarray]
    incompressibility: cp.ndarray
    momentum: list              # list[cp.ndarray]

    def stack(self, weights=None):
        blocks = []

        if weights is None:
            blocks += self.data
            blocks.append(self.incompressibility)
            blocks += self.momentum
        else:
            blocks += [weights["data"] * r for r in self.data]
            blocks.append(weights["div"] * self.incompressibility)
            blocks += [weights["mom"] * r for r in self.momentum]

        return cp.concatenate(blocks)


# ============================================================
# PhysicsAwareGMDH — GPU Only
# ============================================================

class PhysicsAwareGMDH_GPU:

    def __init__(self, viscosity):
        self.viscosity = viscosity

        # frozen GPU tensors
        self.u = None
        self.du_dx = None
        self.du_dt = None
        self.lap_u = None

    # ============================================================
    # Freeze full state on GPU
    # ============================================================

    def freeze_fields(self, models, X_cpu):
        """
        Evaluate velocity, gradient, time derivative, and Laplacian
        directly on GPU.
        """

        Xg = cp.asarray(X_cpu)

        # velocity tensor: (n_components, n_samples)
        self.u = cp.stack(
            [cp.stack([m.evaluate(Xg) for m in comp], axis=0)
             for comp in models],
            axis=0
        )

        n_vars = Xg.shape[1]
        n_spatial = n_vars - 1
        t_idx = n_vars - 1

        # gradients: (comp, model, dim, samples)
        self.du_dx = cp.stack(
            [
                cp.stack(
                    [
                        cp.stack(
                            [m.differentiate(d).evaluate(Xg)
                             for d in range(n_vars)],
                            axis=0
                        )
                        for m in comp
                    ],
                    axis=0
                )
                for comp in models
            ],
            axis=0
        )

        # time derivative
        self.du_dt = self.du_dx[:, :, t_idx]

        # Laplacian (sum spatial second derivatives)
        self.lap_u = cp.stack(
            [
                cp.stack(
                    [
                        sum(
                            m.differentiate(d)
                             .differentiate(d)
                             .evaluate(Xg)
                            for d in range(n_spatial)
                        )
                        for m in comp
                    ],
                    axis=0
                )
                for comp in models
            ],
            axis=0
        )

    # ============================================================
    # Operators (stateless)
    # ============================================================

    def _op_convection(self, u_vec, grad_v):
        R = cp.zeros_like(grad_v[0])
        for j in range(len(u_vec)):
            R += u_vec[j] * grad_v[j]
        return R

    def _op_diffusion(self, lap_v):
        return -self.viscosity * lap_v

    def _op_divergence(self, grad_field):
        R = cp.zeros_like(grad_field[0])
        for j in range(len(grad_field)):
            R += grad_field[j]
        return R

    # ============================================================
    # Base residual (full physics)
    # ============================================================

    def evaluate_residual(self, forcing=None, weights=None):
        """
        Compute full residual blocks from frozen fields.
        """

        n_comp = self.u.shape[0]
        n_models = self.u.shape[1]

        # Use first model per component (minimal assumption)
        u_vec = [self.u[k, 0] for k in range(n_comp)]
        du_dx = [self.du_dx[k, 0] for k in range(n_comp)]
        du_dt = [self.du_dt[k, 0] for k in range(n_comp)]
        lap_u = [self.lap_u[k, 0] for k in range(n_comp)]

        if forcing is None:
            forcing = [cp.zeros_like(u_vec[0]) for _ in range(n_comp)]

        # Data residual = u (no target here in minimal version)
        data_res = u_vec

        # Incompressibility
        incompressibility = self._op_divergence(
            [du_dx[j][j] for j in range(n_comp)]
        )

        # Momentum
        momentum = []
        for k in range(n_comp):
            R = du_dt[k]
            R += self._op_convection(u_vec, du_dx[k])
            R += self._op_diffusion(lap_u[k])
            R -= forcing[k]
            momentum.append(R)

        return ResidualBlocks(data_res, incompressibility, momentum)

    # ============================================================
    # Linearized residual
    # ============================================================

    def evaluate_linearized_residual(
        self,
        phi_data,
        component_i,
    ):
        """
        Compute δR for basis function φ_i.
        phi_data must contain:
            phi
            dphi_dx
            dphi_dt
            laplace_phi
        """

        n_comp = self.u.shape[0]

        u_vec = [self.u[k, 0] for k in range(n_comp)]
        du_dx = [self.du_dx[k, 0] for k in range(n_comp)]

        phi = phi_data["phi"]
        dphi_dx = phi_data["dphi_dx"]
        dphi_dt = phi_data["dphi_dt"]
        lap_phi = phi_data["laplace_phi"]

        # ---- Data
        data = []
        for j in range(n_comp):
            if j == component_i:
                data.append(phi)
            else:
                data.append(cp.zeros_like(phi))

        # ---- Incompressibility
        incompressibility = dphi_dx[component_i]

        # ---- Momentum
        momentum = []
        for k in range(n_comp):
            R = cp.zeros_like(phi)

            if k == component_i:
                R += dphi_dt
                R += self._op_convection(u_vec, dphi_dx)
                R += phi * du_dx[k][k]
                R += self._op_diffusion(lap_phi)
            else:
                R += phi * du_dx[k][component_i]

            momentum.append(R)

        return ResidualBlocks(data, incompressibility, momentum)

    # ============================================================
    # Newton correction
    # ============================================================

    def newton_step(
        self,
        residual_blocks,
        phi_data_list,
        component_i,
        weights=None,
        damping=1e-8,
    ):
        """
        Solve:
            min || R + Σ α_k δR_k ||²
        """

        R_vec = residual_blocks.stack(weights=weights)

        delta_cols = []

        for phi_data in phi_data_list:
            dR = self.evaluate_linearized_residual(
                phi_data,
                component_i,
            )
            delta_cols.append(dR.stack(weights=weights))

        A = cp.stack(delta_cols, axis=1)

        ATA = A.T @ A
        ATA += damping * cp.eye(ATA.shape[0])
        ATb = A.T @ R_vec

        alpha = cp.linalg.solve(ATA, -ATb)

        return alpha
