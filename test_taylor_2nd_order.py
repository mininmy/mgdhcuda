"""
test_taylor_2nd_order.py
========================
Analytical baseline: 4th-order Taylor expansion of the Taylor-Green u-velocity
around (x, y, t) = (0, 0, 0).

  u_true(x, y, t) = sin(x) cos(y) exp(-2νt)

Component expansions (through relevant order):
  sin(x)    = x  -  x³/6  + O(x⁵)
  cos(y)    = 1  -  y²/2  + y⁴/24  + O(y⁶)
  e^{-2νt}  = 1  -  2νt  + 2ν²t²  - (4ν³/3)t³  + (2ν⁴/3)t⁴  + O(t⁵)

Product, keeping only monomials with total degree (i+j+k) ≤ 4:
  x·e terms (sin·cos gives x, -x³/6, -xy²/2):
    x × 1              =   x
    x × (−2νt)         = −2νxt
    x × 2ν²t²         =  2ν²xt²
    x × (−4ν³t³/3)    = −(4ν³/3)xt³
    (−x³/6) × 1        = −x³/6
    (−x³/6) × (−2νt)   =  (ν/3)x³t
    (−xy²/2) × 1       = −xy²/2
    (−xy²/2) × (−2νt)  =  νxy²t

  u_approx(x, y, t)  =  x
                       − 2ν·xt
                       + 2ν²·xt²
                       − x³/6
                       − xy²/2
                       − (4ν³/3)·xt³
                       + (ν/3)·x³t
                       + ν·xy²t

Exact first-order derivatives:
  ∂u/∂x = 1 − 2νt + 2ν²t² − x²/2 − y²/2 − (4ν³/3)t³ + νx²t + νy²t
  ∂u/∂y = xy(2νt − 1)
  ∂u/∂t = −2νx + 4ν²xt − 4ν³xt² + (ν/3)x³ + νxy²

Laplacian (∂²/∂x² + ∂²/∂y²):
  ∂²u/∂x² = x(2νt − 1)      (from −x²/2 and +νx²t terms in ∂u/∂x)
  ∂²u/∂y² = x(2νt − 1)      (from −xy and +2νxyt terms in ∂u/∂y)
  ∇²u      = 2x(2νt − 1) = −2x + 4νxt

Physics residuals:
  DIV  = ∂u_approx/∂x  +  ∂v_known/∂y
  MOM  = ∂u_approx/∂t  +  u_approx·(∂u_approx/∂x)
           + v_known·(∂u_approx/∂y)  −  ν·∇²u_approx  +  pg_x_known

True u satisfies both physics equations exactly, so these residuals measure
the truncation error of the Taylor approximation.
"""

import numpy as np
from gpu_gmdh_newton_known_physics import generate_taylor_green_data_with_known


# --------------------------------------------------------------------------- #
# Core functions                                                               #
# --------------------------------------------------------------------------- #

def taylor_u_4th(X: np.ndarray, nu: float):
    """
    Evaluate the 4th-order Taylor expansion of the Taylor-Green u-velocity
    and its first-order partial derivatives / Laplacian.

    Parameters
    ----------
    X  : ndarray [N, 3]  — columns are (x, y, t)
    nu : float           — kinematic viscosity

    Returns
    -------
    u_approx : [N]   x − 2νxt + 2ν²xt² − x³/6 − xy²/2 − (4ν³/3)xt³ + (ν/3)x³t + νxy²t
    du_dx    : [N]   ∂u_approx/∂x
    du_dy    : [N]   ∂u_approx/∂y
    du_dt    : [N]   ∂u_approx/∂t
    lap_u    : [N]   ∇²u_approx = ∂²u/∂x² + ∂²u/∂y²
    """
    x, y, t = X[:, 0], X[:, 1], X[:, 2]
    nu2 = nu * nu
    nu3 = nu2 * nu

    u_approx = (x
                - 2.0*nu * x*t
                + 2.0*nu2 * x*t**2
                - x**3 / 6.0
                - x*y**2 / 2.0
                - (4.0*nu3/3.0) * x*t**3
                + (nu/3.0) * x**3 * t
                + nu * x*y**2 * t)

    du_dx = (1.0
             - 2.0*nu*t
             + 2.0*nu2*t**2
             - x**2 / 2.0
             - y**2 / 2.0
             - (4.0*nu3/3.0)*t**3
             + nu*x**2*t
             + nu*y**2*t)

    du_dy = x*y * (2.0*nu*t - 1.0)        # −xy + 2νxyt

    du_dt = (-2.0*nu*x
             + 4.0*nu2 * x*t
             - 4.0*nu3 * x*t**2
             + (nu/3.0) * x**3
             + nu * x*y**2)

    # ∂²u/∂x² = x(2νt−1),  ∂²u/∂y² = x(2νt−1)
    lap_u = 2.0 * x * (2.0*nu*t - 1.0)    # −2x + 4νxt

    return u_approx, du_dx, du_dy, du_dt, lap_u

def taylor_u_10th(X: np.ndarray, nu: float):
    """
    10th-order Taylor expansion of:

        u(x,y,t) = sin(x) cos(y) exp(-2νt)

    around (x,y,t) = (0,0,0).

    Keeps ALL monomials with total degree <= 10.

    Returns
    -------
    u_approx : [N]
    du_dx    : [N]
    du_dy    : [N]
    du_dt    : [N]
    lap_u    : [N]
    """
    x, y, t = X[:, 0], X[:, 1], X[:, 2]

    # ------------------------------------------------------------------
    # Precompute powers
    # ------------------------------------------------------------------

    x2 = x*x
    x3 = x2*x
    x4 = x2*x2
    x5 = x4*x
    x6 = x3*x3
    x7 = x6*x
    x8 = x4*x4
    x9 = x8*x
    x10 = x5*x5

    y2 = y*y
    y3 = y2*y
    y4 = y2*y2
    y5 = y4*y
    y6 = y3*y3
    y7 = y6*y
    y8 = y4*y4
    y9 = y8*y
    y10 = y5*y5

    t2 = t*t
    t3 = t2*t
    t4 = t2*t2
    t5 = t4*t
    t6 = t3*t3
    t7 = t6*t
    t8 = t4*t4
    t9 = t8*t
    t10 = t5*t5

    nu2  = nu**2
    nu3  = nu**3
    nu4  = nu**4
    nu5  = nu**5
    nu6  = nu**6
    nu7  = nu**7
    nu8  = nu**8
    nu9  = nu**9
    nu10 = nu**10

    # ------------------------------------------------------------------
    # sin(x) up to x^9
    # ------------------------------------------------------------------

    sin_x = (
        x
        - x3 / 6.0
        + x5 / 120.0
        - x7 / 5040.0
        + x9 / 362880.0
    )

    # ------------------------------------------------------------------
    # cos(y) up to y^10
    # ------------------------------------------------------------------

    cos_y = (
        1.0
        - y2 / 2.0
        + y4 / 24.0
        - y6 / 720.0
        + y8 / 40320.0
        - y10 / 3628800.0
    )

    # ------------------------------------------------------------------
    # exp(-2νt) up to t^10
    # ------------------------------------------------------------------

    exp_t = (
        1.0
        - 2.0*nu*t
        + 2.0*nu2*t2
        - (4.0/3.0)*nu3*t3
        + (2.0/3.0)*nu4*t4
        - (4.0/15.0)*nu5*t5
        + (4.0/45.0)*nu6*t6
        - (8.0/315.0)*nu7*t7
        + (2.0/315.0)*nu8*t8
        - (4.0/2835.0)*nu9*t9
        + (4.0/14175.0)*nu10*t10
    )

    # ------------------------------------------------------------------
    # Full approximation
    # ------------------------------------------------------------------

    u_approx = sin_x * cos_y * exp_t

    # ------------------------------------------------------------------
    # Derivatives
    # ------------------------------------------------------------------

    d_sin_dx = (
        1.0
        - x2 / 2.0
        + x4 / 24.0
        - x6 / 720.0
        + x8 / 40320.0
    )

    d_cos_dy = (
        -y
        + y3 / 6.0
        - y5 / 120.0
        + y7 / 5040.0
        - y9 / 362880.0
    )

    d_exp_dt = (
        -2.0*nu
        + 4.0*nu2*t
        - 4.0*nu3*t2
        + (8.0/3.0)*nu4*t3
        - (4.0/3.0)*nu5*t4
        + (8.0/15.0)*nu6*t5
        - (8.0/45.0)*nu7*t6
        + (16.0/315.0)*nu8*t7
        - (2.0/35.0)*nu9*t8
        + (8.0/1575.0)*nu10*t9
    )

    du_dx = d_sin_dx * cos_y * exp_t

    du_dy = sin_x * d_cos_dy * exp_t

    du_dt = sin_x * cos_y * d_exp_dt

    # ------------------------------------------------------------------
    # Second derivatives
    # ------------------------------------------------------------------

    d2_sin_dx2 = (
        -x
        + x3 / 6.0
        - x5 / 120.0
        + x7 / 5040.0
    )

    d2_cos_dy2 = (
        -1.0
        + y2 / 2.0
        - y4 / 24.0
        + y6 / 720.0
        - y8 / 40320.0
    )

    d2u_dx2 = d2_sin_dx2 * cos_y * exp_t

    d2u_dy2 = sin_x * d2_cos_dy2 * exp_t

    lap_u = d2u_dx2 + d2u_dy2

    return u_approx, du_dx, du_dy, du_dt, lap_u
# ============================================================================
# Generic Taylor-Green Taylor expansion generator
# ============================================================================

import math
import numpy as np


def taylor_u_order(X: np.ndarray, nu: float, order: int):
    """
    Taylor expansion of:

        u(x,y,t) = sin(x) cos(y) exp(-2νt)

    around (0,0,0), keeping terms up to the requested order.

    Parameters
    ----------
    X      : ndarray [N,3]
             columns = (x,y,t)

    nu     : float
             viscosity

    order  : int
             Taylor order (5,6,7,8,9,10,...)

    Returns
    -------
    u_approx
    du_dx
    du_dy
    du_dt
    lap_u
    """

    x, y, t = X[:, 0], X[:, 1], X[:, 2]

    # ------------------------------------------------------------------------
    # Build sin(x)
    # ------------------------------------------------------------------------

    sin_x = np.zeros_like(x)

    for n in range((order + 1) // 2):

        p = 2 * n + 1

        if p > order:
            break

        coeff = (-1.0) ** n / math.factorial(p)

        sin_x += coeff * x**p

    # ------------------------------------------------------------------------
    # Build cos(y)
    # ------------------------------------------------------------------------

    cos_y = np.zeros_like(y)

    for n in range(order // 2 + 1):

        p = 2 * n

        if p > order:
            break

        coeff = (-1.0) ** n / math.factorial(p)

        cos_y += coeff * y**p

    # ------------------------------------------------------------------------
    # Build exp(-2νt)
    # ------------------------------------------------------------------------

    exp_t = np.zeros_like(t)

    for n in range(order + 1):

        coeff = ((-2.0 * nu) ** n) / math.factorial(n)

        exp_t += coeff * t**n

    # ------------------------------------------------------------------------
    # Main approximation
    # ------------------------------------------------------------------------

    u_approx = sin_x * cos_y * exp_t

    # ------------------------------------------------------------------------
    # First derivative wrt x
    # ------------------------------------------------------------------------

    d_sin_dx = np.zeros_like(x)

    for n in range((order + 1) // 2):

        p = 2 * n + 1

        if p > order:
            break

        coeff = (-1.0) ** n / math.factorial(p)

        d_sin_dx += coeff * p * x**(p - 1)

    du_dx = d_sin_dx * cos_y * exp_t

    # ------------------------------------------------------------------------
    # First derivative wrt y
    # ------------------------------------------------------------------------

    d_cos_dy = np.zeros_like(y)

    for n in range(order // 2 + 1):

        p = 2 * n

        if p == 0:
            continue

        if p > order:
            break

        coeff = (-1.0) ** n / math.factorial(p)

        d_cos_dy += coeff * p * y**(p - 1)

    du_dy = sin_x * d_cos_dy * exp_t

    # ------------------------------------------------------------------------
    # First derivative wrt t
    # ------------------------------------------------------------------------

    d_exp_dt = np.zeros_like(t)

    for n in range(1, order + 1):

        coeff = ((-2.0 * nu) ** n) / math.factorial(n)

        d_exp_dt += coeff * n * t**(n - 1)

    du_dt = sin_x * cos_y * d_exp_dt

    # ------------------------------------------------------------------------
    # Second derivative wrt x
    # ------------------------------------------------------------------------

    d2_sin_dx2 = np.zeros_like(x)

    for n in range((order + 1) // 2):

        p = 2 * n + 1

        if p < 2:
            continue

        if p > order:
            break

        coeff = (-1.0) ** n / math.factorial(p)

        d2_sin_dx2 += coeff * p * (p - 1) * x**(p - 2)

    d2u_dx2 = d2_sin_dx2 * cos_y * exp_t

    # ------------------------------------------------------------------------
    # Second derivative wrt y
    # ------------------------------------------------------------------------

    d2_cos_dy2 = np.zeros_like(y)

    for n in range(order // 2 + 1):

        p = 2 * n

        if p < 2:
            continue

        if p > order:
            break

        coeff = (-1.0) ** n / math.factorial(p)

        d2_cos_dy2 += coeff * p * (p - 1) * y**(p - 2)

    d2u_dy2 = sin_x * d2_cos_dy2 * exp_t

    # ------------------------------------------------------------------------
    # Laplacian
    # ------------------------------------------------------------------------

    lap_u = d2u_dx2 + d2u_dy2

    return u_approx, du_dx, du_dy, du_dt, lap_u

def physics_residuals(u_approx, du_dx, du_dy, du_dt, lap_u,
                      v_known, dv_dy_known, pg_x_known, nu):
    """
    Divergence  : ∂u/∂x + ∂v/∂y
    x-Momentum  : ∂u/∂t + u·∂u/∂x + v·∂u/∂y − ν∇²u + ∂p/∂x
    """
    div_res = du_dx + dv_dy_known
    mom_res = (du_dt
               + u_approx * du_dx
               + v_known  * du_dy
               - nu       * lap_u
               + pg_x_known)
    return div_res, mom_res


def compute_errors(X, u_true, v_known, dv_dy_known, pg_x_known, nu):
    """
    Evaluate all error metrics for the 4th-order Taylor baseline.

    Returns
    -------
    dict with keys:
      u_approx  — predicted field [N]
      rmse_data — RMSE vs u_true
      rmse_div  — RMSE of divergence residual
      rmse_mom  — RMSE of x-momentum residual
      rmse_comb — equal-weight mean of the three RMSEs
      max_data, max_div, max_mom — maximum absolute residuals
      rel_data  — rmse_data / std(u_true)  (dimensionless)
    """
    #u_approx, du_dx, du_dy, du_dt, lap_u = taylor_u_4th(X, nu)
    u_approx, du_dx, du_dy, du_dt, lap_u = taylor_u_order(X, nu,30)
    div_res, mom_res = physics_residuals(
        u_approx, du_dx, du_dy, du_dt, lap_u,
        v_known, dv_dy_known, pg_x_known, nu)

    def rmse(r): return float(np.sqrt(np.mean(r ** 2)))

    r_data = rmse(u_approx - u_true)
    r_div  = rmse(div_res)
    r_mom  = rmse(mom_res)
    r_comb = (r_data + r_div + r_mom) / 3.0

    return dict(
        u_approx  = u_approx,
        rmse_data = r_data,
        rmse_div  = r_div,
        rmse_mom  = r_mom,
        rmse_comb = r_comb,
        max_data  = float(np.max(np.abs(u_approx - u_true))),
        max_div   = float(np.max(np.abs(div_res))),
        max_mom   = float(np.max(np.abs(mom_res))),
        rel_data  = r_data / float(np.std(u_true)) if np.std(u_true) > 0 else float('inf'),
    )


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main(n_points: int = 1_000_000, nu: float = 0.01):
    print(f"Generating {n_points:,} Taylor-Green points  (ν = {nu}) ...")
    X, y_list, v_known, dv_dy_known, pg_x_known = \
        generate_taylor_green_data_with_known(n_points, nu=nu)
    u_true = y_list[0]

    err = compute_errors(X, u_true, v_known, dv_dy_known, pg_x_known, nu)

    width = 62
    print()
    print("=" * width)

    print(f"  Domain   x,y ∈ [-1,1]   t ∈ [0,1]")
    print(f"  N points = {n_points:,}")
    print("=" * width)
    print(f"  {'Metric':<26} {'RMSE':>12}   {'Max |err|':>12}")
    print(f"  {'-'*26}   {'-'*12}   {'-'*12}")
    print(f"  {'Data  (u_approx − u_true)':<26} {err['rmse_data']:>12.6e}   {err['max_data']:>12.6e}")
    print(f"  {'Divergence  (∂u/∂x + ∂v/∂y)':<26} {err['rmse_div']:>12.6e}   {err['max_div']:>12.6e}")
    print(f"  {'x-Momentum':<26} {err['rmse_mom']:>12.6e}   {err['max_mom']:>12.6e}")
    print(f"  {'-'*26}   {'-'*12}   {'-'*12}")
    print(f"  {'Combined (mean)':<26} {err['rmse_comb']:>12.6e}")
    print()
    print(f"  Relative data RMSE : {err['rel_data']:.4f}  × std(u_true)")
    print("=" * width)


if __name__ == "__main__":
    main()
