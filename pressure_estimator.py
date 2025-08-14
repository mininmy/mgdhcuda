# pressure_estimator.py
import cupy as cp
import cudf
from gpu_polynomial_module import decode_keys, PolynomialGPU, Polynomial
from config_constants import MAX_EXP, PRUNE_THRESHOLD


def _ensure_gpu_poly(poly):
    """
    Ensure that the input is a PolynomialGPU.
    Converts from Polynomial (CPU) if necessary.
    """
    if isinstance(poly, PolynomialGPU):
        return poly
    elif isinstance(poly, Polynomial):
        return PolynomialGPU.from_polynomial(poly)
    else:
        raise TypeError("Input must be Polynomial or PolynomialGPU")


def sum_polynomials_gpu(polys_or_arrays):
    """
    Sum a list of PolynomialGPU objects or raw (exponents, coeffs) tuples/arrays.

    Parameters
    ----------
    polys_or_arrays : list
        Either:
          - List of PolynomialGPU objects
          - List of (exponents, coeffs) CuPy arrays

    Returns
    -------
    PolynomialGPU
        Combined polynomial with duplicate terms reduced.
    """
    if not polys_or_arrays:
        return PolynomialGPU(
            cp.zeros((0, 0), dtype=cp.uint8),
            cp.zeros((0,), dtype=cp.float64)
        )

    first = polys_or_arrays[0]
    if isinstance(first, PolynomialGPU):
        polys = [_ensure_gpu_poly(p) for p in polys_or_arrays]
        nvars = polys[0].exponents.shape[1]
        exponents_all = cp.concatenate([p.exponents for p in polys], axis=0)
        coeffs_all = cp.concatenate([p.coeffs for p in polys], axis=0)
    else:
        exps_list, coeffs_list = [], []
        for exps, coeffs in polys_or_arrays:
            if exps.shape[0] > 0:
                exps_list.append(exps)
                coeffs_list.append(coeffs)
        if not exps_list:
            return PolynomialGPU(
                cp.zeros((0, 0), dtype=cp.uint8),
                cp.zeros((0,), dtype=cp.float64)
            )
        exponents_all = cp.concatenate(exps_list, axis=0)
        coeffs_all = cp.concatenate(coeffs_list, axis=0)
        nvars = exponents_all.shape[1]

    if exponents_all.shape[0] == 0:
        return PolynomialGPU(exponents_all, coeffs_all)

    # Encode exponents to keys
    keys_combined = cp.zeros(exponents_all.shape[0], dtype=cp.uint64)
    base = 1
    for i in reversed(range(nvars)):
        keys_combined += exponents_all[:, i].astype(cp.uint64) * base
        base *= MAX_EXP

    # Reduce duplicate terms with cuDF
    df = cudf.DataFrame({'key': keys_combined, 'coeff': coeffs_all})
    reduced = df.groupby('key').agg({'coeff': 'sum'}).reset_index()

    decoded_exponents = decode_keys(reduced['key'].to_cupy(), nvars=nvars)
    return PolynomialGPU(decoded_exponents, reduced['coeff'].to_cupy()).prune(threshold=PRUNE_THRESHOLD)


def compute_pressure_from_polynomials(polynomials, viscosity, time_index=None):
    """
    Estimate scalar pressure polynomial from velocity polynomials using
    the incompressible Navier–Stokes equation.

    Parameters
    ----------
    polynomials : list of Polynomial or PolynomialGPU
        Velocity components u_1, u_2, ..., u_d
    viscosity : float
        Kinematic viscosity ν
    time_index : int or None
        Index of time variable in polynomials (if present).
        If None, the time derivative is skipped.

    Returns
    -------
    PolynomialGPU
        Estimated pressure polynomial (on GPU).
    """
    u_polys = [_ensure_gpu_poly(p) for p in polynomials]
    d = len(u_polys)
    nvars = u_polys[0].exponents.shape[1]
    pressure_candidates = []

    for i in range(d):
        u_i = u_polys[i]

        # Time derivative
        dt_poly = (u_i.differentiate(time_index) if time_index is not None
                   else PolynomialGPU(cp.zeros((0, nvars), dtype=cp.uint8),
                                       cp.zeros((0,), dtype=cp.float64)))

        # Convection term
        conv_terms = []
        for j in range(d):
            du_i_dj = u_i.differentiate(j)
            prod = u_polys[j].combine_with_gpu(du_i_dj, c0=0.0, c1=0.0, c2=0.0, c3=1.0)
            conv_terms.append(prod)
        convection = sum_polynomials_gpu(conv_terms)

        # Viscous term
        laplace_terms = [u_i.differentiate(j).differentiate(j) for j in range(d)]
        laplace = sum_polynomials_gpu(laplace_terms)
        viscous_term = PolynomialGPU(laplace.exponents, laplace.coeffs * float(viscosity))

        # Total derivative: dt + convection − viscous
        sum_terms = []
        if dt_poly.exponents.shape[0] > 0:
            sum_terms.append(dt_poly)
        if convection.exponents.shape[0] > 0:
            sum_terms.append(convection)
        if viscous_term.exponents.shape[0] > 0:
            visc_neg = PolynomialGPU(viscous_term.exponents, -viscous_term.coeffs)
            sum_terms.append(visc_neg)

        total_derivative = sum_polynomials_gpu(sum_terms)

        if total_derivative.exponents.shape[0] > 0:
            pressure_grad_i = PolynomialGPU(total_derivative.exponents, -total_derivative.coeffs)
            p_i = pressure_grad_i.integrate(i)
            pressure_candidates.append(p_i)

    # Weighted merge of pressure candidates
    all_exps, all_coeffs = [], []
    for p in pressure_candidates:
        if p.exponents.shape[0] == 0:
            continue

        exps, coeffs = p.exponents, p.coeffs
        if time_index is not None:
            mask = cp.ones(exps.shape[1], dtype=cp.bool_)
            mask[time_index] = False
            n_nonzero = cp.count_nonzero(exps[:, mask], axis=1)
        else:
            n_nonzero = cp.count_nonzero(exps, axis=1)

        valid_mask = n_nonzero > 0
        if not cp.any(valid_mask):
            continue

        all_exps.append(exps[valid_mask])
        all_coeffs.append(coeffs[valid_mask] / n_nonzero[valid_mask].astype(cp.float64))

    if not all_exps:
        return PolynomialGPU(cp.zeros((0, nvars), dtype=cp.uint8),
                              cp.zeros((0,), dtype=cp.float64))

    all_exps = cp.vstack(all_exps)
    all_coeffs = cp.concatenate(all_coeffs)

    pressure = sum_polynomials_gpu([(all_exps, all_coeffs)])
    return pressure.prune(threshold=PRUNE_THRESHOLD)
