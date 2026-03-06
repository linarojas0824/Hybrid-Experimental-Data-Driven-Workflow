from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List, Union,Sequence

import numpy as np


@dataclass
class LifetimeFitter:
    n_samples: int = 5000
    k_best: int = 20
    seed: int = 0
    loss: str = "linear"  # or "soft_l1"
    scale_residuals: bool = True
    PARAM_NAMES = ["dt_t_nt", "tau_th", "tau_p", "dt_t_th", "dt_t_l"]

    @staticmethod
    def model(time, dt_t_nt, tau_th, tau_p, dt_t_th, dt_t_l):
        time = np.asarray(time, dtype=float)
        return (
            dt_t_nt * np.exp(-time * ((1 / tau_th) + (1 / tau_p)))
            + dt_t_th * (np.exp(-time / tau_p) * (1 - np.exp(-time / tau_th)))
            + dt_t_l
        )

    def objective(self, p, time, y_exp):
        r = self.model(time, *p) - y_exp
        return np.sum(r * r)

    def bounds_from_data(self, time: np.ndarray, y_exp: np.ndarray):
        y_scale = np.std(y_exp) if np.std(y_exp) > 0 else 1.0
        tpos = time[time > 0]
        if tpos.size == 0:
            raise ValueError("time must contain positive values to define tau bounds.")
        tmin_pos = np.min(tpos)
        tmax = np.max(time)

        return [
            (-10 * y_scale, 10 * y_scale),        # dt_t_nt
            (tmin_pos * 0.1, tmax * 10),          # tau_th
            (tmin_pos * 0.1, tmax * 10),          # tau_p
            (-10 * y_scale, 10 * y_scale),        # dt_t_th
            (np.min(y_exp), np.max(y_exp)),       # dt_t_l
        ]

    def find_initial_guesses(self, bounds, time, y_exp) -> List[np.ndarray]:
        rng = np.random.default_rng(self.seed)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        candidates = []
        for _ in range(self.n_samples):
            p = lo + (hi - lo) * rng.random(len(bounds))
            val = self.objective(p, time, y_exp)
            candidates.append((val, p))

        candidates.sort(key=lambda x: x[0])
        return [p for (_, p) in candidates[: self.k_best]]

    def fit(self, time: np.ndarray, y_exp: np.ndarray, bounds=None) -> np.ndarray:
        time = np.asarray(time, dtype=float)
        y_exp = np.asarray(y_exp, dtype=float)

        if bounds is None:
            bounds = self.bounds_from_data(time, y_exp)

        lb = np.array([b[0] for b in bounds], dtype=float)
        ub = np.array([b[1] for b in bounds], dtype=float)

        # initial candidates
        p0_list = self.find_initial_guesses(bounds, time, y_exp)

        # scaling (optional)
        scale = np.std(y_exp) if self.scale_residuals else 1.0
        scale = scale if scale > 0 else 1.0

        best_res = None
        best_cost = np.inf

        for p0 in p0_list:
            res = least_squares(
                fun=lambda p: (self.model(time, *p) - y_exp) / scale,
                x0=p0,
                bounds=(lb, ub),
                loss=self.loss,
            )
            if res.cost < best_cost:
                best_cost = res.cost
                best_res = res

        if best_res is None:
            raise RuntimeError("Fitting failed to produce a solution.")

        p_opt = best_res.x
        y_fit = self.model(time, *p_opt)

        residual = y_fit - y_exp
        sq_error = residual**2
        abs_error = np.abs(residual)

        mse = float(np.mean(sq_error))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(abs_error))
        sse = float(np.sum(sq_error))

        # optional R^2
        ss_tot = float(np.sum((y_exp - np.mean(y_exp))**2))
        r2 = float(1.0 - sse / ss_tot) if ss_tot > 0 else np.nan

        # named params (easy to print/save)
        params = dict(zip(self.PARAM_NAMES, p_opt))

        # return a bundle you can save + plot later
        return {
            "params": params,                   
            "y_fit": y_fit,
            "residual": residual,
            "sq_error": sq_error,
            "abs_error": abs_error,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "sse": sse,
            "r2": r2,

        }