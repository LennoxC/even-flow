import torch
import numpy as np
from even_flow.guidance.problem import GuidanceProblem
from typing import List, Optional, Union

class GaussNewtonSampler:
    """
    Solves using the Gauss-Newton method for data consistency.
    Renoising and exploration are inspired by the FLAIR paper (https://arxiv.org/abs/2506.02680)
    """

    def __init__(self, problem: GuidanceProblem, verbose=False):
            self.p = problem
            self.verbose = verbose

    def cg_solve(self, matvec, b, iters=8, tol=1e-3):
        """
        Solve Ax = b using the conjugate gradient method, where A is represented by the matvec function.
        """
        x = torch.zeros_like(b)
        r = b.clone(); p = r.clone(); rs = r @ r
        for _ in range(iters):
            Ap = matvec(p)
            a = rs / (p @ Ap + 1e-20)
            x = x + a * p
            r = r - a * Ap
            rs_new = r @ r
            if rs_new.sqrt() < tol * b.norm():
                break
            p = r + (rs_new / rs) * p
            rs = rs_new
        return x

    def inference(self, 
                  latent_shape,
                  guidance_masked,
                  mask,
                  mu_x = None,
                  num_steps=50,
                  sigma_obs: Union[float, torch.Tensor] = 0.05,
                  sigma_A: Union[float, torch.Tensor] = 0.0,
                  prior_scale: float = 1.0,
                  kappa: float = 0.0,
                  robust_nu: Optional[float] = None,
                  gn_iters: int = 1,
                  final_gn_iters: int = 2,
                  eta: float = 0.0,
                  A=None,
                  cg_iters: int = 8,
                  fd_eps: float = 1e-3,
                  max_step_frac = None):
        """
        Performs inference using the Gauss-Newton method.
        """

        C = np.shape(mask)[1] # number of channels
        g = torch.as_tensor(guidance_masked, device=self.p.device, dtype=torch.float32)[:, :C]
        m = torch.as_tensor(mask, device=self.p.device, dtype=torch.float32)[:, :C] > 0.5
        obs_idx = m.flatten().nonzero(as_tuple=True)[0]
        n_obs = obs_idx.numel()
        assert n_obs > 0, "Mask selects no observations. Check channel count and mask shape."
        y_target = g.flatten()[obs_idx]

        s_obs = torch.as_tensor(sigma_obs, device=self.p.device, dtype=torch.float32)
        s_A = torch.as_tensor(sigma_A, device=self.p.device, dtype=torch.float32)
        if s_obs.shape == torch.Size([]):
            s_obs = s_obs.expand(n_obs)
        if s_A.shape == torch.Size([]):
            s_A = s_A.expand(n_obs)

        var0 = (s_obs ** 2 + s_A ** 2).clamp(min=1e-8)

        e_hat = torch.randn(latent_shape, device=self.p.device)
        if mu_x is None:
            mu_x = torch.randn(latent_shape, device=self.p.device)
        mu_x = mu_x.detach()

        def h(x, t):
            z = self.p.denorm_latent(x)
            dec = self.p.vae_model.decode(z, self.p.static_skips) if self.p.static_skips is not None else self.p.vae_model.decode(z)
            if A is not None:
                dec = A(dec)
            return dec[:, :C].flatten()[obs_idx]

        def gn(xt, t, s2):

            with torch.enable_grad():
                xg = xt.detach().requires_grad_(True)
                y_pred = h(xg, t)
            y0 = y_pred.detach()

            def Jt(u):
                (gx,) = torch.autograd.grad(y_pred, xg, grad_outputs=u, retain_graph=True)
                return gx.flatten()

            def J(v):
                with torch.no_grad():
                    eps = fd_eps * xg.norm() / (v.norm() + 1e-12)
                    return (h(xg.detach() + eps * v.view_as(xg), t) - y0) / eps

            with torch.no_grad():
                r = y_target - y0
                var = var0 * (1 + kappa * s2.sqrt().pow(2) / (1 - s2.sqrt()).clamp_min(0.1))
                if robust_nu is not None:
                    var = var * (robust_nu + r ** 2 / var) / (robust_nu + 1)

                matvec = lambda c: s2 * J(Jt(c)) + var * c
                c = self.cg_solve(matvec, r, iters=cg_iters)
                dx = (s2 * Jt(c)).view_as(xg)

                if max_step_frac is not None:
                    max_norm = max_step_frac * xg.norm()
                    if dx.norm() > max_norm:
                        dx = dx * (max_norm / dx.norm())
            
            return dx.detach(), r

        # the actual iterative FLAIR/Gauss Newton loop
        for i, step in enumerate(reversed(range(num_steps))):
            s = torch.tensor([(step + 0.5) / num_steps], device=self.p.device, dtype=torch.float32)
            t = 1 - s

            xt = t * mu_x + s * e_hat
            s2 = prior_scale * s.pow(2)
            n_it = final_gn_iters if step == 0 else gn_iters

            for k in range(n_it):
                dx, r = gn(xt, t, s2)
                xt = xt.detach() + dx

                if self.verbose:
                    print(f"Step {i}.{k} s={s.item():.3f}  raw|r|={r.norm().item():.4f}  "
                      f"whitened rms={(r / var0.sqrt()).pow(2).mean().sqrt().item():.3f}  "
                      f"|dx|/|xt|={(dx.norm() / xt.norm()).item():.4f}")

            with torch.no_grad():
                x1_hat = self.p.x1(xt, t)
                eps_hat = (xt - t * x1_hat) / s
                if eta > 0:
                    eps_hat = (1 - eta ** 2) ** 0.5 * eps_hat + eta * torch.randn_like(eps_hat)

                mu_x = x1_hat
                e_hat = eps_hat

        z = self.p.denorm_latent(mu_x)
        with torch.no_grad():
            x_decoded = self.p.vae_model.decode(z, self.p.static_skips) if self.p.static_skips is not None else self.p.vae_model.decode(z)
            x_unnorm = self.p.denorm_data(x_decoded)

        return {
            "z": z,
            "x_decoded": x_decoded,
            "x_unnorm": x_unnorm
        }
