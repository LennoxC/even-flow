import torch
import numpy as np
from even_flow.guidance.problem import GuidanceProblem
from typing import List, Literal, Optional, Union

class GaussNewtonSampler:
    """
    Solves using the Gauss-Newton method for data consistency.
    Renoising and exploration are inspired by the FLAIR paper (https://arxiv.org/abs/2506.02680)
    """

    def __init__(self, problem: GuidanceProblem, verbose=False):
            self.p = problem
            self.verbose = verbose

    def cg_solve(self, matvec, b, iters=8, tol=1e-4):
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
                  guidance_masked,
                  mask,
                  batch_size=1,
                  mu_x = None,
                  num_steps=50,
                  solve_method: Literal["cg", "inv"] = "cg",
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
                  max_step_frac = None,
                  unnorm_solve: bool = True,
                  returns: List[str] = ["z", "x"]):
        """
        Performs inference using the Gauss-Newton method.
        """

        B = batch_size  # batch size
        latent_shape = (B, *self.p.latent_shape)  # latent shape for the batch
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
            x = self.p.x1(x, t)
            dec = self.p.decode(x)
            if A is not None:
                dec = A(dec)
            return dec[:, :C].flatten()[obs_idx]

        def compute_var(r, s2):
            var = var0 * (1 + kappa * s2.sqrt().pow(2) / (1 - s2.sqrt()).clamp_min(0.1))
            if robust_nu is not None:
                var = var * (robust_nu + r ** 2 / var) / (robust_nu + 1)
            return var

        def clip_step(dx, xt):
            if max_step_frac is None:
                return dx
            max_norm = max_step_frac * xt.norm()
            return dx * torch.clamp(max_norm / (dx.norm() + 1e-12), max=1.0)  # no CPU sync

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
                var = compute_var(r, s2)
                matvec = lambda c: s2 * J(Jt(c)) + var * c
                c = self.cg_solve(matvec, r, iters=cg_iters)
                dx = clip_step((s2 * Jt(c)).view_as(xg), xg)
            return dx.detach(), r

        def build_G(xt, t, chunk=512):
            """Returns (h(xt), G) with G = dh/dx of shape (n_obs, D)."""
            with torch.enable_grad():
                xg = xt.detach().requires_grad_(True)
                y = h(xg, t)
                rows = []
                for i in range(0, n_obs, chunk):
                    j = min(i + chunk, n_obs)
                    E = torch.zeros(j - i, n_obs, device=xt.device, dtype=y.dtype)
                    E[torch.arange(j - i, device=xt.device), torch.arange(i, j, device=xt.device)] = 1.0
                    (gx,) = torch.autograd.grad(y, xg, grad_outputs=E,
                                                is_grads_batched=True, retain_graph=True)
                    rows.append(gx.flatten(1))
            return y.detach(), torch.cat(rows)  # (n_obs,), (n_obs, D)

        def gn_inv(xt, t, s2):
            y0, G = build_G(xt, t)
            with torch.no_grad():
                r = y_target - y0                      # residual at xt, NOT G @ xt
                var = compute_var(r, s2)
                Amat = s2 * (G @ G.T)
                Amat.diagonal().add_(var)
                L, info = torch.linalg.cholesky_ex(Amat)
                if info.item() == 0:
                    c = torch.cholesky_solve(r.unsqueeze(-1), L).squeeze(-1)
                else:                                   # fallback if not numerically PD
                    c = torch.linalg.solve(Amat, r)
                dx = clip_step((s2 * (G.T @ c)).view_as(xt), xt)
            return dx.detach(), r

        step_fn = gn_inv if solve_method == "inv" else gn

        # the actual iterative FLAIR/Gauss Newton loop
        for i, step in enumerate(reversed(range(num_steps))):
            s = torch.tensor([(step + 0.5) / num_steps], device=self.p.device, dtype=torch.float32)
            t = 1 - s

            xt = t * mu_x + s * e_hat
            s2 = prior_scale * s.pow(2)
            n_it = final_gn_iters if step == 0 else gn_iters

            for k in range(n_it):
                dx, r = step_fn(xt, t, s2)
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

        # final denormalization and decoding
        return_dict = {}
        z = self.p.denorm_latent(mu_x) # returned as the denormalized latent
        if "z" in returns:
            return_dict["z"] = z
        if "x" in returns:
            with torch.no_grad():
                x_decoded = self.p.decode(mu_x)
                if unnorm_solve:
                    x_decoded = self.p.denorm_data(x_decoded)
            return_dict["x"] = x_decoded

        return return_dict