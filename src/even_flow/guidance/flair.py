import torch
from even_flow.guidance.problem import GuidanceProblem

class FLAIRSampler:
    """
    Based on the FLAIR paper (https://arxiv.org/abs/2506.02680)
    Initialize this from a Guidance class, which should implement the abstract methods in GuidanceBase.
    The FLAIRSampler class implements the inference method, which performs the FLAIR guidance.
    """
    def __init__(self, problem: GuidanceProblem, verbose=False):
        self.p = problem
        self.verbose = verbose

    # this can be overridden to implement different guidance strategies
    def a(self, t):
        return 1 - t

    def distance(self, x_decoded, g, m):
        c = x_decoded.shape[1] # number of channels
        guidance, mask = g[:c], m[:c].bool()
        diff = torch.where(mask, x_decoded - guidance, torch.zeros_like(x_decoded))
        return torch.sqrt(torch.sum(diff ** 2))

    def hard_data_consistency(self, 
                              mu_x, 
                              g, 
                              m, 
                              epoch,
                              iter=10, 
                              lr=lambda s: 0.001, 
                              static_skips=None, 
                              **kwargs):
        vae_model = self.p.vae_model # on device and in eval mode
        mu_x = mu_x.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([mu_x], lr=lr(epoch))
        losses = []

        for i in range(iter):
            optimizer.zero_grad()
            z = self.p.denorm_latent(mu_x)
            x_decoded = vae_model.decode(z, static_skips=static_skips) if static_skips is not None else vae_model.decode(z)
            
            loss = self.distance(x_decoded, g, m)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

        if self.verbose:
            print(f"Epoch {epoch}, Loss start: {losses[0]:.6f}, Loss end: {losses[-1]:.6f}, reduction: {losses[0] - losses[-1]:.6f}")

        return mu_x.detach(), losses
        
    def inference(self, 
                  latent_shape, 
                  guidance_masked, 
                  mask, 
                  mu_x = None, 
                  lr=lambda s: 0.001, 
                  num_steps=50, 
                  dc_iter=10,
                  returns=["z", "x_unnorm", "x_decoded"],
                  **kwargs):
        """
        Performs inference using the FLAIR guidance.
        Args:
            latent_shape: The shape of the latent space.
            guidance_masked: The guidance mask.
            mask: The mask for the input data.
            mu_x: Optional initial guess for x0. If None, a random tensor is used.
            lr: Learning rate function that takes a step index and returns a learning rate.
            num_steps: Number of steps for the inference process.
            dc_iter: Number of iterations for the data consistency step.
        Returns:
            The inferred latent representation after applying FLAIR guidance.
        Kwargs:
            clamp_min_value: Minimum value for clamping during the inference process.
        """
        device = self.p.device
        
        vae_model = self.p.vae_model # on device and in eval mode
        
        static = self.p.static_data # should have moved static to device
        static_skips = self.p.static_skips # should have moved static_skips to device

        clamp_min_value = kwargs.get("clamp_min_value", 1e-3)

        e_hat = torch.randn(latent_shape, device=device) # initial noise
        mu_x = torch.randn(latent_shape, device=device) if mu_x is None else mu_x.to(device) # initial guess for x0

        B = latent_shape[0]

        g = torch.tensor(guidance_masked, device=device, dtype=torch.float32)
        m = torch.tensor(mask, device=device, dtype=torch.float32)

        for _, step in enumerate(reversed(range(num_steps))):
            s = torch.tensor([(step + 0.5) / num_steps], device=device, dtype=torch.float32)
            t = 1 - s

            # find xt, ut
            xt = t * mu_x + s * e_hat
            ut = (e_hat - xt) / t.clamp_min(clamp_min_value)

            with torch.no_grad():
                x1_hat = self.p.x1(xt, t)

            v_theta = (xt - x1_hat) / s
            delta_r = v_theta - ut

            mu_x = mu_x - s * delta_r
            mu_x, losses = self.hard_data_consistency(mu_x, g, m, step, iter=dc_iter, lr = lr, static_skips=static_skips)

            e = torch.randn(latent_shape, device=device)

            x_hat_1 = xt + t * v_theta
            e_hat = self.a(s) * x_hat_1 + torch.sqrt(1 - self.a(s) ** 2) * e

        # unnorm and decode
        z = self.p.denorm_latent(mu_x)
        x_decoded = vae_model.decode(z, static_skips=static_skips) if static_skips is not None else vae_model.decode(z)
        x_unnorm = self.p.denorm_data(x_decoded)

        if "z" in returns:
            z = self.p.denorm_latent(mu_x)
        if "x_unnorm" in returns:
            x_unnorm = self.p.denorm_data(x_decoded)
        if "x_decoded" in returns:
            x_decoded = vae_model.decode(z, static_skips=static_skips) if static_skips is not None else vae_model.decode(z)

        return {
            "z": z,
            "x_decoded": x_decoded,
            "x_unnorm": x_unnorm,
        }