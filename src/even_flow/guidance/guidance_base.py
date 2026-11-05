import torch
import numpy as np
from scipy.interpolate import griddata
from abc import ABC, abstractmethod

class GuidanceBase(ABC):
    """
    Base class for guidance methods in flow matching models.
    
    To use: create a subclass that implements the abstract methods defined below.
    You can then initialize a Guidance method (e.g. FLAIRSampler) from the subclass.
    """
    def __init__(self, fm_model, vae_model=None, device=None, verbose=False):
        self.fm_model = fm_model
        self.vae_model = vae_model
        self.device = device
        self.verbose = verbose

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def setup(self, fm_weights_path=None, vae_weights_path=None):
        # if evaluating inference speed, the weights might not be needed (the model might not be trained yet)
        if fm_weights_path:
            self.fm_model.load_state_dict(torch.load(fm_weights_path, map_location=self.device))
        
        self.fm_model.to(self.device)
        self.fm_model.eval()

        # if evaluating inference speed, the weights might not be needed (the model might not be trained yet)
        if self.vae_model:
            if vae_weights_path:
                self.vae_model.load_state_dict(torch.load(vae_weights_path, map_location=self.device))
            
            self.vae_model.to(self.device)
            self.vae_model.eval()

    def interpolate_mu_x(self, guidance_masked, mask):
        vae = self.vae_model
        guidance_interp = np.copy(guidance_masked)
        guidance_interp = np.nan_to_num(guidance_interp, nan=0.0)
        # use a simple linear interpolation to fill in the NaN values
        x = np.arange(guidance_interp.shape[2])
        y = np.arange(guidance_interp.shape[1])
        xv, yv = np.meshgrid(x, y)
        guidance_interp = np.zeros_like(guidance_masked)
        for i in range(guidance_masked.shape[0]):
            points = np.column_stack((xv[mask[i] == 1], yv[mask[i] == 1]))
            values = guidance_masked[i][mask[i] == 1]
            grid_z = griddata(points, values, (xv, yv), method='linear', fill_value=0.0)
            guidance_interp[i] = grid_z

        vae.eval()
        with torch.no_grad():
            guidance_interp_tensor = torch.tensor(guidance_interp).unsqueeze(0).to(torch.float32).to(device)
            mu, _ = vae.encoder(guidance_interp_tensor)

        return mu


    """
    Abstract methods that must be implemented by subclasses:
        - many of these methods are application specific.
        - implementation is left to the user as the guidance framework is independent of the specific application.
    """
    
    # static data for the flow matching model (conditioning)
    @abstractmethod
    def load_static_data(self):
        pass

    @abstractmethod
    def load_static_skips(self):
        pass

    # normalization of the input data
    @abstractmethod
    def norm_data(self, x):
        pass

    # denormalization of the output data
    @abstractmethod
    def denorm_data(self, x):
        pass

    # normalization of the VAE latent space
    @abstractmethod
    def norm_latent(self, x):
        pass

    # denormalization of the VAE latent space
    @abstractmethod
    def denorm_latent(self, x):
        pass

    # setup function: load models, set device, etc. Must set self.setup = True if setup is successful
    # called before inference to ensure that norm params, models etc are loaded
    @abstractmethod
    def setup(self):
        # self.setup = True
        pass

    # Perform guided inference using the algorithm defined in the subclass
    @abstractmethod
    def inference(self):
        pass

    @abstractmethod
    def x1(self, x, t, static=None, **kwargs):
        pass

    @abstractmethod
    def v(self, x, t, **kwargs):
        pass

    # pass inference parameters as kwargs
    # evaluates the average inference time over a specified number of runs
    # doesn't required trained models, so can evaluate inference speed using guidance before training
    def benchmark_inference(self, num_runs=1, **kwargs):
        import time

        durations_s = []
        for _ in range(num_runs):
            start_time = time.time()
            self.inference(**kwargs)
            end_time = time.time()
            durations_s.append(end_time - start_time)

        avg_time = sum(durations_s) / len(durations_s)

        print(f"Average inference time over {num_runs} runs: {avg_time:.4f} seconds")
        return avg_time