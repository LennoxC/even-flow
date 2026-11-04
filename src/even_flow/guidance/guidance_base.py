import torch
from abc import ABC, abstractmethod

class GuidanceBase(ABC):
    def __init__(self, fm_model, vae_model=None, device=None):
        self.fm_model = fm_model
        self.vae_model = vae_model
        self.device = device

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
    
    # static data for the flow matching model (conditioning)
    # def load_static_data()

    # static data for the VAE model (decoder skip connections)
    # def load_static_skips()

    # normalization of the input data
    # def norm_data()

    # denormalization of the output data
    # def denorm_data()

    # normalization of the VAE latent space
    # def norm_latent()

    # denormalization of the VAE latent space
    # def denorm_latent()

    # Perform guided inference using the algorithm defined in the subclass
    @abstractmethod
    def inference(self):
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

class MaskBase(ABC):
    @abstractmethod
    def __init__(self, mask):
        self.mask = mask

    @abstractmethod
    def apply(self, x):
        pass