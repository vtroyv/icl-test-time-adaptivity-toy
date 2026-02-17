#minimal, practical "sample functions + (x,y) points" module

import torch 
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional 

@dataclass 
class FourierTask: 
    """One sampled function f(x) on [0,1], represented by its fourier coefficients"""
    K: int
    a: torch.Tensor #[k]
    b: torch.Tensor #[k]
    
    def f(self, x:torch.Tensor) -> torch.Tensor: 
        """
        Args:
            x : [...], assumed in [0,1]
        Returns:
            [...], same shape 
        """
        ks = torch.arange(1, self.K+1, device=x.device, dtype=x.dtype)
        ang = 2* torch.pi * x.unsqueeze(-1) * ks
        return (self.a * torch.sin(ang) + self.b * torch.cos(ang)).sum(dim=-1)
    
class TaskSampler:
    """
    Samples tasks on the fly:
    - sample a function f (Fourier series truncated at K)
    - sample n contet points (x_i, y_i)
    - sample a query x_q and target y_q
    Reproducible if you pass a fxied seed. 
    
    """
    def __init__(
        self, 
        K_choices: Tuple[int,...] = (3,6,12,24),
        coef_decay: float = 1.0, 
        noise_std: float = 0.01, 
        n_min: int = 8, 
        n_max: int =64, 
        seed: Optional[int] = 0, 
        device: str = "cpu", 
        dtype: torch.dtype = torch.float32, 
        shuffle_context: bool =True,
    ): 
        self.K_choices = torch.tensor(K_choices)
        self.coef_decay = coef_decay
        self.noise_std = noise_std
        self.n_min = n_min
        self.n_max = n_max
        self.device =torch.device(device)
        self.dtype = dtype
        self.shuffle_context = shuffle_context
        
    #Use a dedicated RNG so calls are reproducible and self-contained 
        self.g = torch.Generator(device="cpu") #generator lives on CPU
        if seed is not None:
            self.g.manual_seed(seed)
            
    
    def sample_function(self) -> FourierTask:
        K= int(self.K_choices[torch.randint(0, len(self.K_choices), (1, ), generator=self.g)].item())
        k = torch.arange(1, K +1, dtype =self.dtype)    #[K]
        scale = 1.0/ (k ** self.coef_decay)             #[K]
        a = torch.randn(K,generator=self.g, dtype=self.dtype) * scale
        b = torch.randn(K,generator=self.g, dtype=self.dtype) * scale
        
        return FourierTask(K=K, a=a.to(self.device), b=b.to(self.device))
    
    def _randn_like(self, x: torch.Tensor) -> torch.Tensor:
    # torch.randn_like may not accept generator on some builds
        return torch.randn(
        x.shape,
        generator=self.g,
        device=x.device,
        dtype=x.dtype,
    )
    
    def sample_task(self) -> Dict[str, torch.Tensor]:
        """
        Returns one training example:
            Shapes (1D x): 
            x_ctx: [n]
            y_ctx: [n]
            x_q: [1]
            y_q: [1]
            K: [] (scalar metadata for analysis)
            
        """
        task = self.sample_function()
        
        n = int(torch.randint(self.n_min, self.n_max+1, (1, ), generator = self.g).item())
        x_ctx = torch.rand(n, generator= self.g, dtype=self.dtype).to(self.device) #[n]
        y_ctx = task.f(x_ctx)
        
        x_q = torch.rand(1, generator=self.g, dtype = self.dtype).to(self.device) #[1]
        y_q = task.f(x_q)
        
        if self.noise_std > 0:
            y_ctx = y_ctx + self.noise_std * self._randn_like(y_ctx)
            y_q   = y_q   + self.noise_std * self._randn_like(y_q)
            
        if self.shuffle_context:
            perm = torch.randperm(n, generator=self.g, device=self.device)
            x_ctx, y_ctx = x_ctx[perm], y_ctx[perm]
            
        return {"x_ctx": x_ctx, "y_ctx": y_ctx, "x_q": x_q, "y_q": y_q, "K": torch.tensor(task.K)}
    
    
    def sample_eval_pool(self,M:int=128) -> Dict[str, torch.Tensor]:
        """
        Returns a fixed pool for ONE function so you can vary context size fairly.
            Shapes: 
            x_pool: [M]
            y_pool: [M]
            K: [] (scalar metadata)
        """
        
        task = self.sample_function()
        x_pool = torch.rand(M, generator=self.g, dtype=self.dtype).to(self.device) #[M]
        y_pool = task.f(x_pool)
        
        if self.noise_std > 0:
            y_pool = y_pool + self.noise_std * self._randn_like(y_pool)
            
        #sort optional for nicer plots/debugging
        order = torch.argsort(x_pool)
        x_pool, y_pool = x_pool[order],y_pool[order]
        
        return {"x_pool": x_pool, "y_pool": y_pool, "K": torch.tensor(task.K)}
    
    
    
    
    
if __name__ == "__main__":
    sampler = TaskSampler(seed=123, n_min=4, n_max=8, noise_std=0.8)
    
    for t in range(3):
        ex= sampler.sample_task()
        print(f"\nExample {t}: K={int(ex['K'])} , n={ex['x_ctx'].numel()}")
        print("x_ctx:", ex["x_ctx"][:3])
        print("y_ctx:", ex["y_ctx"][:3])
        print("x_q:", ex["x_q"], " y_q:", ex["y_q"])
        
    # Evaluation pool: one function, many points
    pool = sampler.sample_eval_pool(M=16)
    print(f"\nEval pool: K={int(pool['K'])}, M={pool['x_pool'].numel()}")
    print("x_pool:", pool["x_pool"][:5])
    print("y_pool:", pool["y_pool"][:5])

    # Vary context size using the same function pool
    for n in [4, 8, 12]:
        x_ctx = pool["x_pool"][:n]
        y_ctx = pool["y_pool"][:n]
        x_q = pool["x_pool"][n:n+1]
        y_q = pool["y_pool"][n:n+1]
        print(f"\nFrom same function: context n={n}")
        print("  x_q:", x_q.item(), " y_q:", y_q.item())
        
        
        
        
        
    
        