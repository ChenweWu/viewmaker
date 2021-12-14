import numpy as np
import torch
from torch import nn, Tensor
from typing import Tuple
from .vcg import to_vcg, to_ecg

class _Augmentation(nn.Module):
  def __init__(self, no_op: bool) -> None:
    super().__init__()
    self._no_op = no_op

class _RandomVCGAugmentation(_Augmentation):
  def __init__(self, no_op: bool) -> None: super().__init__(no_op)

class _RandomECGAugmentation(_Augmentation):
  def __init__(self, no_op: bool) -> None: super().__init__(no_op)

class MultiRandomTransform(nn.Module):
  def __init__(self, nviews: int, args: Tuple[_Augmentation]) -> None:
    super().__init__()

    vcg = []
    ecg = []
    for arg in args:
      if   isinstance(arg, _RandomVCGAugmentation) and not arg._no_op: vcg.append(arg)
      elif isinstance(arg, _RandomECGAugmentation) and not arg._no_op: ecg.append(arg)
    self.vcg = nn.Sequential(*vcg)
    self.ecg = nn.Sequential(*ecg)
    self.nviews = nviews
  
  def forward(self, x: Tensor):
    with torch.no_grad():
      if len(x.shape) == 2: 
        x = x.unsqueeze(0)
      elif len(x.shape) != 3: 
        raise ValueError(f"Input tensor to {self.__class__.__name__} must be 2 or 3 dimensions.")
      return tuple(self._forward_impl(x).squeeze_(0) for _ in range(self.nviews))

  def _forward_impl(self, x: Tensor) -> Tensor:
    if self.vcg: x = to_ecg(self.vcg(to_vcg(x)))
    elif self.ecg: x = x.clone()
    
    return self.ecg(x)

class SingleRandomTransform(MultiRandomTransform):
  def __init__(self, *args: _Augmentation) -> None:
    super().__init__(1, args)

class DoubleRandomTransform(MultiRandomTransform):
  def __init__(self, *args: _Augmentation) -> None:
    super().__init__(2, args)



class RandomGaussian(_RandomECGAugmentation):
  def __init__(self, do_something: bool) -> None:
    super().__init__(not do_something)
  
  def forward(self, x: Tensor) -> Tensor:
    return x + torch.randn_like(x)

class RandomRotation(_RandomVCGAugmentation):
  R = np.array([
    lambda theta: torch.tensor([[1,             0,              0],
                                [0, np.cos(theta), -np.sin(theta)],
                                [0, np.sin(theta),  np.cos(theta)]]),
    lambda theta: torch.tensor([[np.cos(theta),  0, np.sin(theta)],
                                [0,              1,             0],
                                [-np.sin(theta), 0, np.cos(theta)]]),
    lambda theta: torch.tensor([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta),  np.cos(theta), 0],
                                [0,              0,             1]])
  ])

  def __init__(self, max_angle: float) -> None:
    super().__init__(np.isclose(max_angle, 0))
    self.max_angle = np.deg2rad(max_angle)
    self.angle_range = self.max_angle * 2

  def forward(self, x: Tensor) -> Tensor:
    return torch.matmul(self._get_rotations(len(x)), x)
  
  def _get_rotations(self, N: int) -> Tuple[Tensor, Tensor]:
    perm_fns = RandomRotation.R[np.vstack(tuple(np.random.permutation(3) for _ in range(N)))]
    thetas = torch.rand(N, 3).mul_(self.angle_range).sub_(self.max_angle)

    return torch.stack(tuple(
      map(lambda f, a: f[0](a[0]) @ f[1](a[1]) @ f[2](a[2]), perm_fns, thetas)))

class RandomScale(_RandomVCGAugmentation):
  def __init__(self, scale: float) -> None:
    super().__init__(np.isclose(scale, 1))
    self.scale = scale

  def forward(self, x: Tensor) -> Tensor:
    return torch.matmul(self._get_scales(len(x)), x)
  
  def _get_scales(self, N: int) -> Tensor:
    s = torch.from_numpy(np.random.uniform(1, self.scale, size=(N, 3))).float()
    mask = torch.rand_like(s) < 0.5
    s[mask] = 1 / s[mask]
    return torch.diag_embed(s)

class RandomChannelMask(_RandomECGAugmentation):
  def __init__(self, p: float) -> None:
    if p < 0 or p > 1: raise ValueError(f"p not within [0, 1]: saw {p}")
    super().__init__(np.isclose(p, 0))
    self.p = p
  
  def forward(self, x: Tensor) -> Tensor:
    x[self._get_mask(*x.shape[:2])]  = 0
    return x
  
  def _get_mask(self, N: int, C: int):
    mask = torch.zeros(N, C, dtype=torch.bool)
    for i, channels in enumerate(torch.multinomial(torch.ones(N, C), int(self.p * C))):
      mask[i, channels] = True
    return mask

class RandomTimeMask(_RandomECGAugmentation):
  def __init__(self, p: float) -> None:
    if p < 0 or p > 1: raise ValueError(f"p not within [0, 1]: saw {p}")
    super().__init__(np.isclose(p, 0))
    self.p = p

  def forward(self, x: Tensor) -> Tensor:
    N, C, L = x.shape

    mask_len = int(L * self.p)
    for n in range(N):
      for c in range(C):
        start = np.random.randint(L)
        stop = start + mask_len
        if stop >= L:
          mod = stop - L
          x[n, c, start:L]    = 0
          x[n, c,      :mod]  = 0
        else: 
          x[n, c, start:stop] = 0
    
    return x