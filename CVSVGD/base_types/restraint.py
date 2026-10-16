from typing import Protocol

import torch


class Restraint(Protocol):
    @staticmethod
    def force(x:torch.Tensor,x0:torch.Tensor,length_scale:float):
        raise NotImplementedError

class HarmonicRestraint(Restraint):
    @staticmethod
    def force(x:torch.Tensor,x0:torch.Tensor,length_scale:float):
        return -2 * ((x-x0)/length_scale)
