import torch
import torch.nn as nn
from collections import OrderedDict
from zytlib.utils.classfunc import save_args
from zytlib import vector, table
from zytlib.utils.wrapper import registered_property
import random
from torchfunction.lossfunc import softmax_cross_entropy_with_logits_sparse
from torchfunction.select import select_index_by_batch
import math
from torchfunction.random import randn_seed
from torch import einsum
import numpy as np
from torch.distributions import Categorical
from typing import Union, List, Optional, Tuple, Dict, Any

def _copy(target: torch.Tensor, source: torch.Tensor)->None:
    assert target.dim() == source.dim() and target.dim() in [2, 3]
    if target.dim() == 3:
        assert target.shape[0] == source.shape[0]
        assert target.shape[1] >= source.shape[1]
        assert target.shape[2] >= source.shape[2]
        target[:, :source.shape[1], :source.shape[2]] = source
    else:
        assert target.shape[0] == source.shape[0]
        assert target.shape[1] >= source.shape[1]
        target[:, :source.shape[1]] = source

def logcosh(x: torch.Tensor) -> torch.Tensor:
    return x.cosh().log()

def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.relu(x)

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.sigmoid(x)

def softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x)

def tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.tanh(x)

def retanh(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x.tanh())

def square_root(x: torch.Tensor) -> torch.Tensor:
    return 2 * (torch.sqrt(torch.relu(x) + 1) - 1)

def channel_ind_noise(shape: torch.Size, device: torch.device, sigma: Union[float, str], alpha: float)->torch.Tensor:
    if isinstance(sigma, (int, float)) and sigma > 0:
        return torch.randn(shape, device=device) * sigma * math.sqrt(2 / alpha)
    elif isinstance(sigma, str) and len(sigma) and ("/" in sigma or float(sigma) > 0):
        if "/" in sigma:
            hn = torch.tensor(vector(sigma).split("/").map(float)).to(device)
            return torch.einsum("...r,r->...r", torch.randn(shape, device=device), hn) * math.sqrt(2 / alpha)
        else:
            return torch.randn(shape, device=device) * float(sigma) * math.sqrt(2 / alpha)
    else:
        return torch.zeros(shape, device=device)

def invsqrt(c):
    U, S, Vh = torch.linalg.svd(c)
    return torch.einsum("...ij,...j,...jk->...ik", U, 1 / S.sqrt(), Vh)

def standard_normal(*shape, device="cpu"):
    assert len(shape) >= 2
    x = torch.randn(*shape, device=device)
    x = x - x.mean(-2, keepdim=True)
    cov = torch.einsum("...ij,...ik->...jk", x, x) / x.shape[-2]
    invsqrt_cov = invsqrt(cov)
    return torch.einsum("...ij,...jk->...ik", x, invsqrt_cov)

def standard_normal_like(x):
    return standard_normal(*x.shape, device=x.device)

class encoder(nn.Module):

    def __init__(self, **kwargs: Union[int, float, str, None]):
        super().__init__()
        self.hyper = table()
        self.property = table()
        self.phi = eval(str(kwargs.get("phi", "torch.tanh")))

    def recurrence(self, fr: torch.Tensor, input: Union[torch.Tensor, None]=None, rank_mask=None)->torch.Tensor:
        raise NotImplementedError()

    def pre_forward(self)->None:
        pass

    def init_state(self, batch_size: int) -> torch.Tensor:
        N = self.x_dim()
        return torch.zeros(batch_size, N)

    def forward(self, input_tensor: torch.Tensor, init_state=None, **kwargs: float)->torch.Tensor:
        """
        input_tensor.shape: [batch, time, input_dim]
        output:

        final_state: [batch, N]
        hidden_state: [batch, T + 1, N]
        """

        self.pre_forward()

        hyper = self.hyper.copy().update_exist(kwargs)
        assert input_tensor.shape[-1] == hyper.input_dim

        batch_size, T, input_dim = input_tensor.shape

        N = self.x_dim()

        hidden_state = torch.zeros(T + 1, batch_size, N, device=input_tensor.device)
        if init_state is None:
            hidden_state[0, :, :] = self.init_state(batch_size).type_as(hidden_state)
        else:
            hidden_state[0, :, :] = init_state.type_as(hidden_state)

        if kwargs.get("random_seed", -1) != -1:
            torch.manual_seed(kwargs["random_seed"])

        if kwargs.get("noise_rec", 0) > 0:
            noise_rec = torch.randn([batch_size, T, N], device=input_tensor.device) *  hyper.noise_rec * math.sqrt(2 * hyper.tau / hyper.delta_t)
        elif self.training and hyper.noise_rec > 0:
            noise_rec = torch.randn([batch_size, T, N], device=input_tensor.device) *  hyper.noise_rec * math.sqrt(2 * hyper.tau / hyper.delta_t)
        else:
            noise_rec = torch.zeros([batch_size, T, N], device=input_tensor.device)

        if kwargs.get("random_seed", -1) != -1:
            torch.manual_seed(kwargs["random_seed"])

        if kwargs.get("noise_inp", 0):
            _noise_inp = kwargs["noise_inp"]
        elif self.training and hyper.noise_inp:
            _noise_inp = hyper.noise_inp
        else:
            _noise_inp = 0.0

        noise_inp = channel_ind_noise(input_tensor.shape, input_tensor.device, _noise_inp, hyper.tau / hyper.delta_t)

        rank_mask = kwargs.get("rank_mask", None)
        for index in range(T):
            manipulated_input = self.recurrence(self.phi(hidden_state[index, :, :].clone()), input_tensor[:, index, :] + noise_inp[:, index, :], rank_mask=rank_mask) + noise_rec[:, index, :]
            hidden_state[index + 1, :, :] = (1 - self.property.alpha) *  hidden_state[index, :, :] + self.property.alpha * manipulated_input

        hidden_state = hidden_state.transpose(0, 1) # [batch_size, T+1, N]

        return hidden_state

    def step(self, hidden_state: torch.Tensor, step: int=1, noise_inp: float=0.0, noise_rec: float=0.0)->torch.Tensor:
        """
        hidden_state.shape: [batch_size, N]

        output: [batch_size, step, N]
        """
        batch_size, N = hidden_state.shape
        ret = torch.zeros([batch_size, step, N], device=hidden_state.device)
        pre = hidden_state

        noise_inp = channel_ind_noise(torch.Size([batch_size, self.hyper.input_dim]), hidden_state.device, noise_inp, self.hyper.tau / self.hyper.delta_t)
        noise_rec = torch.randn_like(pre) * noise_rec *  math.sqrt(2 * self.hyper.tau / self.hyper.delta_t)

        for index in range(step):
            manipulated_input = self.recurrence(self.phi(pre), input=noise_inp) + noise_rec
            pre = (1 - self.property.alpha) * pre + self.property.alpha * manipulated_input
            ret[:, index, :] = pre
        return ret

    def velocity(self, hidden_state: torch.Tensor, input: Optional[torch.Tensor]=None)->torch.Tensor:
        """
        hidden_state: [batch, N] or [batch, p, N]
        """

        return -hidden_state + self.recurrence(self.phi(hidden_state), input)

    def outputs(self, x: torch.Tensor)->torch.Tensor:
        return x

    def x_dim(self)->int:
        return self.hyper.encoder_dim

    def parameters_lr(self)->list:
        return [{"params": self.parameters()}]

    def extra_saved(self)->dict:
        return dict()

    def load_extra(self, x: dict)->None:
        return

class fullrank_encoder(encoder):

    def __init__(self, **kwargs: Union[int, float, str, None]):
        super().__init__(**kwargs)

        self.hyper = table(tau=100,
                delta_t = 10,
                encoder_dim = 4096,
                noise_rec = 0.05,
                noise_inp = 0.01,
                input_dim = 1,
                input_gain = 1.0,
                output_dim = 1,
                init_gain = 1.0,
                fixed_I = False,
                fixed_W = False,
                has_bias = False,
                readout_radius = 1.0,
                phi = "torch.tanh",
                task = "",
                device = None)

        self.hyper.update_exist(kwargs)
        if len(kwargs) != len(self.hyper):
            raise TypeError(table(kwargs).key_not_here(self.hyper))
        self.property = table()
        self.property.alpha = self.hyper.delta_t / self.hyper.tau
        self.I = nn.Parameter(torch.zeros(self.hyper.encoder_dim, self.hyper.input_dim, device=self.hyper.device))
        self.J = nn.Parameter(torch.zeros(self.hyper.encoder_dim, self.hyper.encoder_dim, device=self.hyper.device))
        self.W = nn.Parameter(torch.zeros(self.hyper.encoder_dim, self.hyper.output_dim, device=self.hyper.device))
        if self.hyper.has_bias:
            self.B = nn.Parameter(torch.zeros(self.hyper.encoder_dim, device=self.hyper.device))
        assert 0 < self.property.alpha <= 1
        assert self.hyper.input_dim > 0
        assert self.hyper.encoder_dim > 0
        assert self.hyper.output_dim > 0

        self.trainable_mask = table()
        self.trainable_mask.I = torch.ones(self.hyper.encoder_dim, self.hyper.input_dim, device=self.hyper.device)
        self.trainable_mask.W = torch.ones(self.hyper.encoder_dim, self.hyper.output_dim, device=self.hyper.device)

        self.init()

    @torch.no_grad()
    def init(self)->None:
        # init each element as gain/sqrt(n)
        nn.init.xavier_normal_(self.J, gain=self.hyper.init_gain)
        nn.init.normal_(self.I)
        nn.init.normal_(self.W)
        if self.hyper.fixed_I:
            nn.init.zeros_(self.trainable_mask.I)
        if self.hyper.fixed_W:
            nn.init.zeros_(self.trainable_mask.W)

    def x_dim(self)->int:
        return self.hyper.encoder_dim

    def pre_forward(self)->None:
        pass

    def recurrence(self, x_phi: torch.Tensor, input: Optional[torch.Tensor]=None, rank_mask=None)->torch.Tensor:
        """
        x_phi: torch.tensor([b, n])
        input: torch.tensor([b, I])
        """
        if input is None:
            input = torch.zeros(x_phi.shape[0], self.hyper.input_dim, device=x_phi.device)
        batch_size = x_phi.shape[0]
        assert rank_mask is None
        next_x = (self.J @ x_phi.T).T + input @ self.I.T
        if self.hyper.has_bias:
            next_x = next_x + self.B.view(1, -1)
        assert next_x.dim() == 2
        assert next_x.shape == x_phi.shape
        return next_x.view(x_phi.shape[0], -1)

    def outputs(self, x: torch.Tensor)->torch.Tensor:
        """
        x: [b, n]
        or x: [b, t, n]
        """
        return x @ self.W / self.hyper.encoder_dim * self.hyper.readout_radius

    @property
    def regulization_parameters(self)->table:
        ret = table()
        ret.J = self.J
        ret.I = self.I
        ret.W = self.W
        ret.B = self.B
        return ret

    @torch.no_grad()
    def load(self, I: torch.Tensor, J: torch.Tensor, W: torch.Tensor)->None:
        assert I.shape == (self.hyper.encoder_dim, self.hyper.input_dim)
        assert J.shape == (self.hyper.encoder_dim, self.hyper.encoder_dim)
        assert W.shape == (self.hyper.encoder_dim, self.hyper.output_dim)
        self.I.data.copy_(I)
        self.J.data.copy_(J)
        self.W.data.copy_(W)

    def loadingvectors(self)->torch.Tensor:
        raise RuntimeError()

    def justify_grad(self)->None:
        desired_grad = table()

        for name, param in self.named_parameters():
            if name in self.trainable_mask:
                desired_grad[name] = param.grad.data.clone() * self.trainable_mask[name]

        for name, param in self.named_parameters():
            if name in desired_grad:
                param.grad.data.copy_(desired_grad[name])

class lowrank_encoder(encoder):

    def __init__(self, **kwargs: Union[int, float, str, None]):
        super().__init__(**kwargs)

        self.hyper = table(tau=100,
                delta_t = 10,
                encoder_dim = 4096,
                noise_rec = 0.05,
                noise_inp = 0.01,
                input_dim = 1,
                input_gain = 1.0,
                output_dim = 1,
                max_rank = 1,
                init_gain = 1.0,
                fixed_I = False,
                fixed_W = False,
                has_bias = False,
                readout_radius = 1.0,
                fixed_rank = "",
                phi = "torch.tanh",
                device = None)

        self.hyper.update_exist(kwargs)
        if len(kwargs) != len(self.hyper):
            raise TypeError(table(kwargs).key_not_here(self.hyper))
        self.property = table()
        self.property.alpha = self.hyper.delta_t / self.hyper.tau
        self.property.I_index_shift = 0
        self.property.U_index_shift = self.hyper.input_dim + self.hyper.has_bias
        self.property.V_index_shift = self.hyper.input_dim + self.hyper.max_rank + self.hyper.has_bias
        self.property.W_index_shift = self.hyper.input_dim + self.hyper.max_rank * 2 + self.hyper.has_bias
        self.property.I_index = vector.range(self.hyper.input_dim)
        self.property.U_index = vector.range(self.hyper.max_rank).map_add(self.property.U_index_shift)
        self.property.V_index = vector.range(self.hyper.max_rank).map_add(self.property.V_index_shift)
        self.property.W_index = vector.range(self.hyper.output_dim).map_add(self.property.W_index_shift)
        self.property.I_slice = slice(self.property.I_index_shift, self.property.I_index_shift + self.hyper.input_dim)
        self.property.U_slice = slice(self.property.U_index_shift, self.property.U_index_shift + self.hyper.max_rank)
        self.property.V_slice = slice(self.property.V_index_shift, self.property.V_index_shift + self.hyper.max_rank)
        self.property.W_slice = slice(self.property.W_index_shift, self.property.W_index_shift + self.hyper.output_dim)
        if self.hyper.has_bias:
            self.property.B_index = [self.hyper.input_dim]
            self.property.B_index_shift = self.hyper.input_dim
            self.property.B_slice = slice(self.hyper.input_dim, self.hyper.input_dim + 1)
        else:
            self.property.B_index = []
            self.property.B_index_shift = None
            self.property.B_slice = slice(0)
        self.property.R = self.hyper.input_dim + 2 * self.hyper.max_rank + self.hyper.output_dim + self.hyper.has_bias
        assert 0 < self.property.alpha <= 1
        assert self.hyper.input_dim > 0
        assert self.hyper.max_rank > 0
        assert self.hyper.output_dim > 0

        self.trainable_mask = table()
        self.trainable_mask.L = torch.ones(self.hyper.encoder_dim, self.property.R, device=self.hyper.device)

        self._L = nn.Parameter(torch.zeros(self.hyper.encoder_dim, self.property.R, device=self.hyper.device))

        def _rank_ind_value(x: Any, rank: int, device: torch.DeviceObjType) -> torch.Tensor:
            def _inside(x: Any, rank: int, device: torch.DeviceObjType) -> torch.Tensor:
                if isinstance(x, float):
                    return torch.Tensor([x] * rank).to(device)
                if not x:
                    return torch.zeros(rank, device=device)
                if isinstance(x, str) and len(x):
                    if "/" in x:
                        return torch.tensor(vector(x.split("/")).map(float)).to(device)
                    else:
                        return torch.Tensor([float(x)] * rank).to(device)
                if isinstance(x, (list, tuple)) and len(x):
                    return torch.tensor(x).to(device)
                if isinstance(x, torch.Tensor):
                    return x.to(device)
                raise ValueError()
            ret = _inside(x, rank, device)
            assert isinstance(ret, torch.Tensor)
            assert ret.dim() == 1
            assert ret.shape[0] == rank
            return ret

        self.property.fixed_rank = vector(self.hyper.fixed_rank.split("/")).filter(len).map(int)

        self.init()

    @torch.no_grad()
    def init(self)->None:
        nn.init.zeros_(self._L)
        self._L.data[:, self.property.I_index] = torch.randn_like(self._L[:, self.property.I_index])
        self._L.data[:, self.property.U_index + self.property.V_index] = torch.randn_like(self._L[:, self.property.U_index + self.property.V_index]) * self.hyper.init_gain
        self._L.data[:, self.property.W_index] = torch.randn_like(self._L[:, self.property.W_index])
        if self.hyper.fixed_I:
            self.trainable_mask.L.data[:, self.property.I_index] = 0
        if self.hyper.fixed_W:
            self.trainable_mask.L.data[:, self.property.W_index] = 0
        for r in self.property.fixed_rank:
            self.trainable_mask.L[:, r + self.property.U_index_shift, :] = 0
            self.trainable_mask.L[:, r + self.property.V_index_shift, :] = 0

    def x_dim(self)->int:
        return self.hyper.encoder_dim

    def pre_forward(self)->None:
        self._I = self._L[:, self.property.I_index] # nI
        self._U = self._L[:, self.property.U_index] # nr
        self._V = self._L[:, self.property.V_index] # nr
        self._W = self._L[:, self.property.W_index] # nz
        if self.hyper.has_bias:
            self._B = self._L[:, self.property.B_index].squeeze(1)

    def init_state(self, batch_size: int)->torch.Tensor:
        return torch.zeros(batch_size, self.x_dim()).to(self.hyper.device)

    def recurrence(self, x_phi: torch.Tensor, input: Optional[torch.Tensor]=None, rank_mask=None)->torch.Tensor:
        """
        x_phi: torch.tensor([b, n])
        input: torch.tensor([b, I])
        """
        if input is None:
            input = torch.zeros(x_phi.shape[0], self.hyper.input_dim, device=x_phi.device)
        batch_size = x_phi.shape[0]
        if rank_mask is not None:
            selection = torch.einsum("nr,r,bn->br", self._V, rank_mask, x_phi) / self.hyper.encoder_dim
        else:
            selection = torch.einsum("nr,bn->br", self._V, x_phi) / self.hyper.encoder_dim
        next_x = torch.einsum("br,nr->bn", selection, self._U) +  torch.einsum("nI,bI->bn", self._I, input)
        if self.hyper.has_bias:
            next_x = next_x + self._B.unsqueeze(0)
        return next_x.view(x_phi.shape[0], -1)

    def outputs(self, x: torch.Tensor)->torch.Tensor:
        """
        x: [b, n]
        or x: [b, t, n]
        """
        batch_size = x.shape[0]
        if x.dim() == 2:
            return torch.einsum("nz,bn->bz", self._W, x) / self.hyper.encoder_dim * self.hyper.readout_radius
        elif x.dim() == 3:
            return torch.einsum("nz,btn->btz", self._W, x) / self.hyper.encoder_dim * self.hyper.readout_radius
        raise RuntimeError()

    @property
    def regulization_parameters(self)->table:
        ret = table()
        ret.L = self._L
        return ret

    @torch.no_grad()
    def load(self, L: torch.Tensor)->None:
        assert L.shape == (self.hyper.encoder_dim, self.property.R)
        self._L.data.copy_(L)

    def loadingvectors(self)->torch.Tensor:
        return self._L

    def parameters_lr(self)->List[Dict]:
        params = [self._L]
        return [{"params": params}]

    def overlap(self, mode: Optional[str]=None) -> torch.Tensor:
        assert mode is None or mode in ["I", "U", "V", "W"] or mode in [i + j for i in ["I", "U", "V", "W"] for j in ["I", "U", "V", "W"]] or mode in [i + "+" + j for i in ["I", "U", "V", "W"] for j in ["I", "U", "V", "W"] if i != j]
        if mode is None or mode == "":
            return self._L.T @ self._L / self.hyper.encoder_dim
        elif len(mode) == 1:
            i = self.property[f"{mode}_index"]
            j = self.property[f"{mode}_index"]
        elif len(mode) == 2:
            i = self.property[f"{mode[0]}_index"]
            j = self.property[f"{mode[1]}_index"]
        else:
            assert mode[1] == "+"
            i = self.property[f"{mode[0]}_index"] + self.property[f"{mode[2]}_index"]
            j = self.property[f"{mode[0]}_index"] + self.property[f"{mode[2]}_index"]
        return self._L[:, i].T @ self._L[:, j] / self.hyper.encoder_dim

    def correlation(self, mode: Optional[str]=None) -> torch.Tensor:
        assert mode is None or mode in ["I", "U", "V", "W"] or mode in [i + j for i in ["I", "U", "V", "W"] for j in ["I", "U", "V", "W"]] or mode in [i + "+" + j for i in ["I", "U", "V", "W"] for j in ["I", "U", "V", "W"] if i != j]
        L = self._L - self._L.mean(0, keepdim=True)
        if mode is None or mode == "":
            return L.T @ L / self.hyper.encoder_dim
        elif len(mode) == 1:
            i = self.property[f"{mode}_index"]
            j = self.property[f"{mode}_index"]
        elif len(mode) == 2:
            i = self.property[f"{mode[0]}_index"]
            j = self.property[f"{mode[1]}_index"]
        else:
            assert mode[1] == "+"
            i = self.property[f"{mode[0]}_index"] + self.property[f"{mode[2]}_index"]
            j = self.property[f"{mode[0]}_index"] + self.property[f"{mode[2]}_index"]
        return L[:, i].T @ L[:, j] / self.hyper.encoder_dim

    def justify_grad(self)->None:
        desired_grad = table()
        for name, param in self.named_parameters():
            assert param.dim() > 1
            if param.requires_grad:
                desired_grad[name] = param.grad.data.clone()

        for name in desired_grad:
            if name == "_L":
                desired_grad[name] = desired_grad[name] * self.trainable_mask.L
            else:
                raise RuntimeError()

        for name, param in self.named_parameters():
            if name in desired_grad:
                param.grad.data.copy_(desired_grad[name])

class reparameter_encoder(encoder):

    def __init__(self, R: int=-1, **kwargs: Union[int, float, str, None]):
        super().__init__(**kwargs)

        self.hyper = table(tau=100,
                delta_t = 10,
                encoder_dim = 4096,
                noise_rec = 0.05,
                noise_inp = 0.01,
                input_dim = 1,
                input_gain = 1.0,
                output_dim = 1,
                max_rank = 2,
                init_gain = 1.0,
                n_populations = 1,
                weights_lr = -1,
                fixed_I = False,
                fixed_W = False,
                readout_radius = 1.0,
                zero_mean = False,
                zero_variance = False,
                fixed_population = "",
                fixed_rank = "",
                phi = "torch.tanh",
                symmetric_expanded = None,
                has_bias = False,
                device = None)

        self.hyper.update_exist(kwargs)
        if len(kwargs) != len(self.hyper):
            raise TypeError(table(kwargs).key_not_here(self.hyper))
        self.property = table()
        self.property.alpha = self.hyper.delta_t / self.hyper.tau
        self.property.I_index_shift = 0
        self.property.U_index_shift = self.hyper.input_dim + self.hyper.has_bias
        self.property.V_index_shift = self.hyper.input_dim + self.hyper.max_rank + self.hyper.has_bias
        self.property.W_index_shift = self.hyper.input_dim + self.hyper.max_rank * 2 + self.hyper.has_bias
        self.property.I_index = vector.range(self.hyper.input_dim)
        self.property.U_index = vector.range(self.hyper.max_rank).map_add(self.property.U_index_shift)
        self.property.V_index = vector.range(self.hyper.max_rank).map_add(self.property.V_index_shift)
        self.property.W_index = vector.range(self.hyper.output_dim).map_add(self.property.W_index_shift)
        self.property.I_slice = slice(self.property.I_index_shift, self.property.I_index_shift + self.hyper.input_dim)
        self.property.U_slice = slice(self.property.U_index_shift, self.property.U_index_shift + self.hyper.max_rank)
        self.property.V_slice = slice(self.property.V_index_shift, self.property.V_index_shift + self.hyper.max_rank)
        self.property.W_slice = slice(self.property.W_index_shift, self.property.W_index_shift + self.hyper.output_dim)
        if self.hyper.has_bias:
            self.property.B_index = [self.hyper.input_dim]
            self.property.B_index_shift = self.hyper.input_dim
            self.property.B_slice = slice(self.hyper.input_dim, self.hyper.input_dim + 1)
        else:
            self.property.B_index = []
            self.property.B_index_shift = None
            self.property.B_slice = slice(0)
        if R == -1:
            self.property.R = self.hyper.input_dim + 2 * self.hyper.max_rank + self.hyper.output_dim + self.hyper.has_bias
        else:
            self.property.R = R
        self.property.L = self.hyper.input_dim + 2 * self.hyper.max_rank + self.hyper.output_dim + self.hyper.has_bias
        assert 0 < self.property.alpha <= 1
        assert self.hyper.input_dim > 0
        assert self.hyper.max_rank > 0
        assert self.hyper.output_dim > 0

        self.mask = torch.ones(self.hyper.n_populations, self.property.L, self.property.R, device=self.hyper.device)

        self.trainable_mask = table()
        self.trainable_mask.C = torch.ones(self.hyper.n_populations, self.property.L, self.property.R, device=self.hyper.device)
        self.trainable_mask.mu = torch.ones(self.hyper.n_populations, self.property.L, device=self.hyper.device)

        self.weight_logits = nn.Parameter(torch.zeros(self.hyper.n_populations, device=self.hyper.device))
        self._C = nn.Parameter(torch.zeros(self.hyper.n_populations, self.property.L, self.property.R, device=self.hyper.device))
        self._mu = nn.Parameter(torch.zeros(self.hyper.n_populations, self.property.L, device=self.hyper.device))

        def _rank_ind_value(x: Any, rank: int, device: torch.DeviceObjType) -> torch.Tensor:
            def _inside(x: Any, rank: int, device: torch.DeviceObjType) -> torch.Tensor:
                if isinstance(x, float):
                    return torch.Tensor([x] * rank).to(device)
                if not x:
                    return torch.zeros(rank, device=device)
                if isinstance(x, str) and len(x):
                    if "/" in x:
                        return torch.tensor(vector(x.split("/")).map(float)).to(device)
                    else:
                        return torch.Tensor([float(x)] * rank).to(device)
                if isinstance(x, (list, tuple)) and len(x):
                    return torch.tensor(x).to(device)
                if isinstance(x, torch.Tensor):
                    return x.to(device)
                raise ValueError()
            ret = _inside(x, rank, device)
            assert isinstance(ret, torch.Tensor)
            assert ret.dim() == 1
            assert ret.shape[0] == rank
            return ret

        self.property.fixed_rank = vector(self.hyper.fixed_rank.split("/")).filter(len).map(int)
        self.property.fixed_population = vector(self.hyper.fixed_population.split("/")).filter(len).map(int)

        self.init()
        self.resample()

    def _n_populations_after_expansion(self, symmetric_expanded)->int:
        if not symmetric_expanded:
            return self.hyper.n_populations
        assert isinstance(symmetric_expanded, (list, tuple))
        if not isinstance(symmetric_expanded, vector):
            symmetric_expanded = vector(symmetric_expanded)
        if isinstance(symmetric_expanded[0], tuple):
            return symmetric_expanded.map(lambda x: len(x[1]) if not x[2] else 1).sum()
        return len(symmetric_expanded) * self.hyper.n_populations

    @torch.no_grad()
    def init(self)->None:
        nn.init.zeros_(self._C)
        nn.init.zeros_(self._mu)
        nn.init.zeros_(self.weight_logits)
        nn.init.normal_(self._C, mean=0, std=self.hyper.init_gain)
        if self.hyper.fixed_I:
            self.trainable_mask.C.data[:, self.property.I_index, :] = 0
            self.trainable_mask.mu.data[:, self.property.I_index] = 0
        if self.hyper.fixed_W:
            self.trainable_mask.C[:, self.property.W_index, :] = 0
            self.trainable_mask.mu[:, self.property.W_index] = 0
        if self.hyper.zero_mean:
            self._mu.requires_grad_(False)
            nn.init.zeros_(self._mu)
            nn.init.zeros_(self.trainable_mask.mu)
        if self.hyper.zero_variance:
            self._C.requires_grad_(False)
            nn.init.normal_(self._C, std=0.01)
            nn.init.zeros_(self.trainable_mask.C)
            nn.init.zeros_(self._mu)
        for p in self.property.fixed_population:
            self.trainable_mask.C[p, :, :] = 0
            self.trainable_mask.mu[p, :] = 0
        for r in self.property.fixed_rank:
            self.trainable_mask.C[:, r + self.property.U_index_shift, :] = 0
            self.trainable_mask.C[:, r + self.property.V_index_shift, :] = 0
            self.trainable_mask.mu[:, r + self.property.U_index_shift] = 0
            self.trainable_mask.mu[:, r + self.property.V_index_shift] = 0
        if self.hyper.symmetric_expanded and isinstance(self.hyper.symmetric_expanded[0], torch.Tensor):
            self.hyper.symmetric_expanded = vector.range(self.hyper.n_populations).map(lambda p: (p, self.hyper.symmetric_expanded, False))
        self.property.n_populations_after_expansion = self._n_populations_after_expansion(self.hyper.symmetric_expanded)
        self.trainable_mask.C = self.trainable_mask.C * self.mask

    def extra_saved(self)->dict:
        return dict(trainable_mask=self.trainable_mask, mask=self.mask, symmetric_expanded=self.hyper.symmetric_expanded)

    def load_extra(self, loaded: dict)->None:
        if not loaded:
            return
        if "trainable_mask" in loaded:
            for key in self.trainable_mask:
                self.trainable_mask[key].copy_(loaded["trainable_mask"][key])
        if "mask" in loaded:
            self.mask.copy_(loaded["mask"])
        if "symmetric_expanded" in loaded:
            self.hyper.symmetric_expanded = vector()
            for p, expanded, average in loaded["symmetric_expanded"]:
                self.hyper.symmetric_expanded.append((p, vector(expanded).map(lambda x: x.to(self.hyper.device)), average))

    def x_dim(self)->int:
        return self.hyper.encoder_dim * self.property.n_populations_after_expansion

    def pre_forward(self)->None:
        loadings, weights = self.loadingvectors()
        self._I = loadings[:, :, self.property.I_index] # pnI
        self._U = loadings[:, :, self.property.U_index] # pnr
        self._V = loadings[:, :, self.property.V_index] # pnr
        self._W = loadings[:, :, self.property.W_index] # pnz
        if self.hyper.has_bias:
            self._B = loadings[:, :, self.property.B_index].squeeze(-1) # pn
        self._weights = weights

    def init_state(self, batch_size: int)->torch.Tensor:
        return torch.zeros(batch_size, self.x_dim()).to(self.hyper.device)

    def recurrence(self, x_phi: torch.Tensor, input: Optional[torch.Tensor]=None, rank_mask=None)->torch.Tensor:
        """
        x_phi: torch.tensor([b, p * n])
        input: torch.tensor([b, I])
        """
        if input is None:
            input = torch.zeros(x_phi.shape[0], self.hyper.input_dim, device=x_phi.device)
        batch_size = x_phi.shape[0]
        x_phi = x_phi.view(batch_size, self.property.n_populations_after_expansion, self.hyper.encoder_dim) # [b, p, n]
        if rank_mask is not None:
            selection = torch.einsum("pnr,r,bpn,p->br", self._V, rank_mask, x_phi, self._weights) / self.hyper.encoder_dim
        else:
            selection = torch.einsum("pnr,bpn,p->br", self._V, x_phi, self._weights) / self.hyper.encoder_dim
        next_x = torch.einsum("br,pnr->bpn", selection, self._U) +  torch.einsum("pnI,bI->bpn", self._I, input)
        if self.hyper.has_bias:
            next_x = next_x + self._B.unsqueeze(0)
        return next_x.view(x_phi.shape[0], -1)

    def outputs(self, x: torch.Tensor)->torch.Tensor:
        """
        x: [b, p * n]
        or x: [b, t, p * n]
        """
        batch_size = x.shape[0]
        if x.dim() == 2:
            x = x.view(x.shape[0], self.property.n_populations_after_expansion, self.hyper.encoder_dim)
            ret = torch.einsum("pnz,bpn,p->bz", self._W, x, self._weights) / self.hyper.encoder_dim * self.hyper.readout_radius
        elif x.dim() == 3:
            x = x.view(x.shape[0], x.shape[1], self.property.n_populations_after_expansion, self.hyper.encoder_dim)
            ret = torch.einsum("pnz,btpn,p->btz", self._W, x, self._weights) / self.hyper.encoder_dim * self.hyper.readout_radius
        else:
            raise RuntimeError()
        return ret

    def covariances(self)->torch.Tensor:
        return torch.einsum("pij,pkj->pik", self.C(), self.C())

    def entropy(self)->torch.Tensor:
        return Categorical(probs=self.weights()).entropy()

    @property
    def regulization_parameters(self)->table:
        ret = table()
        ret.C = self._C
        return ret

    @torch.no_grad()
    def load(self, weights: torch.Tensor, means: torch.Tensor, covariances: torch.Tensor)->None:
        if means is None:
            means = torch.zeros_like(self._mu)
        if weights is None:
            weights = torch.ones_like(self.weight_logits) / self.hyper.n_populations
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().clone().to(self.hyper.device)
            means = means.detach().clone().to(self.hyper.device)
            covariances = covariances.detach().clone().to(self.hyper.device)
        else:
            weights = torch.tensor(weights).type_as(self.weight_logits)
            means = torch.tensor(means).type_as(self.weight_logits)
            covariances = torch.tensor(covariances).type_as(self.weight_logits)
        assert weights.sum().allclose(torch.ones(1).to(weights.device))
        assert weights.shape == (self.hyper.n_populations,)
        assert means.shape == (self.hyper.n_populations, self.property.R)
        assert covariances.shape == (self.hyper.n_populations, self.property.R, self.property.R)
        with torch.no_grad():
            c = torch.linalg.cholesky(covariances)
            self.weight_logits.data.copy_(weights.log())
            self._mu.data.copy_(means)
            self._C.data.copy_(c)

    def loadingvectors(self)->Tuple[torch.Tensor, torch.Tensor]:
        wlog, c, means = self.symmetric_wlog_c_and_means()
        lv = torch.einsum("pnr,pRr->pnR", self.noise_loading, c) +  means.view(self.property.n_populations_after_expansion, 1, self.property.L)
        weights = torch.softmax(wlog, 0)
        return lv, weights

    def weights(self)->torch.Tensor:
        weights = torch.softmax(self.weight_logits, 0)
        return weights

    def resample(self, whiten=False)->None:
        if not whiten:
            self.noise_loading = torch.randn(self.property.n_populations_after_expansion, self.hyper.encoder_dim, self.property.R, device=self.hyper.device)
        else:
            self.noise_loading = standard_normal(self.property.n_populations_after_expansion, self.hyper.encoder_dim, self.property.R, device=self.hyper.device)
        self.pre_forward()

    def sample(self, n: int)->Tuple[torch.Tensor, torch.Tensor]:
        counts = torch.distributions.multinomial.Multinomial(total_count=n, probs=self.weights()).sample()
        x = torch.empty(0, device=counts.device)
        y = torch.cat([torch.full([int(sample)], j, device=counts.device) for j, sample in enumerate(counts)])

        # Only iterate over components with non-zero counts
        # covariances = self.covariances()
        means = self.means()
        cholesky_components = self.C()

        for k in range(self.hyper.n_populations):
            if counts[k] == 0:
                continue
            x_k = torch.einsum("nR,LR->nL", torch.randn(int(counts[k]), self.property.R, device=cholesky_components.device), cholesky_components[k]) + means[k].view(1, self.property.L)
            x = torch.cat((x, x_k), dim=0)

        return x, y

    def means(self)->torch.Tensor:
        return self._mu

    def C(self)->torch.Tensor:
        def __C() -> torch.Tensor:
            C = self._C
            return C * self.mask
        ret_C = __C()
        return ret_C

    def parameters_lr(self)->List[Dict]:
        if self.hyper.weights_lr < 0:
            return [{"params": self.parameters()}]
        params = [self._C]
        if not self.hyper.zero_mean:
            params.append(self._mu)
        return [{"params": params}, {"params": [self.weight_logits], "lr": self.hyper.weights_lr}]

    def overlap(self, mode: Optional[str]=None, population=False) -> torch.Tensor:
        covariance = self.covariances()
        means = self.means()
        weights = self.weights()
        assert mode is None or mode in ["I", "U", "V", "W", "B"] or mode in [i + j for i in ["I", "U", "V", "W", "B"] for j in ["I", "U", "V", "W", "B"]] or mode in [i + "+" + j for i in ["I", "U", "V", "W", "B"] for j in ["I", "U", "V", "W", "B"] if i != j]
        if mode is None or mode == "":
            if population:
                return covariance + torch.einsum("pi,pj->pij", means, means)
            else:
                return torch.einsum("pij,p->ij", covariance, weights) + torch.einsum("pi,pj,p->ij", means, means, weights)
        elif len(mode) == 1:
            i = self.property[f"{mode}_index"]
            j = self.property[f"{mode}_index"]
        elif len(mode) == 2:
            i = self.property[f"{mode[0]}_index"]
            j = self.property[f"{mode[1]}_index"]
        else:
            assert mode[1] == "+"
            i = self.property[f"{mode[0]}_index"] + self.property[f"{mode[2]}_index"]
            j = self.property[f"{mode[0]}_index"] + self.property[f"{mode[2]}_index"]
        if population:
            return covariance[:, i, :][:, :, j] + torch.einsum("pi,pj->pij", means[:, i], means[:, j])
        else:
            return torch.einsum("pij,p->ij", covariance[:, i, :][:, :, j], weights) + torch.einsum("pi,pj,p->ij", means[:, i], means[:, j], weights)

    def correlation(self, mode: Optional[str]=None)->torch.Tensor:
        if mode is None or mode in ["I", "U", "V", "W", "B"] or mode in [i + "+" + j for i in ["I", "U", "V", "W", "B"] for j in ["I", "U", "V", "W", "B"] if i != j]:
            overlap = self.overlap(mode=mode)
            overlap_diag_invsqrt = overlap.diag().pow(-0.5)
            return torch.einsum("i,ij,j->ij", overlap_diag_invsqrt, overlap, overlap_diag_invsqrt)
        else:
            overlap = self.overlap(mode=mode)
            overlap_left = self.overlap(mode=mode[0])
            overlap_right = self.overlap(mode=mode[1])
            overlap_diag_invsqrt_left = overlap_left.diag().pow(-0.5)
            overlap_diag_invsqrt_right = overlap_right.diag().pow(-0.5)
            return torch.einsum("i,ij,j->ij", overlap_diag_invsqrt_left, overlap, overlap_diag_invsqrt_right)

    @torch.no_grad()
    def expand_population(self, delta_p: int, noise: float=0.0)->"reparameter_encoder":
        if delta_p == 0:
            return self
        print(f"expand {delta_p} populations: {self.hyper.n_populations}->{self.hyper.n_populations+delta_p}")
        assert delta_p > 0
        hyper = self.hyper.copy()
        hyper.n_populations = self.hyper.n_populations + delta_p
        ret = reparameter_encoder(**hyper)
        device = self.hyper.device
        n_populations = self.hyper.n_populations
        input_dim = self.hyper.input_dim
        max_rank = self.hyper.max_rank
        has_mean = not self.hyper.zero_mean
        has_bias = self.hyper.has_bias
        alpha = n_populations / (n_populations + delta_p)
        I_index = self.property.I_index
        U_index = self.property.U_index
        V_index = self.property.V_index
        W_index = self.property.W_index
        B_index = self.property.B_index
        ret._C.data[:n_populations, I_index, :] = self._C[:, I_index, :]
        ret._mu.data[:n_populations, I_index] = self._mu[:, I_index]
        if has_bias:
            ret._C.data[:n_populations, B_index, :] = self._C[:, B_index, :]
            ret._mu.data[:n_populations, B_index] = self._mu[:, B_index]
        ret._C.data[:n_populations, U_index + V_index, :] = self._C[:, U_index + V_index, :] * (1 / alpha) ** 0.5
        ret._mu.data[:n_populations, U_index + V_index] = self._mu[:, U_index + V_index] * (1 / alpha) ** 0.5
        ret._C.data[:n_populations, W_index, :] = self._C.data[:, W_index, :] * (1 / alpha)
        ret._mu.data[:n_populations, W_index, :] = self._mu.data[:, W_index] * (1 / alpha)
        ret.weight_logits.data[:n_populations] = self.weight_logits.data.clone()
        ret.weight_logits.data[n_populations:] = torch.logsumexp(self.weight_logits, 0) - torch.log(torch.tensor(n_populations).to(device))
        return ret

    @torch.no_grad()
    def expand_rank(self, delta_I: int, delta_r: int, delta_z: int)->"reparameter_encoder":
        if delta_I == 0 and delta_r == 0 and delta_z == 0:
            return self
        print(f"expand {delta_I} input and {delta_r} rank and {delta_z} outputs")
        assert delta_r >= 0 and delta_z >= 0 and delta_I >= 0
        hyper = self.hyper.copy()
        hyper.input_dim = self.hyper.input_dim + delta_I
        hyper.max_rank = self.hyper.max_rank + delta_r
        hyper.output_dim = self.hyper.output_dim + delta_z
        ret = reparameter_encoder(**hyper)
        l_index = ["I", "U", "V"]
        l_index.append("W")
        if self.hyper.has_bias:
            l_index.append("B")
        for i in l_index:
            for j in l_index:
                _copy(ret._C[:, ret.property[f"{i}_slice"], ret.property[f"{j}_slice"]], self._C[:, self.property[f"{i}_slice"], self.property[f"{j}_slice"]])
            _copy(ret._mu[:, ret.property[f"{i}_slice"]], self._mu[:, self.property[f"{i}_slice"]])
        ret.weight_logits.data.copy_(self.weight_logits)
        return ret


    def justify_grad(self)->None:
        desired_grad = table()
        for name, param in self.named_parameters():
            if name in ["weight_logits", "V_to_W", "bias_w"]:
                continue
            assert param.dim() > 1
            if param.requires_grad:
                if param.grad is None:
                    desired_grad[name] = None
                else:
                    desired_grad[name] = param.grad.data.clone()

        for name in desired_grad:
            if desired_grad[name] is None:
                continue
            if name == "_C":
                desired_grad[name] = desired_grad[name] * self.trainable_mask.C
            elif name == "_mu":
                desired_grad[name] = desired_grad[name] * self.trainable_mask.mu
            else:
                raise RuntimeError()

        for name, param in self.named_parameters():
            if name in desired_grad:
                if desired_grad[name] is None:
                    continue
                param.grad.data.copy_(desired_grad[name])

    def symmetric_wlog_c_and_means(self)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.hyper.symmetric_expanded:
            return self.weight_logits, self.C(), self.means()
        wlog = self.weight_logits
        c = self.C()
        mean = self.means()
        ret_wlog = vector()
        ret_c = vector()
        ret_mean = vector()
        for p, expanded, average in self.hyper.symmetric_expanded:
            if not average:
                for index, x in enumerate(expanded):
                    ret_wlog.append(wlog[p] - math.log(len(expanded)))
                    ret_c.append(x @ c[p, :, :])
                    ret_mean.append(torch.einsum("ij,j->i", x, mean[p, :]))
            else:
                ret_wlog.append(wlog[p])
                ret_c.append(torch.stack([x @ c[p, :, :] for x in expanded], 0).mean(0))
                ret_mean.append(torch.stack([torch.einsum("ij,j->i", x, mean[p, :]) for x in expanded], 0).mean(0))
        return torch.stack(ret_wlog, 0), torch.stack(ret_c, 0), torch.stack(ret_mean, 0)

    def symmetric_weights_covs_and_means(self)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wlog, c, mean = self.symmetric_wlog_c_and_means()
        covs = torch.einsum("pij,pkj->pik", c, c)
        return torch.softmax(wlog, 0), covs, mean

    def to(self, device):
        super().to(device)
        self.hyper.device = device
        if hasattr(self, 'noise_loading'):
            self.noise_loading = self.noise_loading.to(device)
        self.mask = self.mask.to(device)
        self.trainable_mask.C = self.trainable_mask.C.to(device)
        self.trainable_mask.mu = self.trainable_mask.mu
        self.pre_forward()
        return self
