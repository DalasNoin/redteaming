import torch as t
from torch import nn, optim
import plotly.express as px
from typing import Optional, Union, List
from models import utils
from torch.nn.functional import gelu
from torch.nn import init


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(init.normal_(t.empty((num_embeddings, embedding_dim))))

    def forward(self, x: t.LongTensor) -> t.Tensor:
        '''For each integer in the input, return that row of the embedding.
        '''
        return self.weight[x]

    def extra_repr(self) -> str:
        return "Embedding num_embeddings x embedding dim= {self.num_embeddings} x {self.embedding_dim}"



class GELU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return gelu(x)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-05, elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = normalized_shape

        self.eps = eps
        
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(t.ones(normalized_shape))
            self.bias = nn.Parameter(t.zeros(normalized_shape))


    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        figure out
        """
        first_dim_to_normalize = len(x.shape)-len(self.normalized_shape)
        dims = tuple(range(first_dim_to_normalize, len(x.shape)))
        # print(dims, self.normalized_shape, x.shape)
        mean = t.mean(x, dim=dims, keepdims=True)
        var = t.var(x, dim=dims, unbiased=False, keepdims=True)

        x = (x - mean) / ((var + self.eps) ** 0.5)

        if self.elementwise_affine:
            x = x * self.weight + self.bias

        return x


class Dropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        assert 0 <= p < 1, "p is a probability from 0 to 1"
        self.p = p

    def forward(self, x: t.Tensor) -> t.Tensor:
        if self.training:
            mask = (t.rand(size=x.shape) < self.p)
            mask = mask.to(x.device)
            return t.where(mask, 0.0, x / (1 - self.p))
        else:
            return x

