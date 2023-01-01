import torch
from torch import nn
from torch.nn import GELU, Softmax
from dataclasses import dataclass
from models import utils
from models import modules
import transformers


@dataclass(frozen=True)
class TransformerConfig:
    '''Constants used throughout your decoder-only transformer model.'''

    num_layers: int = 12
    # head_size is not in this config, because in our implementation we're assuming num_heads * head_size = hidden_size
    num_heads: int = 12
    vocab_size: int = 50_257
    # hidden_size is also referred to as embedding_dim, or d_\text{model}d model in some material you might have read.
    hidden_size: int = 768
    # max_seq_len is used just to determine the size of the positional encoding matrix.
    max_seq_len: int = 1024
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-05
    device: str = "cpu"


class GPT2MLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.dropout = config.dropout
        self.mlp_block = nn.Sequential(
            nn.Linear(self.hidden_size, 4*self.hidden_size),
            modules.GELU(),
            nn.Linear(4*self.hidden_size, self.hidden_size),
            modules.Dropout(self.dropout)
        )
    def forward(self, x: torch.Tensor):
        return self.mlp_block(x)

# Q = torch.ones((2,20,4*64))
# K = torch.ones((2,10,4*64))
# V = torch.ones((2,10,4*64))
# num_heads = 4




class GPT2Attention(nn.Module):
    """
    head_size is not in this config, because in our implementation we're assuming num_heads * head_size = hidden_size.
    hidden_size is also referred to as embedding_dim, or d_\text{model}d 
    model in some material you might have read.

    I ignored this for now as it would require changing the masked attention function
    The attention block has two dropout layers: 
    one immediately after the softmax (i.e. before multiplying by V), 
    and one immediately after multiplying with W_O at the very end of the attention block. 
    Note that the dropout layers won't actually affect weight-loading or performance in eval mode 
    (and you should still be able to train your model without them), 
    but all the same it's nice to be able to exactly match GPT's architecture!
    """
    W_QKV: nn.Linear
    W_O: nn.Linear


    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.device = config.device
        self.head_size = self.hidden_size // self.num_heads
        self.W_QKV = nn.Linear(self.hidden_size, self.num_heads*self.head_size*3)
        self.dropout1 = modules.Dropout(config.dropout)
        self.W_O = nn.Linear(self.num_heads*self.head_size, self.hidden_size)
        self.dropout2 = modules.Dropout(config.dropout)
        self.softmax = Softmax(dim=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        '''
        # x = x.repeat((1,1,3)) # repeat trice along dim 2
        x = self.W_QKV(x)
        #print(f"{x.shape=} {num_heads=} {self.hidden_size=}")
        Q, K, V = torch.split(x, self.num_heads*self.head_size, 2)
        #print(f"{Q.shape=} {K.shape=} {V.shape=}")
        
        # Z = multihead_masked_attention(Q, K, V, num_heads=self.num_heads, device=self.device)
        batch, target_seq_len = Q.shape[0:2]
        source_seq_len = K.shape[1] 
        head_size = int(Q.shape[-1]/self.num_heads)
        sqrt_d_k = torch.sqrt(torch.tensor(self.head_size))
        # new_shape = (batch, target_seq_len, num_heads, head_size)
        Q = torch.reshape(Q, (batch, target_seq_len, self.num_heads, self.head_size))
        K = torch.reshape(K, (batch, source_seq_len, self.num_heads, self.head_size))
        V = torch.reshape(V, (batch, source_seq_len, self.num_heads, self.head_size))
        # generate mask
        triangular = torch.triu(torch.ones((target_seq_len, source_seq_len), dtype=torch.bool, device=self.device), diagonal=1)
        
        query_key = torch.einsum("abcd,aecd->acbe", Q, K)
        masked_query_key = torch.where(triangular, -torch.inf, query_key)
        masked_query_key = self.softmax((masked_query_key)/sqrt_d_k)
        masked_query_key = self.dropout1(masked_query_key)
        result = torch.einsum("abcd, adbe-> acbe", masked_query_key, V)
        Z = torch.reshape(result, (batch, target_seq_len, self.num_heads * self.head_size))
        Z = self.dropout2(Z)
        #print(f"{Z.shape=}")
        Z = self.W_O(Z)
        return Z

class GPT2Block(nn.Module):
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = modules.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln2 = modules.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2Model(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        self.text_embedding = modules.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.hidden_size)
        self.position_embedding = modules.Embedding(
            num_embeddings=self.config.max_seq_len,
            embedding_dim=self.config.hidden_size
        )
        list_decoder_blocks = [GPT2Block(config = self.config) 
                                    for _ in range(self.config.num_layers)]
        self.decoder_blocks = nn.Sequential(*list_decoder_blocks)
        self.final_layer_norm = modules.LayerNorm(normalized_shape=self.config.hidden_size,eps=self.config.layer_norm_epsilon)
        # self.unembed = nn.Linear(self.config.hidden_size, config.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, dim=0)
        position = torch.arange(x.shape[1], device=self.config.device)
        x = self.text_embedding(x) + self.position_embedding(position)
        # print(f"x.shape={x.shape}")
        x = self.decoder_blocks(x)
        x = self.final_layer_norm(x)
        # x = self.unembed(x) # ,dim=2)
        x = x @ self.text_embedding.weight.T
        return x

def copy_weights(my_model: GPT2Model, pretrained_model: nn.Module, verbostiy: int=0) -> GPT2Model:
    '''Copy over the weights from gpt to your implementation of gpt.

    gpt should be imported using: 
        gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2")

    Returns your gpt model, with weights loaded in.'''

    pretraineddict = dict(pretrained_model.named_parameters())
    my_state = dict(my_model.named_parameters())

    keys_to_iterate = []

    for pretrainedkey, pretrainedvalue in pretraineddict.items():
        remove_these= ["attn.masked_bias", ".attn.bias", "lm_head.weight"]
        for suffix in remove_these:
            if pretrainedkey.endswith(suffix):
                break
        else:
            keys_to_iterate.append(pretrainedkey)

    # match keys
    remaining_keys = list(my_state.keys())
    matched_keys = {}
    for key in keys_to_iterate:
        for mykey in remaining_keys:
            if key.endswith("weight") and not (key.endswith("wte.weight") or key.endswith("wpe.weight")):
                # transpose shape
                pretrained_values = pretraineddict[key].T
                if verbostiy:
                    print(f"Copied params.T: {key} -> {mykey}")
            else:
                pretrained_values = pretraineddict[key]
                if verbostiy:
                    print(f"Copied params: {key} -> {mykey}")

            myshape = my_state[mykey].shape
            if pretrained_values.shape == myshape:
                matched_keys[mykey] = pretrained_values
                remaining_keys.remove(mykey)
                break
        else: 
            print(f"No match found for key {key}")


    # Check the number of params/buffers is correct
    assert len(matched_keys) == len(keys_to_iterate), "Number of layers is wrong. Have you done the prev step correctly?"

    # Initialise an empty dictionary to store the correct key-value pairs
    # state_dict_to_load = {}

    # for mykey, pretrainedkey in zip(matched_keys, keys_to_iterate):
    #     pretrainedvalue = pretraineddict[pretrainedkey]
    #     state_dict_to_load[mykey] = pretrainedvalue

    my_model.load_state_dict(matched_keys)

    return my_model



def get_pretrained_gpt2():
    config = TransformerConfig()
    model = GPT2Model(config)
    gpt2 = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    my_gpt = copy_weights(model, gpt2, verbostiy=0)
    return my_gpt