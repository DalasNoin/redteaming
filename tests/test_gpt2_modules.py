import torch as t
from torch import nn
import plotly.express as px
from IPython.display import display
import pandas as pd
import numpy as np
from models import modules

def test_embedding(Embedding):
    """Indexing into the embedding should fetch the corresponding rows of the embedding."""
    emb = Embedding(6, 100)
    out = emb(t.tensor([1, 3, 5], dtype=t.int64))
    t.testing.assert_close(out[0], emb.weight[1])
    t.testing.assert_close(out[1], emb.weight[3])
    t.testing.assert_close(out[2], emb.weight[5])

    emb = Embedding(30000, 500)
    t.testing.assert_close(emb.weight.std().item(), 1.0, rtol=0, atol=0.005)

def test_layernorm_mean_1d(LayerNorm):
    """If an integer is passed, this means normalize over the last dimension which should have that size."""
    x = t.randn(20, 10)
    ln1 = LayerNorm(10)
    out = ln1(x)
    max_mean = out.mean(-1).abs().max().item()
    assert max_mean < 1e-5, f"Normalized mean should be about 0, got {max_mean}"
    print(f"All tests in `test_layernorm_mean_1d` passed.")

def test_layernorm_mean_2d(LayerNorm):
    """If normalized_shape is 2D, should normalize over both the last two dimensions."""
    x = t.randn(20, 10)
    ln1 = LayerNorm((20, 10))
    out = ln1(x)
    max_mean = out.mean((-1, -2)).abs().max().item()
    assert max_mean < 1e-5, f"Normalized mean should be about 0, got {max_mean}"
    print(f"All tests in `test_layernorm_mean_2d` passed.")

def test_layernorm_std(LayerNorm):
    """If epsilon is small enough and no elementwise_affine, the output variance should be very close to 1."""
    x = t.randn(20, 10)
    ln1 = LayerNorm(10, eps=1e-11, elementwise_affine=False)
    out = ln1(x)
    var_diff = (1 - out.var(-1, unbiased=False)).abs().max().item()
    assert var_diff < 1e-6, f"Var should be about 1, off by {var_diff}"
    print(f"All tests in `test_layernorm_std` passed.")

def test_layernorm_exact(LayerNorm):
    """Your LayerNorm's output should match PyTorch for equal epsilon, up to floating point rounding error.
    This test uses float64 and the result should be extremely tight.
    """
    x = t.randn(2, 3, 4, 5)
    # Use large epsilon to make sure it fails if they forget it
    ln1 = LayerNorm((5,), eps=1e-2)
    ln2 = t.nn.LayerNorm((5,), eps=1e-2)  # type: ignore
    actual = ln1(x)
    expected = ln2(x)
    t.testing.assert_close(actual, expected)
    print(f"All tests in `test_layernorm_exact` passed.")

def test_layernorm_backward(LayerNorm):
    """The backwards pass should also match PyTorch exactly."""
    x = t.randn(10, 3)
    x2 = x.clone()
    x.requires_grad_(True)
    x2.requires_grad_(True)

    # Without parameters, should be deterministic
    ref = nn.LayerNorm(3, elementwise_affine=False)
    ref.requires_grad_(True)
    ref(x).sum().backward()

    ln = LayerNorm(3, elementwise_affine=False)
    ln.requires_grad_(True)
    ln(x2).sum().backward()
    # Use atol since grad entries are supposed to be zero here
    assert isinstance(x.grad, t.Tensor)
    assert isinstance(x2.grad, t.Tensor)
    t.testing.assert_close(x.grad, x2.grad, atol=1e-5, rtol=1e-5)
    print(f"All tests in `test_layernorm_backward` passed.")

def test_dropout_eval(Dropout):
    dropout = Dropout(p=0.1).eval()
    x = t.randn((3, 4))
    t.testing.assert_close(x, dropout(x), msg="Failed on eval mode (shouldn't change tensor).")
    print(f"All tests in `test_dropout_eval` passed.")

def test_dropout_training(Dropout):

    dropout = Dropout(p=0).train()
    x = t.rand((1000, 1000))
    t.testing.assert_close(x, dropout(x), msg="Failed on p=0 (dropout shouldn't change tensor)")

    for p in (0.1, 0.5, 0.9):
        dropout = Dropout(p=p).train()
        x = t.rand((1000, 1000))
        x_dropout = dropout(x)

        close_to_zero = t.abs(x_dropout) < 0.001
        fraction_close_to_zero = close_to_zero.sum() / close_to_zero.numel()
        close_to_ratio = t.abs(x_dropout / x - (1 / (1 - p))) < 0.001
        fraction_close_to_ratio = close_to_ratio.sum() / close_to_ratio.numel()

        assert abs(fraction_close_to_zero - p) < 0.01, f"p={p}, Wrong number of values set to zero"
        assert fraction_close_to_zero + fraction_close_to_ratio > 0.995, f"p={p}, Incorrect scaling"
    
    print(f"All tests in `test_dropout_training` passed.")

def plot_gelu(GELU):
    gelu = GELU()
    x = t.linspace(-5, 5, steps=100)
    px.line(x=x, y=gelu(x), template="ggplot2").show()


def print_param_count(*models, display_df=True, use_state_dict=False):
    """
    display_df: bool
        If true, displays styled dataframe
        if false, returns dataframe

    use_state_dict: bool
        If true, uses model.state_dict() to construct dataframe
            This will include buffers, not just params
        If false, uses model.named_parameters() to construct dataframe
            This misses out buffers (more useful for GPT)
    """
    df_list = []
    gmap_list = []
    for i, model in enumerate(models, start=1):
        print(f"Model {i}, total params = {sum([param.numel() for name, param in model.named_parameters()])}")
        iterator = model.state_dict().items() if use_state_dict else model.named_parameters()
        df = pd.DataFrame([
            {f"name_{i}": name, f"shape_{i}": tuple(param.shape), f"num_params_{i}": param.numel()}
            for name, param in iterator
        ]) if (i == 1) else pd.DataFrame([
            {f"num_params_{i}": param.numel(), f"shape_{i}": tuple(param.shape), f"name_{i}": name}
            for name, param in iterator
        ])
        display(df)
        df_list.append(df)
        gmap_list.append(np.log(df[f"num_params_{i}"]))
    df = df_list[0] if len(df_list) == 1 else pd.concat(df_list, axis=1).fillna(0)
    for i in range(1, len(models) + 1):
        df[f"num_params_{i}"] = df[f"num_params_{i}"].astype(int)
    if len(models) > 1:
        param_counts = [df[f"num_params_{i}"].values.tolist() for i in range(1, len(models) + 1)]
        if all([param_counts[0] == param_counts[i] for i in range(1, len(param_counts))]):
            print("All parameter counts match!")
        else:
            print("Parameter counts don't match up exactly.")
    if display_df:
        s = df.style
        for i in range(1, len(models) + 1):
            s = s.background_gradient(cmap="viridis", subset=[f"num_params_{i}"], gmap=gmap_list[i-1])
        with pd.option_context("display.max_rows", 1000):
            display(s)
    else:
        return df

def test_load_pretrained_weights(model, tokenizer):

    model.eval()
    device = next(model.parameters()).device
    
    def encode(text: str) -> t.Tensor:
        """Return a Tensor of shape (batch=1, seq)."""
        return tokenizer(text, return_tensors="pt")["input_ids"].to(device)

    prompt = "Former President of the United States of America, George"
    input_ids = encode(prompt)
    with t.inference_mode():
        output = model(input_ids)
        logits = output[0, -1] if isinstance(output, t.Tensor) else output.logits[0, -1]
    topk = t.topk(logits, k=10).indices
    next_tokens = tokenizer.batch_decode(topk.reshape(-1, 1))
    print("Prompt: ", prompt)
    print("Your model's top 10 predictions: ", next_tokens)
    assert " Washington" in next_tokens
    assert " Bush" in next_tokens

def test_make_additive_attention_mask(make_additive_attention_mask):
    from solutions_build_bert import make_additive_attention_mask as make_additive_attention_mask_soln
    arr = t.randint(low=0, high=2, size=(3, 4))
    expected = make_additive_attention_mask_soln(arr)
    actual = make_additive_attention_mask(arr)
    t.testing.assert_close(expected, actual)

test_embedding(modules.Embedding)

test_layernorm_mean_1d(modules.LayerNorm)
test_layernorm_mean_2d(modules.LayerNorm)
test_layernorm_std(modules.LayerNorm)
test_layernorm_exact(modules.LayerNorm)
test_layernorm_backward(modules.LayerNorm)

test_dropout_eval(modules.Dropout)
test_dropout_training(modules.Dropout)