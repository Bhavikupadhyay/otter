# Otter

Otter is a lightweight C++ autodiff library. It supports automatic differentiation, multi-backend execution, and a tensor API designed to be installed and used as a library dependency.

```cpp
Tensor loss = W.matmul(x).add(b).sum();
loss.backward();
// W.grad() and b.grad() are now populated
```

Otter traces a computation graph during the forward pass and reverses it on `.backward()`. Gradients accumulate into leaf tensors. The graph is freed after each pass unless you keep it.

---

## Tensor operations

- Elementwise: `add`, `mul`, `sub`, `div`
- Matrix multiply: `matmul`
- Reductions: `sum`, `mean`
- Activation functions: `relu`, `exp`, `log`
- Views: `view`, `reshape`, `transpose`, `contiguous`
- In-place: `fill_`, `zero_grad`

## Autograd

- Reverse-mode automatic differentiation
- `loss.backward()` — computes and accumulates gradients
- `retain_graph` — keeps the graph alive for multiple backward passes
- `detach()` — returns a tensor with no grad history
- `no_grad` guard — disables graph construction for inference

## Backends

Otter separates the tensor abstraction from the compute backend. CPU is the current target. CUDA is the planned second backend. Adding a backend does not require changes to `Tensor` or `Operation` code.

- CPU — functional
- CUDA — planned

Tensors can be moved between devices with `.to(device)`.

## Types

- `float64`
