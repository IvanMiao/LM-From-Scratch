from collections.abc import Callable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
	def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2) -> None:
		if lr < 0:
			raise ValueError(f"Invalid learning rate: {lr}")
		if not 0.0 <= betas[0] < 1.0:
			raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
		if not 0.0 <= betas[1] < 1.0:
			raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
		if not 0.0 <= eps:
			raise ValueError(f"Invalid epsilon value: {eps}")
		if not 0.0 <= weight_decay:
			raise ValueError(f"Invalid weight_decay value: {weight_decay}")

		defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
		super().__init__(params, defaults)

	@torch.no_grad()
	def step(self, closure:Optional[Callable] = None):
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()
		
		for group in self.param_groups:
			lr = group["lr"]
			beta1, beta2 = group["betas"]
			eps = group["eps"]
			weight_decay = group["weight_decay"]

			for p in group["params"]:
				if p.grad is None:
					continue
				grad = p.grad
				if grad.is_sparse:
					raise RuntimeError("AdamW does not suppotr sparse")
				
				state = self.state[p]

				if len(state) == 0:
					state["step"] = 0
					state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
					state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
				
				exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

				state["step"] += 1
				t = state["step"]

				exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
				exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

				bias_correction1 = 1 - beta1 ** t
				bias_correction2 = 1 - beta2 ** t

				step_size = lr / bias_correction1

				denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
				p.addcdiv_(exp_avg, denom, value=-step_size)

				# Apply weight decay (λ)
				if weight_decay != 0:
					p.add_(p, alpha=-lr * weight_decay)
		return loss


# SGD Optimizer from the assignment
class SGD(torch.optim.Optimizer):
	def __init__(self, params, lr=1e-3):
		if lr < 0:
			raise ValueError(f"Invalid learning rate: {lr}")
		defaults = {"lr": lr}
		super().__init__(params, defaults)
	
	def step(self, closure: Optional[Callable] = None):
		loss = None if closure is None else closure()
		for group in self.param_groups:
			lr = group["lr"]
			for p in group["params"]:
				if p.grad is None:
					continue

				state = self.state[p]
				t = state.get("t", 0)
				grad = p.grad.data
				p.data -= lr / math.sqrt(t + 1) * grad
				state["t"] = t + 1
		return loss


weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)
for t in range(100):
	opt.zero_grad() # Reset the gradients for all learnable parameters.
	loss = (weights**2).mean() # Compute a scalar loss value.
	print(loss.cpu().item())
	loss.backward() # Run backward pass, which computes gradients.
	opt.step() # Run optimizer step.