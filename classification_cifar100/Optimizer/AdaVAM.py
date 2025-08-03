import torch
from torch.optim.optimizer import Optimizer


class AdaVAM(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), q=0.58, weight_decay=0, decoupled_weight_decay=False):
        """
        Implements AdaVAM (Adaptive Variance-Aware Momentum) optimizer.

        Args:
            params (iterable): iterable of parameters to optimize
            lr (float, optional): learning rate η (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients β₁, β₂ for momentum
                and scalar variance estimation (default: (0.9, 0.999))
            q (float, optional): denominator scaling exponent (default: 0.58)
            weight_decay (float, optional): weight decay λ (L2 penalty) (default: 0)
        """
        defaults = dict(lr=lr, betas=betas, q=q, weight_decay=weight_decay,
                        decoupled_weight_decay=decoupled_weight_decay)
        super(AdaVAM, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Mathematical Formulation:
        For each parameter θ at timestep t:
        1. Compute gradient with weight decay:
            gₜ = ∇L(θ_{t-1}) + λθ_{t-1}
        2. Compute normalized gradient:
            ĝₜ = gₜ / max(v_{t-1}^q, χ²) where χ²=1/(10^{5q})
        3. Update momentum buffer:
            mₜ = β₁m_{t-1} + (1-β₁)ĝₜ
        4. Update scalar variance estimate:
            vₜ = β₂v_{t-1} + (1-β₂)Mean(gₜ²)
        5. Parameter update:
            θₜ = θ_{t-1} - ηmₜ
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if group['decoupled_weight_decay']:
                    p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                else:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m_buffer'] = torch.zeros_like(p)
                    state['v_buffer'] = torch.tensor(0.0, device=p.device, dtype=p.dtype)

                m_buffer = state['m_buffer']
                v_buffer = state['v_buffer']
                state['step'] += 1
                beta1, beta2 = group['betas']

                chi2 = 1 / (10 ** (5 * group['q']))


                normalized_grad = grad / torch.max(
                    torch.pow(v_buffer, group['q']),
                    torch.tensor(chi2, device=grad.device, dtype=grad.dtype)
                )

                m_buffer.mul_(beta1).add_(normalized_grad, alpha=1 - beta1)

                v_buffer.mul_(beta2).add_(torch.mean(grad * grad), alpha=1 - beta2)

                p.add_(m_buffer, alpha=-group['lr'])

        return loss

