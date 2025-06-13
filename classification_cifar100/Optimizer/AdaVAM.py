import torch
from torch.optim.optimizer import Optimizer


class AdaVAM(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), partial=0.58, weight_decay=0):
        """
        Implements AdaVAM: Adaptive Variance-Aware Momentum for Accelerating Deep Neural Network Training.

        Args:
            params (iterable): iterable of parameters to optimize
            lr (float, optional): learning rate η (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients β₁, β₂ for momentum
                and scalar variance estimation (default: (0.9, 0.999))
            partial (float, optional): denominator scaling exponent (default: 0.58)
            weight_decay (float, optional): weight decay λ (L2 penalty) (default: 0)
        """
        defaults = dict(lr=lr, betas=betas, partial=partial, weight_decay=weight_decay)
        super(AdaVAM, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                # Step 1: Add weight decay (λθ_{t-1})
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                state = self.state[p]
                # Initialize buffers if first step
                if len(state) == 0:
                    state['step'] = 0
                    state['m_buffer'] = torch.zeros_like(p)  # Momentum buffer m
                    state['v_buffer'] = torch.zeros_like(p)  # Scalar variance buffer v

                m_buffer, v_buffer = state['m_buffer'], state['v_buffer']
                state['step'] += 1
                beta1, beta2 = group['betas']

                # Compute eps = 1/(10^{5p}) for denominator stability
                eps = 1 / (10 ** (5 * group['partial']))

                # Step 2: Normalize gradient ĝₜ
                normalized_grad = grad / torch.max(
                    torch.pow(v_buffer, group['partial']),
                    torch.tensor(eps, device=grad.device)
                )

                # Step 3: Update momentum buffer mₜ = β₁m_{t-1} + (1-β₁)ĝₜ
                m_buffer.mul_(beta1).add_(normalized_grad, alpha=1 - beta1)

                # Step 4: Update scalar variance vₜ = β₂s_{t-1} + (1-β₂)Mean(gₜ²)
                # Use mean of squared gradients for scalar variance
                v_buffer.mul_(beta2).add_(torch.mean(grad * grad), alpha=1 - beta2)

                # Step 5: Parameter update θₜ = θ_{t-1} - ηmₜ
                p.add_(m_buffer, alpha=-group['lr'])

        return loss
