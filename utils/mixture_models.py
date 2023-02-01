
import typing

import mpmath
import numpy as np

import torch
import torch.nn as nn


class vMFLogPartition(torch.autograd.Function):
    """
    Evaluates log C_d(kappa) for vMF density.
    Allows autograd with respect to kappa.
    """
    besseli = np.vectorize(mpmath.besseli)
    log = np.vectorize(mpmath.log)
    nhlog2pi = -0.5 * np.log(2 * np.pi)

    @staticmethod
    def forward(ctx, *args):
        '''
        In the forward pass, we receive a Tensor containing the input and return
        a tensor containing the output. ctx is a context object that can be used to
        stash information for backward computation. You can cache arbitrary objects
        for use in the backward pass using the ctx.save_for_backward method.
        Arguments:
            args[0] = d; scalar (>0)
            args[1] = kappa; (>0) torch tensor of any shape
        Returns:
            logC = logC_d(kappa); torch tensor of the same shape as kappa
        '''
        d = args[0]
        kappa = args[1]

        s = 0.5 * d - 1

        # log I_s(\kappa)
        mp_kappa = mpmath.mpf(1.0) * kappa.detach().cpu().numpy()
        mp_logI = vMFLogPartition.log(vMFLogPartition.besseli(s, mp_kappa))
        logI = torch.from_numpy(np.array(mp_logI.tolist(), dtype=float)).to(kappa.device)

        # raise error if nan
        if (logI != logI).sum().item() > 0:
            raise ValueError("NaN detected from the output of log-besseli() function.")

        logC = d * vMFLogPartition.nhlog2pi + s * kappa.log() - logI

        # save for backward()
        ctx.s, ctx.mp_kappa, ctx.logI = s, mp_kappa, logI

        return logC

    @staticmethod
    def backward(ctx, *grad_output):
        """
        In the backward pass, we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        s, mp_kappa, logI = ctx.s, ctx.mp_kappa, ctx.logI

        # log I_{s+1}{kappa}
        mp_logI2 = vMFLogPartition.log(vMFLogPartition.besseli(s+1, mp_kappa))
        logI2 = torch.from_numpy(np.array(mp_logI2.tolist(), dtype=float)).to(logI.device)

        # raise error if nan
        if (logI2 != logI2).sum().item() > 0:
            raise ValueError('NaN is detected from the output of log-besseli() function.')

        dlogC_dkappa = -(logI2 - logI).exp()

        return None, grad_output[0] * dlogC_dkappa


class vMF(nn.Module):
    """
    Pytorch implementation of the von Mises-Fisher distribution.
        The density function is defined as: vMF(x; mu, kappa).
    """
    def __init__(self,
                 x_dim: int,
                 reg: float = 1e-6,
                 ) -> None:
        super(vMF, self).__init__()

        self.x_dim: int = x_dim
        self.reg = reg  # TODO: eps? just remove...

        # Do not assign or access directly. Use `set_params` and `get_params` methods instead.
        self.mu_unnorm = nn.Parameter(torch.randn(x_dim))
        self.log_kappa = nn.Parameter(0.01 * torch.randn([]))

    def reset(self) -> None:
        device = self.mu_unnorm.device
        self.mu_unnorm.data.copy_(torch.randn(self.x_dim, device=device))
        self.log_kappa.data.copy_(0.01 * torch.randn([], device=device))

    @torch.no_grad()
    def set_params(self, mu: torch.Tensor, kappa: torch.Tensor) -> None:
        """Set parameters."""
        self.mu_unnorm.copy_(mu)
        self.log_kappa.copy_(torch.log(kappa + 1e-10))

    def get_params(self) -> typing.Tuple[torch.Tensor]:
        """Retrieve parameters."""
        mu = self.mu_unnorm / self.norm(self.mu_unnorm, p=2, dim=0)
        kappa = self.log_kappa.exp() + self.reg
        return mu, kappa

    def forward(self, x: torch.FloatTensor, utc: bool = False) -> torch.FloatTensor:
        """
        Evaluates log-likelihood; log p(x).
        Arguments:
            x: 2d FloatTensor of shape (B, d), batch of features.
                Make sure that the input is L2-normalized.
            utc: bool, whether to evaluate only up to constant or exactly.
        Returns:
            log_likelihood of shape (B, ).
        """
        mu, kappa = self.get_params()
        dotp = torch.sum(mu.unsqueeze(0) * x, dim=1, keepdim=False)  # (1, d) * (B, d) -> (B, d) -> (B, ); mu * r
        if utc:
            log_likelihood = kappa * dotp
        else:
            logC = vMFLogPartition.apply(self.x_dim, kappa)          # see arguments of `.forward` function
            log_likelihood = kappa * dotp + logC

        return log_likelihood  # (B, )

    @torch.no_grad()
    def sample(self, N: int = 1, rsf: int = 10):
        """
        Note that this function does not support auto-differentiation.
        Arguments:
            N: int, number of samples to generate
            rsf: int, multiplicative factor for extra backup samples in rejection sampling
        Returns:
            samples: torch.FloatTensor, N samples generated.
        """

        d: int = self.x_dim
        mu, kappa = self.get_params()

        # Step 1. sample uniform unit vectors in R^{d-1}
        v = torch.randn(N, d-1).to(mu.device)
        v = v.div(self.norm(v, p=2, dim=1))

        # Step 2. sample v0

        # Step 3. Form x = [v0; sqrt(1-v0^2)*v]

        # Step 4. householder information

        raise NotImplementedError

    @staticmethod
    def norm(x: torch.FloatTensor, p: int = 2, dim: int = 0, eps: float = 1e-12) -> torch.FloatTensor:
        """
        Helper function which:
            1. Computes the p-norm
            2. Keeps the dimension
            3. Avoids zero division
            4. Returns it as the same shape as x using `expand`.
                Unlike `torch.repeat`, `torch.expand` does not allocate new memory.
        Arguments:
            x: torch.FloatTensor,
            p: int,
            dim: int,
            eps: float
        Returns:
            p-normalized torch.FloatTensor with shape equivalent to that of x.
        """
        return x.norm(p=p, dim=dim, keepdim=True).clamp(min=eps).expand_as(x)


class vMFMixture(nn.Module):
    def __init__(self,
                 x_dim: int,         # TODO: rename to `n_features`
                 order: int,         # TODO: rename to `n_components`
                 reg: float = 1e-6,  # TODO: rename to `eps`
                 ) -> None:
        super(vMFMixture, self).__init__()

        self.x_dim: int = x_dim  # number of features
        self.order: int = order  # number of components
        self.reg: float = reg    # epsilon

        # un-normalized mixture weights; softmax is used to normalize them
        self.alpha_logit = nn.Parameter(0.01 * torch.randn(self.n_components))

        # list of mixture components
        self.components = nn.ModuleList(
            [vMF(self.n_features, self.eps) for _ in range(self.n_components)]
        )

    def reset(self) -> None:
        """Reset parameters to initial values."""
        device = self.alpha_logit.device
        self.alpha_logit.data.copy_(0.01 * torch.randn(self.n_components, device=device))
        for k in range(self.n_components):
            self.components[k].reset()

    @property
    def n_features(self) -> int:  # TODO: remove later
        return self.x_dim

    @property
    def n_components(self) -> int:  # TODO: remove later
        return self.order

    @property
    def eps(self) -> float:  # TODO: remove later
        return self.reg

    @torch.no_grad()
    def set_params(self, alpha, mus, kappas) -> None:
        """Set paramters of mixture components."""

        self.alpha_logit.copy_(torch.log(alpha + 1e-10))

        for k in range(self.n_components):
            # Assign mean directions
            if hasattr(self.components[k], 'mu_unnorm'):
                assert isinstance(self.components[k].mu_unnorm, nn.Parameter)
                self.components[k].mu_unnorm.copy_(mus[k])
            else:
                raise AttributeError
            # Assign concentration parameters
            if hasattr(self.components[k], 'log_kappa'):
                assert isinstance(self.components[k].log_kappa, nn.Parameter)
                self.components[k].log_kappa.copy_(torch.log(kappas[k] + 1e-10))
            else:
                raise AttributeError

    def update_params(self, alpha, mus, kappas) -> None:
        self.set_params(alpha, mus, kappas)

    @torch.no_grad()
    def get_params(self) -> typing.Tuple[torch.FloatTensor]:
        """Get parameters of mixture components."""

        logalpha = self.alpha_logit.log_softmax(dim=0)  # (K, )

        mus, kappas = list(), list()
        for k in range(self.n_components):
            mu, kappa = self.components[k].get_params()
            mus += [mu]
            kappas += [kappa]

        mus = torch.stack(mus, dim=0)        # K * (d, ) -> (K, d)
        kappas = torch.stack(kappas, dim=0)  # K * ( , ) -> (K,  )

        return logalpha, mus, kappas         # (K,  ), (K, d), (K,  )

    def forward(self,
                x: torch.FloatTensor,
                return_log_prob: bool = True) -> typing.Union[torch.FloatTensor,
                                                              typing.Tuple[torch.FloatTensor]]:
        """
        Evaluate log-likelihood; log p(x).
        Arguments:
            x: 2D torch.FloatTensor of shape (N, d).
        Returns:
            log_likelihood = log p(x); having shape (N, )
            logpcs = log p_k(x|k); having shape (N, K)
                Notes:
                    `log_likelihood` is calculated using `logalpha` and `logpcs`.
        """
        
        if return_log_prob:
            # To return log probabilities, the `__score` method is not sufficient.
            log_prob = self._estimate_log_prob(x=x)
            weighted_log_prob = log_prob + self.logalpha
            log_likelihood = torch.logsumexp(weighted_log_prob, dim=1)
            return log_likelihood, log_prob           # (N,  ), (N, K)
        else:
            return self.__score(x, reduction='none')  # (N,  )

    def sample(self, N: int = 1, rsf: int = 10):
        raise NotImplementedError

    def _estimate_log_prob(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Returns the estimated log-likelihood of N samples belonging to the k-th mixture component.
        Arguments:
            x: `torch.FloatTensor` of shape (N, d)
        Returns:
            log_prob: `torch.FloatTensor` of shape (N, K)
        """
        log_prob = list()
        for k in range(self.n_components):
            log_prob += [self.components[k](x)]  # forward function of component returns log-likelihood
        return torch.stack(log_prob, dim=1)  # (N, K) <- (N, ) * K

    @torch.no_grad()
    def fit(self,
            x: torch.FloatTensor,
            n_iter: int = 100,
            tol: float = 1e-5,
            verbose: bool = False,
            precise: bool = False,
            ) -> None:
        """
        Fits von Mises-Fisher mixture model to data (x).
        Arguments:
            x: torch.FloatTensor of shape (N, d)
            n_iter: int, maximum number of EM iterations.
            verbose: bool, controls logging verbosity.
        """
        j = -np.inf
        for i in range(n_iter):
            
            log_likelihood = self.__em(x=x)
            if precise:
                log_likelihood = self.__score(x=x, reduction='sum')

            # check tolerance
            if i > 0:
                rel_imp = (log_likelihood - j).abs().div(j.abs() + 1e-6)
                if rel_imp < tol:
                    if verbose:
                        print(f"Terminating EM with relative improvement {rel_imp:.6f} < {tol:.6f} ")
                    break
            
            # update current log-likelihood
            j = log_likelihood

        ll: torch.FloatTensor = self.__score(x, reduction='sum')
        if verbose:
            print(f"Training finished with log-likelihood={ll.item():.4f} ")

    @torch.no_grad()
    def predict(self, x: torch.FloatTensor, probs: bool = False):
        """
        Assigns input data `x` to one of the mixture components by evaluating the likelihood
        under each. If `probs=True`, returns normalized probabilities of class membership.
        Arguments:
            x: torch.FloatTensor of shape (N, d)
            probs: bool, if True returns probabilities of class membership.
        Returns:
            p_k: torch.FloatTensor of shape (N, K) if `probs=True`.
            or
            y: torch.LongTensor of shape (N)
        """
        log_prob = self._estimate_log_prob(x=x)
        weighted_log_prob = log_prob + self.logalpha
        if probs:
            return weighted_log_prob.softmax(dim=1)  # p_k
        else:
            _, idx = torch.max(weighted_log_prob, dim=1)
            return torch.squeeze(idx).long()

    def predict_proba(self, x: torch.FloatTensor):
        """
        Returns normalized probabilities of class membership.
        Arguments:
            x: torch.FloatTensor of shape (N, d)
        Returns:
            p_k: torch.FloatTensor of shape (N, K) with each row summing up to one.
        """
        return self.predict(x, probs=True)

    @torch.no_grad()
    def _e_step(self, x: torch.FloatTensor) -> typing.Tuple[torch.FloatTensor]:
        """
        Expectation step.
        Arguments:
            x: torch.FloatTensor of shape (N, d)
        Returns:
            resp: torch.FloatTensor of shape (N, K)
            total_log_likelihood: torch.FloatTensor of shape (1,  )
        """
        # Compute responsibilities; q(z=k|x)
        log_likelihood, log_prob = self.forward(x)     # (N,  ), (N, K)
        total_log_likelihood = log_likelihood.sum()    # (1,  ); total log-likelihood
        weighted_log_prob = log_prob + self.logalpha   # (N, K) <- (N, K) + (1, K)
        resp = weighted_log_prob.softmax(dim=1)        # (N, K); responsibilities

        return resp, total_log_likelihood

    @torch.no_grad()
    def _m_step(self, x: torch.FloatTensor, resp: torch.FloatTensor) -> typing.Tuple[torch.FloatTensor]:
        """
        Maximization step.
        Arguments:
            x: torch.FloatTensor of shape (N, d)
            resp: torch.FloatTensor of shape (N, K)
        Returns:
            alpha_new:
            mus_new:
            kappas_new:
        """
        # Compute new parameters; alpha, mu, kappa
        qzx = (resp.unsqueeze(2) * x.unsqueeze(1)).sum(dim=0) # (N, K, 1) * (N, 1, d) -> (N, K, d) -> (K, d)
        qzx_norms = vMF.norm(qzx, p=2, dim=1)                 # (K, d)
        mus_new = qzx.div(qzx_norms)                          # (K, d) / (K, d)
        Rs = qzx_norms[:, 0].div(resp.sum(dim=0) + 1e-6)      # (K,  ) / (K,  ); Bessel ratio
        kappas_new = (self.n_features * Rs - torch.pow(Rs, 3)).div(1 - torch.pow(Rs, 2))
        alpha_new = resp.sum(dim=0).div(x.shape[0])          # (K,  )

        return alpha_new, mus_new, kappas_new

    @torch.no_grad()
    def __em(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Perform a single iteration of expectation-maximization.
        Arguments:
            x: torch.FloatTensor of shape (N, d)
        Returns:
            total_log_likelihood: torch.FloatTensor of shape (1,  )
        """
        resp, total_log_likelihood = self._e_step(x=x)
        alpha, mus, kappas = self._m_step(x=x, resp=resp)
        self.update_params(alpha=alpha, mus=mus, kappas=kappas)

        return total_log_likelihood

    def __score(self, x: torch.FloatTensor, reduction: str = 'none') -> torch.FloatTensor:
        """
        Computes the log-likelihood of the data under the current model parameters.
        Arguments:
            x: torch.FloatTensor of shape (N, d)
            as_average: bool;
        """
        weighted_log_prob = self._estimate_log_prob(x) + self.logalpha  # (N, K) <- (N, K) + (1, K)
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)    # (N,  )

        if reduction == 'mean':
            return per_sample_score.mean()          # (1, )
        elif reduction == 'sum':
            return per_sample_score.sum()           # (1, )
        elif reduction == 'none':
            return torch.squeeze(per_sample_score)  # (N, )
        else:
            raise NotImplementedError

    @property
    def logalpha(self) -> torch.FloatTensor:
        return self.alpha_logit.log_softmax(dim=0).unsqueeze(0)  # (1, K)
