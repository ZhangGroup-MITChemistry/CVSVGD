import torch

from CVSVGD.base_types.cv_def import CVDef

''' Using data from CVSVGD to fit a Radial, Periodic or Mixed Basis Function (MBF) to mean forces '''
class FreeEnergyProfileMBF(torch.nn.Module):
    def __init__(self, centers: torch.Tensor, cv_defs: list[CVDef]) -> None:
        super().__init__()
        self.centers: torch.Tensor = centers.detach().clone()
        self.weights: torch.Tensor = -torch.ones((self.centers.shape[-1],), dtype=centers.dtype, device=centers.device)
        self.ndim: int = self.centers.shape[0]
        self.sigma: torch.Tensor = torch.full((self.ndim, 1), 0.1, dtype=centers.dtype, device=centers.device)
        self.device: torch.device = centers.device
        self.dtype: torch.dtype = centers.dtype
        self._periodic_mask: torch.Tensor = torch.tensor([cv.periodic for cv in cv_defs], device=self.device, dtype=torch.bool)
        self._period: torch.Tensor = torch.tensor(
            [(cv.period if cv.period is not None else 2.0 * torch.pi) if cv.periodic else 1.0 for cv in cv_defs],
            device=self.device, dtype=self.dtype,
        )

    def _r2_dr2(self, delta):
        shape = (-1,) + (1,) * (delta.dim() - 1)
        p = self._periodic_mask.view(shape)
        k = (2.0 * torch.pi / self._period).view(shape)
        r2 = torch.where(p, (1.0 - torch.cos(k * delta)) / k**2, 0.5 * delta**2)
        dr2 = torch.where(p, torch.sin(k * delta) / k, delta)
        return r2, dr2

    def fit(self, mean_forces: torch.Tensor, initial_sigma: torch.Tensor|None=None, n_iter:int=250, lr:float=1e-1, reg:float=1e-10):
        if initial_sigma is None:
            initial_sigma = self.sigma.clone()
        else:
            initial_sigma = initial_sigma.view(self.sigma.shape)
        _, K = self.centers.shape
        diff = self.centers[:, :, None] - self.centers[:, None, :]
        log_sigma = torch.log(initial_sigma).requires_grad_(True)
        opt = torch.optim.Adam([log_sigma], lr=lr, betas=(0.1, 0.9))
        best_sigma = log_sigma.detach().clone()
        loss_min = torch.inf
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.9, patience=1, cooldown=0, threshold=1e-6
        )

        def solve_a(sigma):
            raw_grad = self.get_raw_grad(diff, sigma)
            A = raw_grad.reshape(-1, raw_grad.shape[-1]).double()
            b = mean_forces.reshape(-1, 1).double()
            kc = A.shape[1]

            lam = reg * (A.norm() ** 2) / kc
            sqrt_lam = lam.sqrt()

            aug_A = torch.cat(
                [A, sqrt_lam * torch.eye(kc, device=A.device, dtype=A.dtype)], dim=0
            )
            b_aug = torch.cat(
                [b, torch.zeros(kc, 1, device=A.device, dtype=A.dtype)], dim=0
            )

            Q, R = torch.linalg.qr(aug_A, mode="reduced")
            a = torch.linalg.solve_triangular(R, Q.T @ b_aug, upper=True).squeeze(-1)
            return a.to(raw_grad.dtype), raw_grad

        for i in range(n_iter):
            opt.zero_grad()
            sigma = log_sigma.exp()
            a, raw_grad = solve_a(sigma)
            grad_A = -torch.einsum("djk,k->dj", raw_grad, a)
            loss = ((grad_A + mean_forces) ** 2).sum()
            if loss < loss_min:
                loss_min = loss
                best_sigma = log_sigma.detach().clone()
            loss.backward()
            opt.step()
            scheduler.step(loss.item())
            grad_norm = torch.norm(log_sigma.grad.detach().flatten())
            print(
                f"Iteration {i} | Loss: {loss.item():6f} | Gradient Norm: {grad_norm.item():6f}",
                flush=True,
            )

        with torch.no_grad():
            best_sigma = best_sigma.exp()
            best_a, raw_grad = solve_a(best_sigma)
            grad_A = -torch.einsum("djk,k->dj", raw_grad, best_a)
            best_residual = ((grad_A + mean_forces) ** 2).sum().sqrt().item() / K

        sigma_str = (
            f"{best_sigma.detach().cpu().numpy()}"
            if best_sigma.numel() == 1
            else str(best_sigma.tolist())
        )
        print(f"Optimal sigma = {sigma_str}, residual/center = {best_residual:.6f}")

        self.weights.data = best_a
        self.sigma.data = best_sigma

    def get_raw_grad(self, delta, sigma):
        r2, dr2 = self._r2_dr2(delta)
        u = (r2) / (sigma[:, None, :] ** 2)
        u = u.sum(0)
        w = torch.exp(-1 * u)
        u_p = -dr2 / (sigma[:, None, :] ** 2)
        rbf_force = u_p * w[None, :, :]
        return rbf_force

    def evaluate(self, x:torch.Tensor):
        sigma = self.sigma
        w = self.weights

        delta = x[:, :, None] - self.centers[:, None, :]
        r2, _ = self._r2_dr2(delta)
        u = (r2) / (sigma[:, None, :] ** 2)

        u = u.sum(0)
        pmf = (w[None, :] * torch.exp(-u)).sum(-1)
        return pmf

    def default_bounds(self, n_sigma=1.0):
        pad = n_sigma * self.sigma.reshape(self.ndim, -1).max(dim=1).values
        lo = self.centers.min(dim=1).values - pad
        hi = self.centers.max(dim=1).values + pad
        bounds = []
        for d in range(self.ndim):
            if self._periodic_mask[d]:
                L = self._period[d].item()
                bounds.append((-L / 2, L / 2))
            else:
                bounds.append((lo[d].item(), hi[d].item()))
        return bounds

    def marginalize_PMF(self, bounds:list[list[float]]|None=None, grid_density:int=50, marginal:tuple[int,...]=(0, 1), batch_size:int=1_000_000):
        if not 1 <= len(marginal) <= self.ndim or len(set(marginal)) != len(marginal):
            raise ValueError(f"marginal must be 1..{self.ndim} distinct CV indices, got {marginal}")

        hidden = [d for d in range(self.ndim) if d not in marginal]

        if bounds is None:
            bounds = self.default_bounds()
        dens = [grid_density] * self.ndim if isinstance(grid_density, int) else list(grid_density)

        def axis(d, integrated):
            lo, hi = bounds[d]
            n = dens[d]
            if integrated and self._periodic_mask[d]:
                return torch.linspace(lo, hi, n + 1, device=self.device, dtype=self.dtype)[:-1]
            return torch.linspace(lo, hi, n, device=self.device, dtype=self.dtype)

        vis_axes = [axis(d, False) for d in marginal]
        vis_pts = torch.stack(torch.meshgrid(*vis_axes, indexing="ij"), 0).reshape(len(marginal), -1)
        if hidden:
            hid_axes = [axis(d, True) for d in hidden]
            hid_pts = torch.stack(torch.meshgrid(*hid_axes, indexing="ij"), 0).reshape(len(hidden), -1)
        else:
            hid_pts = torch.zeros((0, 1), device=self.device, dtype=self.dtype)

        Pv, Ph = vis_pts.shape[1], hid_pts.shape[1]
        v_chunk = max(1, batch_size // Ph)
        h_chunk = min(Ph, batch_size)
        PMF = torch.empty(Pv, device=self.device, dtype=self.dtype)

        marg = list(marginal)

        with torch.no_grad():
            for v0 in range(0, Pv, v_chunk):
                v = vis_pts[:, v0:v0 + v_chunk]
                c = v.shape[1]
                lse = torch.full((c,), -torch.inf, device=self.device, dtype=self.dtype)
                for h0 in range(0, Ph, h_chunk):
                    h = hid_pts[:, h0:h0 + h_chunk]
                    nh = h.shape[1]
                    x = torch.empty((self.ndim, c, nh), device=self.device, dtype=self.dtype)
                    x[marg] = v[:, :, None].expand(-1, -1, nh)
                    if hidden:
                        x[hidden] = h[:, None, :].expand(-1, c, -1)
                    energy = self.evaluate(x.reshape(self.ndim, -1)).reshape(c, nh)
                    lse = torch.logaddexp(lse, torch.logsumexp(-energy, dim=1))
                PMF[v0:v0 + c] = -lse

        PMF = PMF.reshape([a.numel() for a in vis_axes])
        PMF = PMF - PMF.min()
        return PMF, bounds
