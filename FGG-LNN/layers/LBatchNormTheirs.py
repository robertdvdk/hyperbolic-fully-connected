import torch
import torch.nn as nn
from .lorentz import Lorentz
from geoopt import ManifoldParameter

class LorentzBatchNorm(nn.Module):
    """ Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: Lorentz, num_features: int):
        super(LorentzBatchNorm, self).__init__()
        self.manifold = manifold

        self.beta = ManifoldParameter(self.manifold.origin(num_features), manifold=self.manifold)
        self.gamma = torch.nn.Parameter(torch.ones((1,)))
        self.eps = 1e-5

        # running statistics
        self.register_buffer('running_mean', torch.zeros(num_features - 1))
        self.register_buffer('running_var', torch.ones((1,)))

    def forward(self, x, momentum=0.1):
        assert (len(x.shape)==2) or (len(x.shape)==3), "Wrong input shape in Lorentz batch normalization."

        beta = self.beta

        if self.training:
            # Compute batch mean
            mean = self.manifold.centroid(x)
            if len(x.shape) == 3:
                mean = self.manifold.centroid(mean)

            # Transport batch to origin (center batch)
            x_T = self.manifold.logmap(mean, x)
            x_T = self.manifold.transp0back(mean, x_T)

            # Compute Fréchet variance
            if len(x.shape) == 3:
                var = torch.mean(torch.norm(x_T, dim=-1), dim=(0,1))
            else:
                var = torch.mean(torch.norm(x_T, dim=-1), dim=0)

            # Rescale batch
            x_T = x_T*(self.gamma/(var+self.eps))

            # Transport batch to learned mean
            x_T = self.manifold.transp0(beta, x_T)
            output = self.manifold.expmap(beta, x_T)

            # Save running parameters
            with torch.no_grad():
                running_mean = self.manifold.expmap0(self.running_mean)
                means = torch.concat((running_mean.unsqueeze(0), mean.detach().unsqueeze(0)), dim=0)
                self.running_mean.copy_(
                    self.manifold.logmap0(
                        self.manifold.centroid(
                            means,
                            weights=torch.tensor(((1 - momentum), momentum), device=means.device),
                        )
                    )
                )
                self.running_var.copy_((1 - momentum)*self.running_var + momentum*var.detach())

        else:
            # Transport batch to origin (center batch)
            running_mean = self.manifold.expmap0(self.running_mean)
            x_T = self.manifold.logmap(running_mean, x)
            x_T = self.manifold.transp0back(running_mean, x_T)

            # Rescale batch
            x_T = x_T*(self.gamma/(self.running_var+self.eps))

            # Transport batch to learned mean
            x_T = self.manifold.transp0(beta, x_T)
            output = self.manifold.expmap(beta, x_T)



        return output

class LorentzBatchNorm1d(LorentzBatchNorm):
    """ 1D Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: Lorentz, num_features: int):
        super(LorentzBatchNorm1d, self).__init__(manifold, num_features)

    def forward(self, x, momentum=0.1):
        return super(LorentzBatchNorm1d, self).forward(x, momentum)

class LorentzBatchNorm2d(LorentzBatchNorm):
    """ 2D Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: Lorentz, num_features: int):
        super(LorentzBatchNorm2d, self).__init__(manifold, num_features)

    def forward(self, x, momentum=0.1):
        """ x has to be in channel last representation -> Shape = bs x H x W x C """
        bs, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(bs, -1, c)
        x = super(LorentzBatchNorm2d, self).forward(x, momentum)
        x = x.reshape(bs, h, w, c).permute(0, 3, 1, 2)

        return x
