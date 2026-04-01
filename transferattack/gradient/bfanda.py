import torch
import math
from torch.nn import functional as F

from ..utils import *
from ..attack import Attack


class BFANDA(Attack):
    """
    BF-ANDA Attack
    Boundary-based ANDA with multiple boundary point sampling
    Combines ANDA's ensembled asymptotically normal distribution learning with boundary fitting
    """
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, n_ens=25, aug_max=0.3, sample=False, targeted=False, random_start=False,
                 norm='linfty', loss='crossentropy', device=None, attack='BF-ANDA', 
                 num_boundary_points=10, max_num_move=20, shrinkage_factor=0.8, initial_direction_std=1.0, decay=0.0, **kwargs):
        """
        Arguments:
            num_boundary_points (int): number of boundary points to sample
            max_num_move (int): maximum number of moves to find boundary
            shrinkage_factor (float): shrinkage factor for boundary search
            initial_direction_std (float): standard deviation for initial direction sampling
            decay (float): the decay factor for momentum calculation (default 0.0 for ANDA compatibility)
        """
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.n_ens = n_ens
        self.aug_max = aug_max
        self.sample = sample
        self.num_boundary_points = num_boundary_points
        self.max_num_move = max_num_move
        self.shrinkage_factor = shrinkage_factor
        self.initial_direction_std = initial_direction_std

        def is_sqr(n):
            a = int(math.sqrt(n))
            return a * a == n
        assert is_sqr(self.n_ens), "n_ens must be square number."

        self.thetas = self.get_thetas(int(math.sqrt(self.n_ens)), -self.aug_max, self.aug_max)

    def get_theta(self, i, j):
        theta = torch.tensor([[[1, 0, i], [0, 1, j]]], dtype=torch.float)
        return theta

    def get_thetas(self, n, min_r=-0.5, max_r=0.5):
        range_r = torch.linspace(min_r, max_r, n)
        thetas = []
        for i in range_r:
            for j in range_r:
                thetas.append(self.get_theta(i, j))
        thetas = torch.cat(thetas, dim=0)
        return thetas

    def forward(self, data, label, **kwargs):
        """
        The BF-ANDA attack procedure

        Arguments:
            data (N, C, H, W): tensor for input images
            labels (N,): tensor for ground-truth labels if untargeted
            labels (2,N): tensor for [ground-truth, targeted labels] if targeted
        """
        
        assert data.shape[0] == 1, "BF-ANDA currently only supports batchsize=1"
        if label.ndim == 2:
            assert label.shape[1] == 1, "BF-ANDA currently only supports batchsize=1"
        else:
            assert label.shape[0] == 1, "BF-ANDA currently only supports batchsize=1"

        if self.targeted:
            assert len(label) == 2
            label = label[1]  # the second element is the targeted label tensor
        
        data = data.clone().detach().to(self.device)
        xt = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        min_x = data - self.epsilon
        max_x = data + self.epsilon

        data_shape = data.shape[1:]
        stat = ANDA_STATISTICS(data_shape=(1,) + data_shape, device=self.device)

        # Initialize momentum
        momentum_grad = torch.zeros_like(data)

        for i in range(self.epoch):
            grad_boundary = []
            
            # Sample multiple boundary points
            for b in range(self.num_boundary_points):
                # Sample random direction di ~ N(0, σ²I)
                di = torch.randn_like(xt) * (self.initial_direction_std / 255.0)
                
                # Get original prediction
                with torch.no_grad():
                    original_output = self.get_logits(xt)
                    original_pred = torch.argmax(original_output, dim=1)
                
                # Find boundary point by moving along di
                boundary_x = self.find_boundary_point(xt, di, original_pred)
                
                # Augment the boundary data
                boundary_x_batch = boundary_x.repeat(self.n_ens, 1, 1, 1)
                boundary_x_batch.requires_grad = True            
                aug_boundary_x_batch = self.transform(thetas=self.thetas, data=boundary_x_batch)
                labels = label.repeat(boundary_x_batch.shape[0])

                # Obtain the output
                logits = self.get_logits(aug_boundary_x_batch)

                # Calculate the loss
                loss = self.get_loss(logits, labels)

                # Calculate the gradients
                grad = self.get_grad(loss, boundary_x_batch)
                
                grad_boundary.append(grad.detach())
            
            # Average all boundary gradients
            avg_boundary_grad = torch.stack(grad_boundary).mean(dim=0)
            
            # Collect the grads into stat
            stat.collect_stat(avg_boundary_grad)  # Removed unsqueeze(0) to match expected shape

            # Get mean of grads
            sample_noise = stat.noise_mean

            if self.sample and i == self.epoch - 1:
                # Sample noise
                sample_noises = stat.sample(n_sample=1, scale=1)
                sample_xt = self.alpha * sample_noises.squeeze().sign() + xt
                sample_xt = torch.clamp(sample_xt, 0.0, 1.0).detach()
                sample_xt = torch.max(torch.min(sample_xt, max_x), min_x).detach()

            # Update momentum
            momentum_grad = self.decay * momentum_grad + sample_noise / torch.norm(sample_noise, p=1)
            
            # Update adv using sign of momentum gradient
            xt = xt + self.alpha * momentum_grad.sign()

            # Clamp data into valid range
            xt = torch.clamp(xt, 0.0, 1.0).detach()
            xt = torch.max(torch.min(xt, max_x), min_x).detach()
        
        if self.sample:
            adv = sample_xt.detach().clone()
        else:
            adv = xt.detach().clone()
        
        delta = adv - data

        return delta.detach()
    
    def find_boundary_point(self, x, direction, original_pred):
        """
        Find boundary point by binary search along direction
        
        Arguments:
            x: current adversarial example
            direction: search direction
            original_pred: original prediction class
        
        Returns:
            boundary_x: point on decision boundary
        """
        # Initialize rho to move out of decision boundary
        rho = 1.0
        moved_x = torch.clamp(x + rho * direction, 0.0, 1.0)
        
        # Move out of original classification region
        with torch.no_grad():
            moved_output = self.get_logits(moved_x)
            moved_pred = torch.argmax(moved_output, dim=1)
        
        num_moves = 0
        while torch.equal(moved_pred, original_pred) and num_moves < self.max_num_move:
            rho *= 2.0
            moved_x = torch.clamp(x + rho * direction, 0.0, 1.0)
            
            with torch.no_grad():
                moved_output = self.get_logits(moved_x)
                moved_pred = torch.argmax(moved_output, dim=1)
            num_moves += 1
        
        # Binary search to find boundary point
        while not torch.equal(moved_pred, original_pred) and rho > 1e-6:
            rho *= self.shrinkage_factor
            moved_x = torch.clamp(x + rho * direction, 0.0, 1.0)
            
            with torch.no_grad():
                moved_output = self.get_logits(moved_x)
                moved_pred = torch.argmax(moved_output, dim=1)
        
        return moved_x.clone().detach()
    
    def transform(self, thetas, data):
        grids = F.affine_grid(thetas, data.size(), align_corners=False).to(data.device)
        output = F.grid_sample(data, grids, align_corners=False)
        return output
    
    def get_loss(self, logits, labels):
        loss = F.cross_entropy(logits, labels, reduction="sum")
        return loss


class ANDA_STATISTICS:
    def __init__(self, device, data_shape=(1, 3, 224, 224)):
        self.data_shape = data_shape
        self.device = device

        self.n_models = 0
        self.noise_mean = torch.zeros(data_shape, dtype=torch.float).to(device)
        self.noise_cov_mat_sqrt = torch.empty((0, np.prod(data_shape)), dtype=torch.float).to(device)

    def sample(self, n_sample=1, scale=0.0, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        mean = self.noise_mean
        cov_mat_sqrt = self.noise_cov_mat_sqrt

        if scale == 0.0:
            assert n_sample == 1
            return mean.unsqueeze(0)

        assert scale == 1.0
        k = cov_mat_sqrt.shape[0]
        cov_sample = cov_mat_sqrt.new_empty((n_sample, k), requires_grad=False).normal_().matmul(cov_mat_sqrt)
        cov_sample /= (k - 1)**0.5

        rand_sample = cov_sample.reshape(n_sample, *self.data_shape)
        sample = mean.unsqueeze(0) + scale * rand_sample
        sample = sample.reshape(n_sample, *self.data_shape)
        return sample

    def collect_stat(self, noise):
        mean = self.noise_mean
        cov_mat_sqrt = self.noise_cov_mat_sqrt
        assert noise.device == cov_mat_sqrt.device
        bs = noise.shape[0]
        # first moment
        mean = mean * self.n_models / (self.n_models + bs) + noise.data.sum(dim=0, keepdim=True) / (self.n_models + bs)

        # square root of covariance matrix
        dev = (noise.data - mean).view(bs, -1)
        cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev), dim=0)

        self.noise_mean = mean
        self.noise_cov_mat_sqrt = cov_mat_sqrt
        self.n_models += bs

    def clear(self):
        self.n_models = 0
        self.noise_mean = torch.zeros(self.data_shape, dtype=torch.float).to(self.device)
        self.noise_cov_mat_sqrt = torch.empty((0, np.prod(self.data_shape)), dtype=torch.float).to(self.device)