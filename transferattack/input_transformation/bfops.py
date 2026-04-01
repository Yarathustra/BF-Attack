import torch
import random
import functools
import torch.nn.functional as F
from torchvision.transforms import functional as TFF

from ..utils import *
from ..gradient.bfmifgsm import BF_MIFGSM

class BF_OPS(BF_MIFGSM):
    """
    BF-OPS Attack
    Boundary-based Operator-Perturbation-based Stochastic optimization
    Combines boundary sampling from BF-MIFGSM with operator and perturbation sampling from OPS

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        beta (float): scaling factor for sample radius.
        epoch (int): the number of iterations.
        num_sample_neighbor (int): number of neighbors to sample.
        num_sample_operator (int): number of operators to sample.
        sample_levels (range): levels of operator composition.
        sample_ratios (np.array): ratios for perturbation sampling.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        num_boundary_points (int): number of boundary points to sample.
        max_num_move (int): maximum number of moves to find boundary.
        shrinkage_factor (float): shrinkage factor for boundary search.
        initial_direction_std (float): standard deviation for initial direction sampling.
        device (torch.device): the device for data.

    Official arguments:
        epsilon=16/255, beta=2., epoch=10, num_sample_neighbor=10, num_sample_operator=20,
        sample_levels=range(2, 5), sample_ratios=np.arange(0., 1.5, 0.25) + 0.25,
        num_boundary_points=10, max_num_move=20, shrinkage_factor=0.8, initial_direction_std=1.0

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/bfops/resnet50 --attack bfops --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/bfops/resnet50 --eval
    """

    def __init__(self, model_name, epsilon=16/255, beta=2., epoch=10,
                 num_sample_neighbor=10, num_sample_operator=20,
                 sample_levels=range(2, 5), sample_ratios=np.arange(0., 1.5, 0.25) + 0.25,
                 decay=1., targeted=False, random_start=False, norm='linfty',
                 loss='crossentropy', num_boundary_points=10, max_num_move=20,
                 shrinkage_factor=0.8, initial_direction_std=1.0, device=None, **kwargs):
        super().__init__(model_name, epsilon, epsilon/epoch, epoch, decay, targeted,
                        random_start, norm, loss, num_boundary_points, max_num_move,
                        shrinkage_factor, initial_direction_std, device)

        self.using_sampling = (num_sample_operator * num_sample_neighbor > 0)

        if self.using_sampling:
            # NOTE: operator sampling
            self.num_sample_operator = num_sample_operator
            self.basic_ops = [
                identity, vertical_flip, horizontal_flip, vertical_shift, horizontal_shift,
                rotate(5), rotate(-5), rotate(15), rotate(-15), rotate(45), rotate(-45),
                rotate(90), rotate(-90), rotate(180),
                scaling(2), scaling(3), scaling(4), scaling(5), scaling(6), scaling(7), scaling(8),
                dim(1.1), dim(1.3), dim(1.5), dim(1.7), dim(1.9), dim(2.1), dim(2.3),
                dim(2.5), dim(2.7), dim(2.9),
            ]
            self.sample_levels = sample_levels
            self.op_list = []
            self.num_extra_ops = len(self.basic_ops)

            # NOTE: perturbation sampling
            self.num_sample_neighbor = num_sample_neighbor
            self.sample_radius = beta * epsilon * sample_ratios
            self.eps_list = []
            self.num_extra_eps = self.num_sample_neighbor

    # NOTE: operator sampling
    @property
    def op_num(self):
        return len(self.op_list)

    def get_new_ops(self, k=2):
        sel_ops = random.choices(self.basic_ops, k=k)
        new_op = lambda x: x
        new_op = functools.reduce(lambda f, g: lambda x: f(g(x)), sel_ops, new_op)
        return new_op

    def expand_op_list(self, k=2):
        for _ in range(self.num_extra_ops):
            self.op_list.append(self.get_new_ops(k=k))

    def init_op_list(self):
        self.op_list = []
        for level in self.sample_levels:
            if level == 1:
                self.op_list.extend(self.basic_ops.copy())
            else:
                self.expand_op_list(level)

    # NOTE: perturbation sampling
    @property
    def eps_num(self):
        return len(self.eps_list)

    def expand_eps_list(self, delta, radius=1.):
        shape = (self.num_extra_eps, *delta.shape[1:])
        noise = torch.zeros(shape).uniform_(-radius, radius).to(self.device)
        self.eps_list.extend(noise)

    def init_eps_list(self, delta):
        self.eps_list = []
        for radius in self.sample_radius:
            self.expand_eps_list(delta, radius)

    def forward(self, data, label, **kwargs):
        """
        The BF-OPS attack procedure

        Arguments:
            data (N, C, H, W): tensor for input images
            label (N,): tensor for ground-truth labels if untargeted
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)
        if self.using_sampling:
            self.init_eps_list(delta)

        # Initialize momentum
        momentum_grad = torch.zeros_like(data)

        for t in range(self.epoch):
            x = data + delta
            grad_boundary = []

            # Sample multiple boundary points
            for i in range(self.num_boundary_points):
                # Sample random direction di ~ N(0, σ²I)
                di = torch.randn_like(x) * (self.initial_direction_std / 255.0)

                # Get original prediction
                with torch.no_grad():
                    original_output = self.get_logits(x)
                    original_pred = torch.argmax(original_output, dim=1)

                # Find boundary point by moving along di
                boundary_x = self.find_boundary_point(x, di, original_pred)

                # Apply OPS sampling at boundary point
                if self.using_sampling:
                    # Sample neighbors and operators at boundary
                    selected_eps = random.sample(self.eps_list, min(self.num_sample_neighbor, self.eps_num))

                    for eps in selected_eps:
                        x_near = boundary_x + eps

                        self.init_op_list()
                        selected_ops = random.sample(self.op_list, min(self.num_sample_operator, self.op_num))

                        for op in selected_ops:
                            x_near_op = op(x_near)
                            x_near_op.requires_grad_(True)
                            logits = self.get_logits(x_near_op)
                            loss = self.get_loss(logits, label)
                            grad = torch.autograd.grad(loss, x_near_op,
                                                      retain_graph=False, create_graph=False)[0]
                            grad_boundary.append(grad.detach())
                else:
                    # Without sampling, just compute gradient at boundary point
                    boundary_x.requires_grad_(True)
                    boundary_output = self.get_logits(boundary_x)
                    boundary_loss = self.get_loss(boundary_output, label)
                    grad = torch.autograd.grad(boundary_loss, boundary_x,
                                              retain_graph=False, create_graph=False)[0]
                    grad_boundary.append(grad.detach())

            # Average all boundary gradients
            avg_boundary_grad = torch.stack(grad_boundary).mean(dim=0)

            # Update momentum
            momentum_grad = self.decay * momentum_grad + avg_boundary_grad / torch.norm(avg_boundary_grad, p=1)

            # Update delta using sign of momentum gradient
            delta = self.update_delta(delta, data, momentum_grad, self.alpha)

        return delta.detach()


# Helper functions (same as in ops.py)
def vertical_shift(x):
    _, _, w, _ = x.shape
    step = np.random.randint(low=0, high=w, dtype=np.int32)
    return x.roll(step, dims=2)

def horizontal_shift(x):
    _, _, _, h = x.shape
    step = np.random.randint(low=0, high=h, dtype=np.int32)
    return x.roll(step, dims=3)

def vertical_flip(x):
    return x.flip(dims=(2,))

def horizontal_flip(x):
    return x.flip(dims=(3,))

class scaling():
    def __init__(self, scale) -> None:
        self.scale = scale

    def __call__(self, x):
        return x / self.scale

class dim():
    def __init__(self, resize_rate=1.1, diversity_prob=0.5) -> None:
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob

    def __call__(self, x):
        """
        Random transform the input images
        """
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        # resize the input image to random size
        rnd = torch.randint(low=min(img_size, img_resize), high=max(img_size, img_resize),
                           size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)

        # randomly add padding
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(),
                                 pad_top.item(), pad_bottom.item()], value=0)

        # resize the image back to img_size
        return F.interpolate(padded, size=[img_size, img_size], mode='bilinear', align_corners=False)

def identity(x):
    return x

class rotate():
    def __init__(self, angle) -> None:
        self.angle = angle

    def __call__(self, x):
        return TFF.rotate(img=x, angle=self.angle)
