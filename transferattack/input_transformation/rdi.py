import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ..utils import *
from ..gradient.mifgsm import MIFGSM

class RDI(MIFGSM):
    """
    RDI (Resized Diverse Input)
    'Improving Transferable Targeted Attacks with Feature Tuning Mixup (CVPR 2025)'(https://arxiv.org/abs/2411.15553)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        resize_rate (float): the resize ratio for RDI transformation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., resize_rate=340/299

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/rdi/resnet50 --attack rdi --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/rdi/resnet50 --eval
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1.,
                 resize_rate=340/299, targeted=False, random_start=False,
                 norm='linfty', loss='crossentropy', device=None, attack='RDI', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.resize_rate = resize_rate

    def transform(self, x, **kwargs):
        """
        Apply RDI (Resized Diverse Input) transformation
        """
        img_width = x.size()[-1]
        enlarged_img_width = int(img_width * self.resize_rate)
        di_pad_amount = enlarged_img_width - img_width
        ori_size = x.shape[-1]

        # Random resize to a size between original and enlarged
        rnd = int(torch.rand(1) * di_pad_amount) + ori_size
        x_transformed = transforms.Resize((rnd, rnd), interpolation=InterpolationMode.NEAREST)(x)

        # Random padding to restore original size
        pad_max = ori_size + di_pad_amount - rnd
        pad_left = int(torch.rand(1) * pad_max)
        pad_right = pad_max - pad_left
        pad_top = int(torch.rand(1) * pad_max)
        pad_bottom = pad_max - pad_top
        x_transformed = F.pad(x_transformed, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

        # Resize back to original size (skip for small images like CIFAR-10)
        if img_width > 64:
            x_transformed = transforms.Resize((ori_size, ori_size), interpolation=InterpolationMode.NEAREST)(x_transformed)

        return x_transformed
