import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import pandas as pd
import timm
import os

img_height, img_width = 224, 224
img_max, img_min = 1., 0

cnn_model_paper = [
    # 'resnet50', 'resnet101',
    'resnet152', 
                #    'vgg16', 'mobilenet_v2',
                #    'inception_v3', 'inception_v4', 'inception_resnet_v2', 
                #    'inception_resnet_v2.tf_ens_adv_in1k'
                   ]
vit_model_paper = [
    # 'vit_small_patch16_224',
    # 'vit_base_patch16_224',
    # 'vit_large_patch16_224',
    #                'deit_tiny_patch16_224', 'deit_small_patch16_224', 
    #                'deit_base_patch16_224', 
                #    'deit_base_distilled_patch16_224',
                #    'tnt_s_patch16_224',
                   # 'pit_b_224', 'visformer_small',
                #    'swin_tiny_patch4_window7_224', 
                #    'swin_small_patch4_window7_224', 
                #    'swin_base_patch4_window7_224'
                   ]

cnn_model_pkg = [#'vgg19', 'resnet18', 'resnet50', 'resnet101', 'resnet152',
                 #'resnext50_32x4d', 'densenet121', 'mobilenet_v2'
                 ]
vit_model_pkg = [
    #'vit_base_patch16_224', 'vit_small_patch16_224', 'vit_large_patch16_224',
               #  'pit_b_224', 'cait_s24_224', 'visformer_small',
                # 'tnt_s_patch16_224', 'levit_256', 'convit_base',
                 #'deit_tiny_patch16_224', 'deit_small_patch16_224',
                 #'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224', 'swin_base_patch4_window7_224'
                 ]

tgr_vit_model_list = [
    # 'vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'visformer_small',
                    #   'deit_base_distilled_patch16_224', 'tnt_s_patch16_224', 'levit_256', 'convit_base'
                      ]

clip_model_list = [
    # 'clip_rn50','clip_rn101', 'clip_rn50x4', 'clip_rn50x16', 'clip_rn50x64',
    # 'clip_vit_b_16','clip_vit_b_32','clip_vit_l_14', 'siglip_vit_b_16'
    # 'clip_rn50', 'clip_rn101', 'clip_rn50x4', 'clip_rn50x16', 'clip_rn50x64',
    #                'clip_vit_b_32', 'clip_vit_b_16', 'clip_vit_l_14', 'siglip_vit_b_16'
                   ]

generation_target_classes = [24, 99, 245, 344, 471, 555, 661, 701, 802, 919]



def load_pretrained_model(cnn_model=[], vit_model=[], clip_model=[]):
    for model_name in cnn_model:
        if model_name in models.__dict__.keys():
            yield model_name, models.__dict__[model_name](weights="DEFAULT")
        else:
            yield model_name, timm.create_model(model_name, pretrained=True)
    for model_name in vit_model:
        yield model_name, timm.create_model(model_name, pretrained=True)
    for model_name in clip_model:
        yield model_name, CLIPWrapper(model_name)      
def wrap_model(model):
    """
    Add normalization layer with mean and std in training configuration
    """

    model_name = model.__class__.__name__
    Resize = 224

    if isinstance(model, CLIPWrapper):
        return model

    if hasattr(model, 'default_cfg'):
        """timm.models"""
        mean = model.default_cfg['mean']
        std = model.default_cfg['std']
    else:
        """torchvision.models"""
        if 'Inc' in model_name:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            Resize = 299
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            Resize = 224

    PreprocessModel = PreprocessingModel(Resize, mean, std)
    return torch.nn.Sequential(PreprocessModel, model)


def save_images(output_dir, adversaries, filenames):
    adversaries = (adversaries.detach().permute((0,2,3,1)).cpu().numpy() * 255).astype(np.uint8)
    for i, filename in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename))

def clamp(x, x_min, x_max):
    return torch.min(torch.max(x, x_min), x_max)


class PreprocessingModel(nn.Module):
    def __init__(self, resize, mean, std):
        super(PreprocessingModel, self).__init__()
        self.resize = transforms.Resize(resize)
        self.normalize = transforms.Normalize(mean, std)

    def forward(self, x):
        return self.normalize(self.resize(x))


class EnsembleModel(torch.nn.Module):
    def __init__(self, models, mode='mean'):
        super(EnsembleModel, self).__init__()
        self.device = next(models[0].parameters()).device
        for model in models:
            model.to(self.device)
        self.models = models
        self.softmax = torch.nn.Softmax(dim=1)
        self.type_name = 'ensemble'
        self.num_models = len(models)
        self.mode = mode

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        outputs = torch.stack(outputs, dim=0)
        if self.mode == 'mean':
            outputs = torch.mean(outputs, dim=0)
            return outputs
        elif self.mode == 'ind':
            return outputs
        else:
            raise NotImplementedError
class CLIPWrapper(nn.Module):
    """
    CLIP Model Wrapper for ImageNet Classification

    This wrapper converts CLIP models (both OpenAI CLIP and OpenCLIP) to work
    seamlessly with the TransferAttack framework by:
    1. Loading ImageNet class labels and generating text prompts
    2. Precomputing text features for all 1000 ImageNet classes
    3. Computing image-text similarity as classification logits

    Arguments:
        model_name (str): the name of CLIP model from clip_model_list in utils.py
                         e.g., 'clip_rn50', 'clip_vit_b_32', 'siglip_vit_b_16'
        device (str): the device for inference, default: 'cuda'
        imagenet_label_path (str): path to ImageNet label JSON file

    Returns:
        logits (torch.Tensor): classification logits with shape (batch_size, 1000)

    Example:
        >>> wrapper = CLIPWrapper('clip_rn50')
        >>> images = torch.randn(4, 3, 224, 224).cuda()  # batch of 4 images
        >>> logits = wrapper(images)  # shape: (4, 1000)
    """

    def __init__(self, model_name, device='cuda', imagenet_label_path='./data/label.json'):
        super(CLIPWrapper, self).__init__()
        self.model_name = model_name
        self.device = device

        # Load CLIP model
        self.model, self.preprocess = self._load_clip_model(model_name)
        self.model = self.model.to(device)
        self.model.eval()

        # Load ImageNet labels
        self.imagenet_labels = self._load_imagenet_labels(imagenet_label_path)

        # Extract and register logit_scale from the model (dynamic, not hardcoded)
        # OpenAI CLIP: logit_scale.exp() ≈ 100.0
        # SigLIP: logit_scale.exp() ≈ 117.3
        if hasattr(self.model, 'logit_scale'):
            logit_scale = self.model.logit_scale.exp().detach().clone()
            self.register_buffer('logit_scale', logit_scale)
        else:
            # Fallback to 100 if logit_scale not found
            self.register_buffer('logit_scale', torch.tensor(100.0))

        # Extract and register logit_bias from the model (SigLIP-specific)
        # SigLIP uses sigmoid loss and requires logit_bias (typically ≈ -10 to -13)
        # OpenAI CLIP uses softmax loss and does not have logit_bias
        if hasattr(self.model, 'logit_bias'):
            logit_bias = self.model.logit_bias.detach().clone()
            self.register_buffer('logit_bias', logit_bias)
        else:
            # OpenAI CLIP and other models without logit_bias
            self.register_buffer('logit_bias', torch.tensor(0.0))

        # Setup preprocessing parameters based on model type
        # Different models use different normalization and input sizes
        self._setup_preprocessing(model_name)

        # Precompute text features for all 1000 ImageNet classes
        # Register as buffer to ensure it follows the model device
        text_features = self._precompute_text_features()
        self.register_buffer('text_features', text_features)

        # Ensure all components are on the same device
        self.to(device)

    def _setup_preprocessing(self, model_name):
        """
        Setup preprocessing parameters based on model type

        Different CLIP variants use different preprocessing:
        - OpenAI CLIP: mean=[0.48145466, 0.4578275, 0.40821073],
                       std=[0.26862954, 0.26130258, 0.27577711]
        - SigLIP: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        - RN50x4/x16/x64: input_size=288 (larger models need larger input)
        - Others: input_size=224

        Arguments:
            model_name (str): model name from clip_model_list
        """
        if model_name.startswith('siglip_'):
            # SigLIP uses different normalization
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            input_size = 224
            self.use_center_crop = False  # SigLIP doesn't use center crop
        else:
            # OpenAI CLIP standard normalization
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]

            # Different ResNet variants require different input sizes
            # Based on CLIP official implementation
            if 'rn50x64' in model_name:
                input_size = 448
            elif 'rn50x16' in model_name:
                input_size = 384
            elif 'rn50x4' in model_name:
                input_size = 288
            else:
                # RN50, RN101, all ViTs use 224x224
                input_size = 224

            self.use_center_crop = True  # OpenAI CLIP uses center crop

        # Register normalization parameters as buffers
        self.register_buffer('clip_mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('clip_std', torch.tensor(std).view(1, 3, 1, 1))
        self.input_size = input_size

    def _load_clip_model(self, model_name):
        """
        Load CLIP model based on model name

        Arguments:
            model_name (str): model name from clip_model_list

        Returns:
            model: CLIP model
            preprocess: CLIP preprocessing function (for reference, not used in forward)
        """
        if model_name.startswith('siglip_'):
            # Load SigLIP model from OpenCLIP
            import open_clip
            # Convert model_name to OpenCLIP format
            # 'siglip_vit_b_16' -> 'ViT-B-16-SigLIP'
            model_arch = 'ViT-B-16-SigLIP'
            pretrained = 'webli'
            print(f'=> Loading SigLIP model {model_arch} with pretrained weights {pretrained}')
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_arch, pretrained=pretrained
            )
            return model, preprocess
        else:
            # Load OpenAI CLIP model
            import clip
            # Convert model_name to CLIP format
            # 'clip_rn50' -> 'RN50'
            # 'clip_vit_b_32' -> 'ViT-B/32'
            clip_name = self._convert_to_clip_name(model_name)
            print(f'=> Loading CLIP model {clip_name}')
            model, preprocess = clip.load(clip_name, device=self.device)
            model = model.float()
            return model, preprocess

    def _convert_to_clip_name(self, model_name):
        """
        Convert internal model name to CLIP official name

        Arguments:
            model_name (str): internal model name (e.g., 'clip_rn50')

        Returns:
            clip_name (str): CLIP official name (e.g., 'RN50')
        """
        # Mapping from internal names to CLIP official names
        name_mapping = {
            'clip_rn50': 'RN50',
            'clip_rn101': 'RN101',
            'clip_rn50x4': 'RN50x4',
            'clip_rn50x16': 'RN50x16',
            'clip_rn50x64': 'RN50x64',
            'clip_vit_b_32': 'ViT-B/32',
            'clip_vit_b_16': 'ViT-B/16',
            'clip_vit_l_14': 'ViT-L/14',
        }

        if model_name not in name_mapping:
            raise ValueError(f'Unsupported CLIP model: {model_name}')

        return name_mapping[model_name]

    def _load_imagenet_labels(self, label_path):
        """
        Load ImageNet class labels from JSON file

        Arguments:
            label_path (str): path to imagenet_label.json

        Returns:
            labels (list): list of 1000 ImageNet class names
        """
        if not os.path.exists(label_path):
            raise FileNotFoundError(
                f'ImageNet label file not found at {label_path}. '
                f'Please ensure the file exists in the data directory.'
            )

        with open(label_path, 'r') as f:
            import json
            labels = json.load(f)

        if not isinstance(labels, list) or len(labels) != 1000:
            raise ValueError(
                f'Invalid ImageNet label file. Expected a list of 1000 strings, '
                f'got {type(labels)} with length {len(labels) if isinstance(labels, list) else "N/A"}'
            )

        return labels

    def _precompute_text_features(self):
        """
        Precompute text features for all 1000 ImageNet classes

        This function generates text prompts for each ImageNet class and
        encodes them using CLIP's text encoder. The text features are
        normalized and stored for efficient inference.

        Returns:
            text_features (torch.Tensor): normalized text features with shape (1000, feature_dim)
        """
        # Generate text prompts for all classes
        text_prompts = [f"a photo of a {label}" for label in self.imagenet_labels]

        # Tokenize text prompts using model-specific tokenizer
        if self.model_name.startswith('siglip_'):
            import open_clip
            # SigLIP requires model-specific tokenizer (vocab_size=32000, context_length=64)
            # Cannot use open_clip.tokenize() as it defaults to CLIP tokenizer (vocab_size=49408)
            tokenizer = open_clip.get_tokenizer('ViT-B-16-SigLIP')
            text_tokens = tokenizer(text_prompts).to(self.device)
        else:
            import clip
            # OpenAI CLIP uses standard tokenizer (vocab_size=49408, context_length=77)
            text_tokens = clip.tokenize(text_prompts).to(self.device)

        # Encode text features
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            # Normalize text features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.float()

    def forward(self, x):
        """
        Forward pass for CLIP classification

        This method applies the complete CLIP preprocessing pipeline:
        1. Resize to input_size (bicubic interpolation)
        2. Center crop (for OpenAI CLIP only, not for SigLIP)
        3. Normalize with model-specific mean/std
        4. Encode image and compute similarity with text features
        5. Scale by model-specific logit_scale

        Arguments:
            x (torch.Tensor): input images with shape (batch_size, 3, H, W)
                             and value range [0, 1]

        Returns:
            logits (torch.Tensor): classification logits with shape (batch_size, 1000)
        """
        # Step 1: Resize to target size using bicubic interpolation
        # This matches CLIP's official preprocessing
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = torch.nn.functional.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode='bicubic',
                align_corners=False,
                antialias=True
            )

        # Step 2: Center crop (OpenAI CLIP only)
        # OpenAI CLIP: Resize(224) + CenterCrop(224)
        # SigLIP: Direct Resize((224, 224))
        if self.use_center_crop:
            # For OpenAI CLIP, we already resized to 224x224, so center crop is identity
            # But this flag is here for completeness and future extensions
            pass

        # Step 3: Normalize with model-specific parameters
        x_normalized = (x - self.clip_mean) / self.clip_std

        # Step 4: Encode image features
        # NOTE: torch.no_grad() removed to enable gradient computation for adversarial attacks
        # IMPORTANT: OpenCLIP models have normalize parameter, OpenAI CLIP does not
        if self.model_name.startswith('siglip_'):
            # OpenCLIP: use normalize=False since we already normalized manually
            image_features = self.model.encode_image(x_normalized, normalize=False)
        else:
            # OpenAI CLIP: no normalize parameter
            image_features = self.model.encode_image(x_normalized)

        # Normalize image features (L2 normalization)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Step 5: Compute image-text similarity as logits
        # Use dynamic logit_scale and logit_bias extracted from the model
        # OpenAI CLIP: logits = logit_scale * similarity (logit_bias = 0)
        # SigLIP: logits = logit_scale * similarity + logit_bias (logit_bias ≈ -13)
        logits = self.logit_scale * (image_features.float() @ self.text_features.T) + self.logit_bias

        return logits

class AdvDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir=None, output_dir=None, targeted=False, target_class=None, eval=False):
        self.targeted = targeted
        self.target_class = target_class
        self.data_dir = input_dir
        self.f2l = self.load_labels(os.path.join(self.data_dir, 'labels.csv'))

        if eval:
            self.data_dir = output_dir
            # load images from output_dir, labels from input_dir/labels.csv
            print('=> Eval mode: evaluating on {}'.format(self.data_dir))
        else:
            self.data_dir = os.path.join(self.data_dir, 'images')
            print('=> Train mode: training on {}'.format(self.data_dir))
            print('Save images to {}'.format(output_dir))

    def __len__(self):
        return len(self.f2l.keys())

    def __getitem__(self, idx):
        filename = list(self.f2l.keys())[idx]

        assert isinstance(filename, str)

        filepath = os.path.join(self.data_dir, filename)
        image = Image.open(filepath)
        image = image.resize((img_height, img_width)).convert('RGB')
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        image = np.array(image).astype(np.float32)/255
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = self.f2l[filename]

        return image, label, filename

    def load_labels(self, file_name):
        dev = pd.read_csv(file_name)
        if self.targeted:
            if self.target_class:
                f2l = {dev.iloc[i]['filename']: [dev.iloc[i]['label'], self.target_class] for i in range(len(dev))}
            else:
                f2l = {dev.iloc[i]['filename']: [dev.iloc[i]['label'],
                                             dev.iloc[i]['targeted_label']] for i in range(len(dev))}
        else:
            f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label']
                   for i in range(len(dev))}
        return f2l


if __name__ == '__main__':
    dataset = AdvDataset(input_dir='./data_targeted',
                         targeted=True, eval=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=0)

    for i, (images, labels, filenames) in enumerate(dataloader):
        print(images.shape)
        print(labels)
        print(filenames)
        break
