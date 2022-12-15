# The modification is based on detectron2. 
# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    DeformConv,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

from .backbone import Backbone
from .build import BACKBONE_REGISTRY

__all__ = [
    "VGGBlockBase",
    "Stem",
    "Res2",
    "Res3",
    "Res4",
    "Res5",
    "VGG",
    "make_stage",
    "build_vgg_backbone",
]

class Stem(CNNBlockBase):
    """
    The standard stem (layers before the first residual block),
    with conv, relu and max_pool.
    """

    def __init__(self, in_channels=3, out_channels=128, norm="BN"):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, 2)
        self.in_channels = in_channels
        self.conv1 = Conv2d(
            in_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, 64),
        )

        self.conv2 = Conv2d(
            64,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, 64),
        )


        self.conv3 = Conv2d(
            64,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv4 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4]:
            weight_init.c2_msra_fill(layer)

    def forward(self, x):
        x = self.conv1(x) # same input/output size
        x = F.relu_(x) 
        x = self.conv2(x) # same input/output size
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1) # downsampling 

        out = self.conv3(x) # same input/output size
        out = F.relu_(out) 
        out = self.conv4(out) # same input/output size
        out = F.relu_(out)
        #out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1) # downsampling 
        #print(out.size())
        return out


class Res2(CNNBlockBase):
    """
    The first stack of layers that will be fed into FPN,
    with conv, relu and max_pool.
    """

    def __init__(self, in_channels=128, out_channels=256, norm="BN"):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, 2)
        self.in_channels = in_channels
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv3 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv4 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4]:
            weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        out = self.conv1(out) # same input/output size
        out = F.relu_(out) 
        out = self.conv2(out) # same input/output size
        out = F.relu_(out)
        out = self.conv3(out) # same input/output size
        out = F.relu_(out) 
        out = self.conv4(out) # same input/output size
        out = F.relu_(out)
        #out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1) # downsampling 
        #print(out.size())
        return out

class Res3(CNNBlockBase):
    """
    The second stack of layers
    """

    def __init__(self, in_channels=256, out_channels=512, norm="BN"):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, 2)
        self.in_channels = in_channels
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv3 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv4 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4]:
            weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = F.max_pool2d(x, kernel_size=3, stride=2, padding=1) # downsampling first 
        out = self.conv1(out) # same input/output size
        out = F.relu_(out) 
        out = self.conv2(out) # same input/output size
        out = F.relu_(out)
        out = self.conv3(out) # same input/output size
        out = F.relu_(out) 
        out = self.conv4(out) # same input/output size
        out = F.relu_(out)
        #out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1) # downsampling 
        #print(out.size())
        return out

class Res4(CNNBlockBase):
    """
    The third stack of layers
    """

    def __init__(self, in_channels=512, out_channels=512, norm="BN"):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, 2)
        self.in_channels = in_channels
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv3 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv4 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4]:
            weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        out = self.conv1(out) # same input/output size
        out = F.relu_(out) 
        out = self.conv2(out) # same input/output size
        out = F.relu_(out)
        out = self.conv3(out) # same input/output size
        out = F.relu_(out) 
        out = self.conv4(out) # same input/output size
        out = F.relu_(out)
        #print(out.size())
        #out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1) # downsampling 
        return out

class Res5(CNNBlockBase):
    """
    The final stack of layers
    """

    def __init__(self, in_channels=512, out_channels=512, norm="BN"):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, 2)
        self.in_channels = in_channels
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv3 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv4 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4]:
            weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x) # same input/output size
        out = F.relu_(out) 
        out = self.conv2(out) # same input/output size
        out = F.relu_(out)
        out = self.conv3(out) # same input/output size
        out = F.relu_(out) 
        out = self.conv4(out) # same input/output size
        out = F.relu_(out)
        out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1) # there needs one as well. 
        #print(out.size())
        return out

class VGG(Backbone):
    """
    Implement :paper:`ResNet (VGG part with the fourth stack being added)`. 
    The implementation is modified based on the original Resnet backbone class 
    """

    def __init__(self, stem, stages, num_classes=None, out_features=None, freeze_at=0):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[CNNBlockBase]]): several (typically 4) stages,
                each contains multiple :class:`CNNBlockBase`.
            num_classes (None or int): if None, will not perform classification.
                Otherwise, will create a linear layer.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
            freeze_at (int): The number of stages at the beginning to freeze.
                see :meth:`freeze` for detailed explanation.
        """
        super().__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stage_names, self.stages = [], []

        if out_features is not None:
            # Avoid keeping unused layers in this module. They consume extra memory
            # and may cause allreduce to fail
            num_stages = max(
                [{"res2": 1, "res3": 2, "res4": 3, "res5": 4}.get(f, 0) for f in out_features]
            )
            stages = stages[:num_stages]
        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)
            for block in blocks:
                assert isinstance(block, CNNBlockBase), block

            name = "res" + str(i + 2)
            stage = nn.Sequential(*blocks)

            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)

            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = curr_channels = blocks[-1].out_channels
        self.stage_names = tuple(self.stage_names)  # Make it static for scripting

        if num_classes is not None:
            self.linear1 = nn.Linear(curr_channels, 4096)
            self.linear2 = nn.Linear(4096, 4096)
            self.linear3 = nn.Linear(4096, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear1.weight, std=0.01)
            nn.init.normal_(self.linear2.weight, std=0.01)
            nn.init.normal_(self.linear3.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))
        self.freeze(freeze_at)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        x = self.stem(x)

        if "stem" in self._out_features:
            outputs["stem"] = x
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            #x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this ResNet itself
        """
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self

    @staticmethod
    def make_stage(block_class):
        """
        Create a list of blocks of the same type that forms one ResNet stage.

        Args:
            block_class (type): a subclass of CNNBlockBase that's used to create all blocks in this
                stage. A module of this type must not change spatial resolution of inputs unless its
                stride != 1.
            num_blocks (int): number of blocks in this stage
            in_channels (int): input channels of the entire stage.
            out_channels (int): output channels of **every block** in the stage.
            kwargs: other arguments passed to the constructor of
                `block_class`. If the argument name is "xx_per_block", the
                argument is a list of values to be passed to each block in the
                stage. Otherwise, the same argument is passed to every block
                in the stage.

        Returns:
            list[CNNBlockBase]: a list of block module.

        Examples:
        ::
            stage = ResNet.make_stage(
                BottleneckBlock, 3, in_channels=16, out_channels=64,
                bottleneck_channels=16, num_groups=1,
                stride_per_block=[2, 1, 1],
                dilations_per_block=[1, 1, 2]
            )

        Usually, layers that produce the same feature map spatial size are defined as one
        "stage" (in :paper:`FPN`). Under such definition, ``stride_per_block[1:]`` should
        all be 1.
        """
        blocks = []
        blocks.append(block_class())
        return blocks

    @staticmethod
    def make_default_stages():
        """
        Created list of ResNet stages from pre-defined depth (one of 18, 34, 50, 101, 152).
        If it doesn't create the ResNet variant you need, please use :meth:`make_stage`
        instead for fine-grained customization.

        Args:
            depth (int): depth of ResNet
            block_class (type): the CNN block class. Has to accept
                `bottleneck_channels` argument for depth > 50.
                By default it is BasicBlock or BottleneckBlock, based on the
                depth.
            kwargs:
                other arguments to pass to `make_stage`. Should not contain
                stride and channels, as they are predefined for each depth.

        Returns:
            list[list[CNNBlockBase]]: modules in all stages; see arguments of
                :class:`ResNet.__init__`.
        """
        #num_blocks_per_stage = {
            #23: [1, 1, 1, 1],
       # }
        
        ret = []

        for block in [Res2, Res3, Res4, Res5]:
            ret.append(
                VGG.make_stage(
                    block_class=block
                )
            )
        return ret


VGGBlockBase = CNNBlockBase
"""
Alias for backward compatibiltiy.
"""


def make_stage(*args, **kwargs):
    """
    Deprecated alias for backward compatibiltiy.
    """
    return VGG.make_stage(*args, **kwargs)


@BACKBONE_REGISTRY.register()
def build_vgg_backbone(cfg, input_shape):
    """
    Create a VGG instance from config.

    Returns:
        VGG: a :class:`VGG` instance.
    """
    # need registration of new blocks/stems?
    #norm = cfg.MODEL.RESNETS.NORM
    stem = Stem(
        in_channels=input_shape.channels,
        out_channels=128,
    )

    # fmt: off
    #freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = ["res2", "res3", "res4", "res5"]
    #depth               = cfg.MODEL.RESNETS.DEPTH
    #num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    #width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    #bottleneck_channels = num_groups * width_per_group
    #in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    #out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    #stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    #res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    #deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    #deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    #deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    # fmt: on
    #assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    #if depth in [18, 34]:
        #assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        #assert not any(
            #deform_on_per_stage
        #), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"
        #assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
        #assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

    stages = VGG.make_default_stages()

    return VGG(stem, stages, out_features=out_features)
