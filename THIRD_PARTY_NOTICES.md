# Third-party notices

The source distribution declares Apache License 2.0.

The convolution/residual and flow-warp primitives in `basicsr/archs/arch_util.py`,
and the SpyNet implementation in `basicsr/archs/spynet_arch.py`, are adapted from
[BasicSR](https://github.com/XPixelGroup/BasicSR).
Copyright 2018–2022 BasicSR Authors. These adapted files have been simplified
and reorganized; their source license is
[Apache-2.0](https://github.com/XPixelGroup/BasicSR/blob/master/LICENSE.txt).

The remaining model modules and loss formulas are retained from the existing
additive/preview source distribution, with experiment-specific
instrumentation removed and portable entry points added.

[MMCV](https://github.com/open-mmlab/mmcv) provides the modulated deformable
convolution operator as an external dependency.
[PyIQA](https://github.com/chaofengc/IQA-PyTorch) provides PSNR/SSIM evaluation
as an optional external dependency. Their source is not vendored here.

Datasets and pretrained weights must be obtained separately from their owners;
none are redistributed in this source package.

