# Copyright (c) [ORGANIZATION]
#
# This software may be used and distributed in accordance with
# the terms of the License Agreement.
#
# NOTE: This is a fork of DINO (DINOv3) with modifications.

from .multidist_meta_arch import MultiDistillationMetaArch
from .ssl_meta_arch import SSLMetaArch
from .train import get_args_parser, main
