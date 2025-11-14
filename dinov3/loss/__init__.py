# Copyright (c) [ORGANIZATION]
#
# This software may be used and distributed in accordance with
# the terms of the License Agreement.
#
# NOTE: This is a fork of DINO (DINOv3) with modifications.

from .dino_clstoken_loss import DINOLoss
from .gram_loss import GramLoss
from .ibot_patch_loss import iBOTPatchLoss
from .koleo_loss import KoLeoLoss, KoLeoLossDistributed
from .contrastive_hierarchical_loss import ContrastiveHCentroidLoss
