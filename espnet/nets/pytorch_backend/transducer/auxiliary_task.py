"""Auxiliary task implementation for transducer models."""

from typing import List
from typing import Tuple

import torch
import torch.nn.functional as F


class AuxiliaryTask(torch.nn.Module):
    """Auxiliary task module."""

    def __init__(
        self,
        joint_network: torch.nn.Module,
        rnnt_criterion: torch.nn.Module,
        aux_task_type: str,
        aux_trans_weight: float,
        aux_js_div_weight: float,
        encoder_out_dim: int,
        joint_dim: int,
    ):
        """Auxiliary task initialization.

        Args:
            joint_network: Joint network module
            aux_task_type: Auxiliary task type
            aux_task_weight: Auxiliary task weight
            encoder_out: Encoder output dimension
            joint_dim: Joint space dimension

        """
        super().__init__()

        self.rnnt_criterion = rnnt_criterion

        self.mlp_net = torch.nn.Sequential(
            torch.nn.Linear(encoder_out_dim, joint_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(joint_dim, joint_dim),
        )

        self.joint_network = joint_network

        if aux_task_type in ["js_div", "both"]:
            self.kl_div = torch.nn.KLDivLoss(reduction="mean")

        self.aux_task_type = aux_task_type
        self.aux_trans_weight = aux_trans_weight
        self.aux_js_div_weight = aux_js_div_weight

    def forward(
        self,
        enc_out_aux: List,
        dec_out: torch.Tensor,
        main_joint: torch.Tensor,
        target: torch.Tensor,
        pred_len: torch.Tensor,
        target_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward auxiliary task.

        Args:
            enc_out_aux: List of encoder intermediate outputs
            dec_out: Decoder outputs
            main_joint: Joint output for main task
            target: Target labels
            pred_len: Prediction lengths
            target_len: Target lengths

        Returns:
            : (Weighted auxiliary transducer loss, weighted auxiliary symmetric KL loss)

        """
        aux_trans = 0.0
        aux_js_div = 0.0

        for i, enc_aux in enumerate(enc_out_aux):
            aux_mlp = self.mlp_net(enc_aux)

            aux_joint = self.joint_network(
                aux_mlp.unsqueeze(2),
                dec_out.unsqueeze(1),
                is_aux=True,
            )

            if self.aux_task_type in ["default", "both"]:
                aux_trans += self.rnnt_criterion(
                    aux_joint,
                    target,
                    pred_len,
                    target_len,
                )

            if self.aux_task_type in ["js_div", "both"]:
                M = 0.5 * (main_joint + aux_joint)

                aux_js_div += 0.5 * (
                    self.kl_div(
                        F.log_softmax(main_joint, dim=-1),
                        F.softmax(M, dim=-1),
                    )
                    + self.kl_div(
                        F.log_softmax(aux_joint, dim=-1),
                        F.softmax(M, dim=-1),
                    )
                )

        return (
            (self.aux_trans_weight * aux_trans),
            (self.aux_js_div_weight * aux_js_div),
        )
