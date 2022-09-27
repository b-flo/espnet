"""Textogram module for Transducer model."""

import random
from typing import Dict, List, Optional, Tuple

import torch


class Textogram(torch.nn.Module):
    """Textogram module definition.

    Args:
        vocab_size: Size of the vocabulary (w/ EOS and blank included).
        mode: Whether to use "dual" modality or "text" only.
        confusion_map: List of possible replacements for the vocabulary elements.
        duration_map: Expected duration for each element of the label sequence.
        duration_variance: Variance to compute duration map, if not provided.
        confusion_rate: Confusion rate.
        masking_rate: Label masking rate.
        pad_id: Padding symbol ID (equivalent to the blank symbol ID).
        ignore_id: Initial padding symbol ID to ignore.

    """

    def __init__(
        self,
        vocab_size: int,
        mode: str,
        confusion_map: Optional[Dict[int, List[int]]] = None,
        duration_map: Optional[List[int]] = None,
        duration_variance: float = 0.5,
        confusion_rate: float = 0.0,
        masking_rate: float = 0.0,
        pad_id: int = 0,
        ignore_id: int = -1,
    ) -> None:
        """Construct a Textogram object."""
        super().__init__()

        self.vocab_size = vocab_size

        self.confusion_map = confusion_map
        self.confusion_rate = confusion_rate

        self.duration_map = duration_map
        self.duration_variance = duration_variance

        self.masking_rate = masking_rate

        self.pad = torch.LongTensor([pad_id])
        self.ignore_id = ignore_id

        self.text_only = mode == "text"
        self.compute_toogle = False if self.text_only else True

    def forward(
        self,
        feats: torch.Tensor,
        text: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute textogram features.

        Args:
            feats: Features sequences. (B, T, D_feats)
            text: Label ID sequences. (B, L)

        Returns:
            feats: Features sequences. (B, T, D_feats + D_textogram)

        """
        if self.compute_toggle:
            textogram = None
        else:
            textogram = self.compute_textograms(feats, text)

        feats = self.get_encoder_input(feats, textogram)

        if not self.text_only:
            self.compute_toggle = not self.compute_toggle

        return feats

    def compute_textograms(
        self, feats: torch.Tensor, text: torch.Tensor
    ) -> torch.Tensor:
        """Compute textogram features.

        Args:
            feats: Features sequences. (B, T, D_feats)
            text: Label ID sequences. (B, L)

        Returns:
            textogram: Textogram features. (B, T, D_vocab)

        """
        device = feats.device
        max_t = feats.size(1) - 1

        textogram = []
        text_unpad = [y[y != self.ignore_id] for y in text]

        self.pad = self.pad.to(device=device)

        for t in text_unpad:
            if self.duration_map is None:
                duration_map = self.compute_duration_map(t.size(0), max_t, device)
            else:
                raise NotImplementedError

            extended_t = torch.repeat_interleave(t, duration_map)

            if self.confusion_map is not None:
                extended_t = self.apply_confusion(extended_t)

            textogram.append(torch.cat([self.pad, extended_t]))

        textogram = torch.nn.functional.one_hot(
            torch.stack(textogram), num_classes=self.vocab_size
        )

        if self.masking_rate > 0:
            textogram = self.apply_label_masking(textogram)

        return textogram

    def get_encoder_input(
        self, feats: torch.Tensor, textogram: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get encoder input with Textogram dimensions filled with zeroes.

        Args:
            feats: Input features. (B, T, D_feats)
            textogram: Textogram features. (B, T, D_vocab)

        Returns:
            : Encoder features. (B, T, D_feats + D_vocab)

        """
        if textogram is None:
            b, t, d = feats.size()

            return torch.cat((feats, feats.new_zeros((b, t, self.vocab_size))), -1)

        return torch.cat((feats.new_zeros(feats.size()), textogram), -1)

    def apply_confusion(self, x: torch.Tensor) -> torch.Tensor:
        """Apply confusion to the input tensor.

        A portion of the labels will be permuted according to a defined mapping,
        where the portion size is defined by 'confusion_rate'.

        Note: 1) Only one-to-one mapping is allowed.
              2) Target label is selected randomly if there are multiple candidates.

        Args:
            x: Textogram sequence.

        Returns:
            x: Confused textogram sequence.

        """
        p_max = int(
            sum(c in self.confusion_map.keys() for c in x.tolist())
            * self.confusion_rate
        )
        p_indices = random.sample(
            [i for i, c in enumerate(x.tolist()) if c in self.confusion_map.keys()],
            p_max,
        )

        for p in p_indices:
            x[p] = random.choice(self.confusion_map[int(x[p])])

        return x

    def apply_label_masking(self, x: torch.Tensor) -> torch.Tensor:
        """Apply label masking to the input tensor.

        Note that the same mask is used for each item of the batch.

        Args:
            x: Textogram sequence.

        Returns:
            x: Masked textogram sequence.

        """
        m_indices = random.sample(
            [i for i in range(x.size(1))], int(self.masking_rate * x.size(1))
        )
        x.index_fill_(1, torch.tensor(m_indices).to(device=x.device), 0)

        return x

    def compute_duration_map(
        self, size: int, sum_d: int, device: torch.device
    ) -> List[int]:
        """Create a list of 'size' elements with pseudo-random values.

        1) The sum of the elements is equal to `sum_d`.
        2) Each value is defined around the mean with a certain allowed variance.

        Args:
            size: Size of the list.
            sum_p: Sum of the elements in the list.

        Returns:
            duration_map: Duration map with defined constraints.

        """
        avg = sum_d / size
        div = [
            int((x + 1) * avg + random.random() * (avg * self.duration_variance / 2))
            for x in range(size - 1)
        ]

        duration_map = torch.LongTensor(
            [a - b for a, b in zip(div + [sum_d], [0] + div)],
        ).to(device=device)

        return duration_map
