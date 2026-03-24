# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" LLaMA model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class PtConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PtModel`]. It is used to instantiate an Pt
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Llama-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Pt model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`PtModel`]
        dim_z (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        dim_g (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_iterations (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_channels (`int`, *optional*, defaults to 32):
            Number of channels for head-selection. Counterparts of attention heads for each attention layer in the
            Transformer decoder.
        potential_func_z (`str` or `function`, *optional*, defaults to `"square"`):
            The potential function for Z nodes. Counterparts of non-linear activation function in Transformer decoder.
            Options:
            - `"exp"`: Counterpart of softmax function.
            - `"abs"`: Absolute value of the input.
            - `"square"`: Counterpart of squared softmax function.
        potential_func_g (`str` or `function`, *optional*, defaults to `"abs"`):
            The potential function for G nodes. Counterparts of non-linear activation function in Transformer decoder.
            Options:
            - `"exp"`: Counterpart of softmax function.
            - `"abs"`: Absolute value of the input.
            - `"square"`: Counterpart of squared softmax function.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
            Llama 2 up to 4096, CodeLlama up to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        binary_initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all binary factor weight matrices.
        ternary_initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all ternary factor weight matrices.
        binary_factor_scaling (`float`, *optional*, defaults to 1.0):
            The scaling factor for the binary factor weight matrices. This is meant to keep the binary factor weights
            small, so that it is easier to optimize. Usually set to 1.0.
        ternary_factor_scaling (`float`, *optional*, defaults to 1.0):
            The scaling factor for the ternary factor weight matrices. This is meant to keep the ternary factor weights
            small, so that it is easier to optimize. Usually set to 1.0.
        classifier_amplifier (`float`, *optional*, defaults to 1.0):
            The scaling factor for the classifier head. This is meant to keep the classifier weights small, so that it
            is easier to optimize. Usually set to dim_z.
        potential_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the potential function. Counterpart of eps of rms normalization layers.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        dropout_prob_z (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        dropout_prob_h (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        regularize_z (`float`, *optional*, defaults to 1):
            The regularization strength for Z nodes. Usually set to 1.
        regularize_h (`float`, *optional*, defaults to 1):
            The regularization strength for H nodes. Usually set to 1/dim_z**2.
        regularize_g (`float`, *optional*, defaults to 1):
            The regularization strength for G nodes. Usually set to 1/dim_z.


    ```python
    >>> from transformers import LlamaModel, LlamaConfig

    >>> # Initializing a LLaMA llama-7b style configuration
    >>> configuration = LlamaConfig()

    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "pt"

    def __init__(
        self,
        vocab_size=32000,
        dim_z=4096,
        dim_g=11008,
        num_iterations=32,
        num_channels=32,
        ternary_rank=None,
        potential_func_z="square",
        potential_func_g="abs",
        max_position_embeddings=2048,
        initializer_range=0.02,
        binary_initializer_range=0.02,
        ternary_initializer_range=0.02,
        binary_factor_scaling=1.0,
        ternary_factor_scaling=1.0,
        classifier_amplifier=1.0,
        potential_eps=1e-6,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        dropout_prob_z=0.1,
        dropout_prob_h=0.1,
        classifier_dropout=None,
        regularize_z=1.0,
        regularize_h=1.0,
        regularize_g=1.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.dim_z = dim_z
        self.dim_g = dim_g

        self.num_iterations = num_iterations
        self.num_channels = num_channels

        if ternary_rank is None:
            ternary_rank = dim_z // num_channels

        self.ternary_rank = ternary_rank
        self.potential_func_z = potential_func_z
        self.potential_func_g = potential_func_g
        self.initializer_range = initializer_range
        self.binary_initializer_range = binary_initializer_range
        self.ternary_initializer_range = ternary_initializer_range
        self.binary_factor_scaling = binary_factor_scaling
        self.ternary_factor_scaling = ternary_factor_scaling
        self.classifier_amplifier = classifier_amplifier
        self.potential_eps = potential_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()

        self.dropout_prob_z = dropout_prob_z
        self.dropout_prob_h = dropout_prob_h
        self.classifier_dropout = classifier_dropout
        self.regularize_z = regularize_z
        self.regularize_h = regularize_h
        self.regularize_g = regularize_g

        # prediction head config
        self.hidden_size = dim_z
        self.hidden_act = kwargs.pop("hidden_act", "gelu")
        self.layer_norm_eps = kwargs.pop("layer_norm_eps", 1e-6)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")
