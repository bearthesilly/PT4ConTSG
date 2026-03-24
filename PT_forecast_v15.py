'''
This model is called 'PT_hybrid', which is the backbone of ST-PT framework
PT_channel has rather satisfying performance, but is lack of novelty
Inspired by Crossformer in Time Series, we design 'PT_hybrid'
For more information about this model, please refer to the model architecture illustrated
in the epan folder

There will be a lot of stuff to tune:
dim_z, dim_g, number of head, the design of unary potential and decoder
number of iteration, learning rate, binary_factor_scaling, ternary_factor_scaling

This code might be unfriendly w.r.t. memory management, and may have . 
If you have any question or suggestion, please drop me a message any time! 

'''
import math
import warnings
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Union
from layers.Embed import PTDataEmbedding
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from .configuration_pt import PtConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding,rotate_half
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, MaskedLMOutput
from transformers.utils import (
    logging,
    ModelOutput
)
# from transformers.modeling_utils import PreTrainedModel

from transformers.modeling_attn_mask_utils import AttentionMaskConverter


logger = logging.get_logger(__name__)
from typing import List, Optional

@dataclass
class Config:
    potential_func_z: str = "square"
    potential_func_g: str = "abs"
    binary_initializer_range: float = 0.2
    ternary_initializer_range: float = 0.2
    binary_factor_scaling: float = 1.0
    ternary_factor_scaling: float = 1.0
    potential_eps: float = 1e-6
    rope_theta: float = 10000.0
    dropout_prob_z: float = 0.0
    dropout_prob_h: float = 0.0
    regularize_z: float = 1
    regularize_g: float = 1.0

    def to_dict(self):
        return asdict(self)

config = Config()

class RopeApplier:
    def __init__(self, cos, sin, position_ids=None, unsqueeze_dim=1) -> None:
        """Applies Rotary Position Embedding to the query, key and value tensors.

        Args:
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        """
        self.cos = cos.unsqueeze(unsqueeze_dim)
        self.sin = sin.unsqueeze(unsqueeze_dim)

    def apply(self, qkv):
        return (qkv * self.cos) + (rotate_half(qkv) * self.sin)
    
    def apply_o(self, o):
        return (o * self.cos) - (rotate_half(o) * self.sin)


class SquaredSoftmax(nn.Module):
    def __init__(self, dim=-1, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
    
    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = hidden_states.pow(2)
        hidden_states = F.normalize(hidden_states, p=1, dim=self.dim, eps=self.eps)
        return hidden_states.to(input_dtype)


class AbsNormalization(nn.Module):
    def __init__(self, dim=-1, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
    
    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = F.relu(hidden_states)
        hidden_states = F.normalize(hidden_states, p=1, dim=self.dim, eps=self.eps)
        return hidden_states.to(input_dtype)


class Softmax(nn.Softmax):
    # This is a workaround to allow passing the eps
    def __init__(self, dim=-1, eps=None):
        super().__init__(dim=dim)


POTENTIAL2ACT = {
    "exp": Softmax,
    "abs": AbsNormalization,
    "square": SquaredSoftmax,
}


class PtHeadSelection(nn.Module):
    """Multi-channel head selection from 'Probabilistic Transformer' paper"""
    
    def __init__(self, args):
        super().__init__()
        self.config = config
        self.dim_z = args.d_model
        self.enc_in = args.enc_in
        # IMPORTANT: This is not the number of channels of the time series, but the number of channel in Head Dependency
        self.num_channels = args.n_heads
        self.ternary_rank = self.dim_z // self.num_channels
        self.rope_theta = config.rope_theta
        # This is settled! No need to tune.
        self.regularize_h = 1/self.dim_z
        '''
        Two sets of ternary UV factors, one for channel and one for time
        '''
        self.ternary_factor_u_time = nn.Parameter(torch.empty(self.num_channels * self.ternary_rank, self.dim_z))
        self.ternary_factor_v_time = nn.Parameter(torch.empty(self.num_channels * self.ternary_rank, self.dim_z))
        
        self.ternary_factor_u_channel = nn.Parameter(torch.empty(self.num_channels * self.ternary_rank, self.dim_z))
        self.ternary_factor_v_channel = nn.Parameter(torch.empty(self.num_channels * self.ternary_rank, self.dim_z))
        self.dropout = nn.Dropout(config.dropout_prob_h)
        self._init_ternary()
    
    def _init_ternary(self):
        nn.init.normal_(self.ternary_factor_u_channel, mean=0.0, std=self.config.ternary_initializer_range)
        nn.init.normal_(self.ternary_factor_v_channel, mean=0.0, std=self.config.ternary_initializer_range)
        nn.init.normal_(self.ternary_factor_u_time, mean=0.0, std=self.config.ternary_initializer_range)
        nn.init.normal_(self.ternary_factor_v_time, mean=0.0, std=self.config.ternary_initializer_range)

    def calculate_messageF(self, qz, dependency_mask, position_ids, position_embeddings, ternary_factor_u, ternary_factor_v):
        # Here the seq_len is actually the number of patch
        bsz, seq_len, _ = qz.size() # this seq_len actually is enc_in, the number of channels of the time series
        qz_u = nn.functional.linear(qz, ternary_factor_u) * self.config.ternary_factor_scaling
        qz_v = nn.functional.linear(qz, ternary_factor_v) * self.config.ternary_factor_scaling
        qz_u = qz_u.view(bsz, seq_len, self.num_channels, self.ternary_rank).transpose(1, 2)
        qz_v = qz_v.view(bsz, seq_len, self.num_channels, self.ternary_rank).transpose(1, 2)
        cos, sin = position_embeddings
        rope_applier = RopeApplier(cos, sin, position_ids)
        qz_uo = rope_applier.apply_o(qz_u)
        qz_u = rope_applier.apply(qz_u)
        qz_v = rope_applier.apply(qz_v)
        message_F = torch.matmul(qz_u, qz_v.transpose(2, 3))
        if message_F.size() != (bsz, self.num_channels, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_channels, seq_len, seq_len)}, but is"
                f" {message_F.size()}"
            )

        if dependency_mask is not None:
            if dependency_mask.size() != (bsz, 1, seq_len, seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, seq_len, seq_len)}, but is {dependency_mask.size()}"
                )
            message_F = message_F + dependency_mask # need mask diag
        return message_F, qz_u, qz_v, qz_uo, bsz, seq_len, qz_u.dtype

    def calculate_messageG(self, qh, qz_uo, qz_v, bsz, seq_len, position_ids, position_embeddings, ternary_factor_u, ternary_factor_v):
        # torch.cuda.empty_cache()  
        cos, sin = position_embeddings
        rope_applier = RopeApplier(cos, sin, position_ids)
        qh_v1 = torch.matmul(qh, qz_v)
        # raise RuntimeError(qh.shape, qz_uo.shape) -> [4, 12, 1024, 1024], [4, 12, 1024, 64]
        qh_v2 = torch.matmul(qh.transpose(2, 3), qz_uo)
        # apply rotary position embedding to the output
        qh_v1 = rope_applier.apply_o(qh_v1)
        qh_v2 = rope_applier.apply(qh_v2)
        if qh_v1.size() != (bsz, self.num_channels, seq_len, self.ternary_rank):
            raise ValueError(
                f"`qh_v1` should be of size {(bsz, self.num_channels, seq_len, self.ternary_rank)}, but is"
                f" {qh_v1.size()}"
            )
        if qh_v2.size() != (bsz, self.num_channels, seq_len, self.ternary_rank):
            raise ValueError(
                f"`qh_v2` should be of size {(bsz, self.num_channels, seq_len, self.ternary_rank)}, but is"
                f" {qh_v2.size()}"
            )
        qh_v1 = qh_v1.transpose(1, 2).contiguous()
        qh_v2 = qh_v2.transpose(1, 2).contiguous()
        qh_v1 = qh_v1.reshape(bsz, seq_len, self.num_channels * self.ternary_rank)
        qh_v2 = qh_v2.reshape(bsz, seq_len, self.num_channels * self.ternary_rank)
        message_G = (torch.matmul(qh_v1, ternary_factor_u) + torch.matmul(qh_v2, ternary_factor_v)) * self.config.ternary_factor_scaling
        return message_G
    
    def forward(
        self,
        qz: torch.Tensor,
        dependency_mask_channel: Optional[torch.Tensor] = None,
        dependency_mask_time: Optional[torch.Tensor] = None,
        position_ids_time: Optional[torch.LongTensor] = None,
        position_ids_channel: Optional[torch.LongTensor] = None,
        output_dependencies: bool = False,
        position_embeddings_time: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_embeddings_channel: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # qz [bs, enc_in, length, dim_z]
        bs, num_channel, length, _ = qz.size()
        qz = qz.view(bs*num_channel, length, -1)
        message_F_time, qz_u_time, qz_v_time, qz_uo_time, bsz_time, seq_len_time, type_time= self.calculate_messageF(
            qz,
            dependency_mask=dependency_mask_time,
            position_ids=position_ids_time,
            position_embeddings=position_embeddings_time,
            ternary_factor_u=self.ternary_factor_u_time,
            ternary_factor_v=self.ternary_factor_v_time
        )
        qz = qz.view(bs, num_channel, length, -1)
        qz = qz.transpose(1,2).reshape(bs*length, num_channel, -1)
        # raise RuntimeError(dependency_mask_channel.shape, dependency_mask_time.shape)
        message_F_channel, qz_u_channel, qz_v_channel, qz_uo_channel, bsz_channel, seq_len_channel, type_channel = self.calculate_messageF(
            qz,
            dependency_mask=dependency_mask_channel,
            position_ids=position_ids_channel,
            position_embeddings=position_embeddings_channel,
            ternary_factor_u=self.ternary_factor_u_channel,
            ternary_factor_v=self.ternary_factor_v_channel
        )
        qz =qz.view(bs,length, num_channel,-1).transpose(1,2)
        # raise RuntimeError(message_F_time.shape, message_F_channel.shape)
        # Given two message F, combine them together (MAKE SURE THE DATA OF ONE Z VARIABLE IS CORRECTLY ALIGNED AND CONCATENATED)
        mF_time_reshaped = message_F_time.view(bs, num_channel, self.num_channels, length, length).permute(0, 2, 1, 3, 4)
        mF_channel_reshaped = message_F_channel.view(bs, length, self.num_channels, num_channel, num_channel).permute(0, 2, 3, 1, 4)
        combined_qh_logits = torch.cat([mF_time_reshaped, mF_channel_reshaped], dim=-1)
        # Here we only have on H node for a Z variable (which stands for the dependency over the variables who share either the same channel or time step)
        # which means the normalization is operated along both time and channel dimension
        # And these two dimension's dependency can have competition. 
        combined_qh = nn.functional.softmax(combined_qh_logits / self.regularize_h, dim=-1, dtype=torch.float32)
        # Split the result back into time and channel components
        qh_time_combined, qh_channel_combined = torch.split(combined_qh, [length, num_channel], dim=-1)
        # Reshape back to the original format expected by calculate_messageG
        # [bs, num_heads, num_channel, length, length] -> [bs, num_channel, num_heads, length, length] -> [bs*num_channel, num_heads, length, length]
        qh_time_output = qh_time_combined.permute(0, 2, 1, 3, 4)
        qh_time = qh_time_output.reshape(bs * num_channel, self.num_channels, length, length).to(type_time)
        # [bs, num_heads, num_channel, length, num_channel] -> [bs, length, num_heads, num_channel, num_channel] -> [bs*length, num_heads, num_channel, num_channel]
        qh_channel_output = qh_channel_combined.permute(0, 3, 1, 2, 4)  # [bs, length, heads, num_channel, num_channel]
        qh_channel = qh_channel_output.reshape(bs * length, self.num_channels, num_channel, num_channel).to(type_channel)
    
        message_G_channel = self.calculate_messageG(
            qh=qh_channel,
            qz_uo=qz_uo_channel,
            qz_v=qz_v_channel,
            bsz=bsz_channel,
            seq_len=seq_len_channel,
            position_ids=position_ids_channel,
            position_embeddings=position_embeddings_channel,
            ternary_factor_u=self.ternary_factor_u_channel,
            ternary_factor_v=self.ternary_factor_v_channel
        ).reshape(bs, num_channel, length, -1)
        message_G_time = self.calculate_messageG(
            qh=qh_time,
            qz_uo=qz_uo_time,
            qz_v=qz_v_time,
            bsz=bsz_time,
            seq_len=seq_len_time,
            position_ids=position_ids_time,
            position_embeddings=position_embeddings_time,
            ternary_factor_u=self.ternary_factor_u_time,
            ternary_factor_v=self.ternary_factor_v_time
        ).reshape(bs, num_channel, length, -1)
        return message_G_time, message_G_channel, qh_time_output, qh_channel_output


class PtTopicModeling(nn.Module):
    """Topic modeling w/ global nodes."""
    def __init__(self, args):
        super().__init__()
        self.config = config
        self.dim_z = args.d_model
        self.dim_g = args.d_ff
        self.binary_factor = nn.Parameter(torch.empty(self.dim_g, self.dim_z))
        self.act = POTENTIAL2ACT[config.potential_func_g](dim=-1, eps=config.potential_eps)
        self._init_binary()
        
    def _init_binary(self):
        nn.init.normal_(self.binary_factor, mean=0.0, std=self.config.binary_initializer_range)

    def forward(self, qz: torch.Tensor):
        # qz [bs, enc_in, length, dim_z]
        qg = nn.functional.linear(qz, self.binary_factor) * self.config.binary_factor_scaling
        qg = self.act(qg / self.config.regularize_g)
        message_G = qg @ self.binary_factor * self.config.binary_factor_scaling
        return message_G

class PtEncoderIterator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.config = config
        self.dim_z = args.d_model
        self.head_selection = PtHeadSelection(args)
        self.topic_modeling = PtTopicModeling(args)
        self.norm = POTENTIAL2ACT[config.potential_func_z](dim=-1, eps=config.potential_eps)
    
    def forward(
        self,
        unary_potentials: torch.Tensor,
        qz: torch.Tensor,
        dependency_mask_channel: Optional[torch.Tensor] = None,
        dependency_mask_time: Optional[torch.Tensor] = None,
        position_ids_time: Optional[torch.LongTensor] = None,
        position_ids_channel: Optional[torch.LongTensor] = None,
        output_dependencies: Optional[bool] = False,
        position_embeddings_time: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_embeddings_channel: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """z
        Args:
            qz (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            dependency_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_dependencies (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        old_qz = qz
        qz = self.norm(qz)
        # head selection
        m_t, m_c, qh_t, qh_c = self.head_selection(
            qz=qz,
            dependency_mask_channel=dependency_mask_channel,
            dependency_mask_time = dependency_mask_time,
            position_ids_time=position_ids_time,
            position_ids_channel=position_ids_channel,
            output_dependencies=output_dependencies,
            position_embeddings_time=position_embeddings_time,
            position_embeddings_channel=position_embeddings_channel,
        )

        # topic modeling
        m_g = self.topic_modeling(qz)
        # unary potentials
        # m_t: message along the dimension of time stamp
        # m_c: message along the dimension of channel
        # m_g: message from global topic modeling
        qz = (m_t + m_c + m_g + unary_potentials) / self.config.regularize_z
        # damping
        qz = (qz + old_qz) * .5

        return qz
    


class PtModel(nn.Module):
    def __init__(self, args):
        super(PtModel, self).__init__()
        self.dim_z = args.d_model
        # Here I manually set the patch length!!!!
        self.patch_len = args.patch_len
        # self.padding_idx = config.pad_token_id

        # Simple linear projection. Bad performance!
        # self.unary_factors = nn.Linear(args.enc_in, self.dim_z)

        self.iterator = PtEncoderIterator(args)
        self.norm = POTENTIAL2ACT[config.potential_func_z](dim=-1, eps=config.potential_eps)

        # XXX: This is a workaround to initialize the rotary embeddings
        '''
        Why we need two RoPE? Since the position mark along the channel and time dimension is not the same
        When receiving channel information, the positional embedding should be related to the channel id
        When receiving time information, the positional embedding should be related to the time stamp
        '''
        config_copy = PtConfig.from_dict(config.to_dict())
        config_copy.hidden_size = self.dim_z
        config_copy.num_attention_heads = args.n_heads
        config_copy.head_dim = self.dim_z // args.n_heads
        self.rotary_emb_time = LlamaRotaryEmbedding(config = config_copy)
        self.rotary_emb_channel = LlamaRotaryEmbedding(config = config_copy)

        # Information about the time series
        self.pred_len = args.pred_len
        self.seq_len = args.seq_len
        self.enc_in = args.enc_in
        self.num_iteration = args.e_layers
        self.patch_num = self.seq_len // self.patch_len
        # Decoder actually can have many ways to do that
        # Here I concatenate all the feature tensors with the dimension of dim_z, and project them into pred_len
        # [bs, enc_in, dim_z, patch_num] -> [bs, enc_in, patch_num*dim_z] -> [bs, enc_in, pred_len]
        self.prediction = nn.Linear(self.patch_num * self.dim_z, self.pred_len, bias=True)
        
        # Currently, no dropout is deployed
        # if args.dropout != None:
        #     self.dropout = nn.Dropout(args.dropout)

        # Currrently I embed self.patch_len points -> patch_len
        self.unary_factors = nn.Sequential(
			nn.Linear(self.patch_len, self.dim_z),
            nn.GELU(),
			nn.Linear(self.dim_z, self.dim_z)
		)


    def forward(
        self,
        time_series: torch.LongTensor = None,
        x_mark_enc = None,
        x_dec = None,
        dependency_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        unary_potentials: Optional[torch.FloatTensor] = None,
        output_dependencies: Optional[bool] = None,
        output_qzs: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        device = time_series.device
        # Z-norm for the time series for each singel channel
        means = time_series.mean(1, keepdim=True).detach()
        time_series = time_series - means
        stdev = torch.sqrt(
            torch.var(time_series, dim=1, keepdim=True, unbiased=False) + 1e-5)
        time_series /= stdev

        # [bs, length, enc_in] -> [bs, enc_in, length]
        time_series = time_series.transpose(1, 2) 
        time_series = time_series.reshape(-1, self.enc_in, self.patch_num, self.patch_len)
        unary_potentials = self.unary_factors(time_series) # [bs, enc_in, patch_num, dim_z]  
        # [bs, enc_in, patch_num, dim_z]  
        batch_size, enc_in, seq_len, _ = unary_potentials.shape
        # seq is actually the patch number
        dependency_mask_time = torch.ones((batch_size*enc_in, seq_len), device = device)
        dependency_mask_channel = torch.ones((batch_size*seq_len, enc_in), device = device)

        unary_type = unary_potentials.dtype
        # create position_ids as the original PT did
        if position_ids is None:
            device = time_series.device if time_series is not None else unary_potentials.device
            position_ids_time = torch.arange(
                0, seq_len, dtype=torch.long, device=device
            )
            position_ids_time = position_ids_time.unsqueeze(0)
            position_ids_channel = torch.arange(
                0, enc_in, dtype=torch.long, device=device
            )
            position_ids_channel = position_ids_channel.unsqueeze(0)
        # update dependency mask
        dependency_mask_channel = self._update_dependency_mask(
            dependency_mask_channel, enc_in, output_dependencies, unary_type
        )
        dependency_mask_time = self._update_dependency_mask(
            dependency_mask_time, seq_len, output_dependencies, unary_type
        )
        # now the initial unary potential is qz, and qz will be updated
        # [bs, enc_in, patch_num, dim_z]  
        qz = unary_potentials
        # for position_embedding_generation_channel

        position_embeddings_channel = self.rotary_emb_channel(unary_potentials.transpose(1, 2).reshape(batch_size * seq_len, enc_in, -1), position_ids_channel)
        # for position_embedding_generation
        position_embeddings_time = self.rotary_emb_time(unary_potentials.view(batch_size*enc_in, seq_len, -1), position_ids_time)
        

        # The following codes remain the same
        all_qzs = () if output_qzs else None
        all_qhs = () if output_dependencies else None

        for idx in range(self.num_iteration):
            if output_qzs:
                all_qzs += (qz,)

            qz = self.iterator(
                unary_potentials,
                qz,
                dependency_mask_channel=dependency_mask_channel,
                dependency_mask_time = dependency_mask_time,
                position_ids_time=position_ids_time,
                position_ids_channel=position_ids_channel,
                output_dependencies=output_dependencies,
                position_embeddings_time=position_embeddings_time,
                position_embeddings_channel=position_embeddings_channel
            )
        # add hidden states from the last decoder layer
        # if output_qzs:
        #     all_qzs += (qz,)
        
        # now qz should be [bs, enc_in, patch_num, dim_z]!!

        # The following final norm is removed, as empirically, doing so will be better
        # qz = self.norm(qz) 

        # project dim_z to one value, since currently no patching is applied
        dec_out = self.prediction(qz.reshape(batch_size, self.enc_in, -1)).permute(0, 2, 1) # [bs, pred_length, enc_in]
        # Normalize back
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len, 1))
        return dec_out


    
    def _update_dependency_mask(
        self, dependency_mask: torch.Tensor, seq_length, output_dependencies: bool, type
    ) -> torch.Tensor:

        attn_mask_converter = AttentionMaskConverter(is_causal=False)
        dependency_mask = attn_mask_converter.to_4d(
            dependency_mask, seq_length, dtype = type
        )
        
        # mask diagonals
        diag_mask = torch.eye(seq_length, dtype=dependency_mask.dtype, device=dependency_mask.device).unsqueeze(0).unsqueeze(0)
        dependency_mask = dependency_mask.masked_fill(diag_mask.to(torch.bool), torch.finfo(dependency_mask.dtype).min)
        return dependency_mask


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.model = PtModel(args)

    def forward(
        self,
        x_enc: Optional[torch.Tensor] = None,
        x_mark_enc = None,
        x_dec = None, 
        x_mark_dec = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        # Actually, only the x_enc is not None, all the others are none
        outputs = self.model(
            time_series = x_enc
        )
        return outputs