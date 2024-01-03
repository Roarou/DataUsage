import logging
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from Model.utils import PointNetSetAbstractionMsg, query_ball_point, index_points


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.00)
    elif type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.00)


class RFFNet(nn.Module):
    def __init__(self, channel_list, input_dim):
        super(RFFNet, self).__init__()

        # Initialize the convolutional layers
        self.layers = nn.ModuleList()

        # First convolutional layer
        self.layers.append(nn.Conv2d(in_channels=1,
                                     out_channels=channel_list[0],
                                     kernel_size=(input_dim, 1)))

        # Remaining convolutional layers
        for i in range(len(channel_list) - 1):
            self.layers.append(nn.Conv2d(in_channels=channel_list[i],
                                         out_channels=channel_list[i + 1],
                                         kernel_size=(1, 1)))

    def forward(self, x):
        # Forward pass through all layers
        for layer in self.layers:
            x = layer(x)
        return x


def create_rFF3d(channel_list, num_points, dim):
    rFF = nn.ModuleList(
        [
            nn.Conv3d(
                in_channels=channel_list[i],
                out_channels=channel_list[i + 1],
                kernel_size=(1, 1, 1),
            )
            for i in range(len(channel_list) - 1)
        ]
    )
    rFF.insert(
        0,
        nn.Conv3d(
            in_channels=1,
            out_channels=channel_list[0],
            kernel_size=(1, num_points, dim),
        ),
    )

    return rFF


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss

class Point_Transformer(nn.Module):
    def __init__(self, num_classes=6):
        super(Point_Transformer, self).__init__()

        # Activation function - Mish is used here for its smoothness and non-monotonic properties.
        self.actv_fn = Mish()

        # Dropout probability for regularization.
        self.p_dropout = 0.5

        # Whether to use normal channels in the input data.
        self.norm_channel = False

        # Dimensionality of the input points. It's 6 if normal channels are used, otherwise 3 (x, y, z coordinates).
        self.input_dim = 3

        # The number of sort networks to be used in local feature generation.
        self.num_sort_nets = 10

        # The number of top k points to consider in the SortNet.
        self.top_k = 16

        # Dimension of the model used in transformer layers.
        self.d_model = 64

        # Local feature generation with a series of 1D convolutional layers.
        self.sort_ch = [64, 128, 256]
        self.sort_cnn = RFFNet(self.sort_ch, self.input_dim)
        self.sort_cnn.apply(init_weights)

        # Batch normalization applied after each convolutional layer in local feature generation.
        self.sort_bn = nn.ModuleList(
            [nn.BatchNorm2d(num_features=self.sort_ch[i]) for i in range(len(self.sort_ch))]
        )

        # Creating a series of SortNet modules for local feature processing.
        self.sortnets = nn.ModuleList(
            [SortNet(self.sort_ch[-1], self.input_dim, self.actv_fn, top_k=self.top_k) for _ in range(self.num_sort_nets)]
        )

        # Global feature generation with a similar approach to local features.
        self.global_ch = [64, 128, 256]
        self.global_cnn = RFFNet(self.global_ch, self.input_dim)
        self.global_cnn.apply(init_weights)

        # Batch normalization for global feature generation.
        self.global_bn = nn.ModuleList(
            [nn.BatchNorm2d(num_features=self.global_ch[i]) for i in range(len(self.global_ch))]
        )

        # Transformer encoder layer for self-attention in global features.
        self.global_selfattention = nn.TransformerEncoderLayer(self.global_ch[-1], nhead=8)

        # Set abstraction layers for hierarchical feature extraction.
        in_channel = self.global_ch[-1]
        self.sa1 = PointNetSetAbstractionMsg(
            256, [0.1, 0.2, 0.4], [16, 32, 64], in_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        self.sa2 = PointNetSetAbstractionMsg(
            64, [0.2, 0.4, 0.6], [32, 64, 128], 320, [[32, 64, 128], [64, 64, 128], [64, 128, 253]]
        )

        # Applying weight initialization to set abstraction layers.
        self.sa1.apply(init_weights)
        self.sa2.apply(init_weights)

        # Transformer decoder layers for combining local and global features.
        self.decoder_layer = nn.TransformerDecoderLayer(self.d_model, nhead=8)
        self.last_layer = PTransformerDecoderLayer(self.d_model, nhead=8, last_dim=self.global_ch[-1])
        self.custom_decoder = PTransformerDecoder(self.decoder_layer, 8, self.last_layer)
        self.transformer_model = nn.Transformer(
            d_model=self.d_model, nhead=8, num_encoder_layers=4, num_decoder_layers=4, custom_decoder=self.custom_decoder
        )
        self.transformer_model.apply(init_weights)

        # Decoder layers for part segmentation.
        self.interp_decoder_layer = nn.TransformerDecoderLayer(self.global_ch[-1], nhead=8)
        self.interp_last_layer = PTransformerDecoderLayer(self.global_ch[-1], nhead=8, last_dim=512)
        self.interp_decoder = PTransformerDecoder(self.interp_decoder_layer, 1, self.interp_last_layer)

        # Final convolutional layers for per-point classification.
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(self.p_dropout)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

        # Additional dropout layers for regularization.
        self.dropout1 = nn.Dropout(p=self.p_dropout)
        self.dropout2 = nn.Dropout(p=self.p_dropout)
        self.dropout3 = nn.Dropout(p=self.p_dropout)

    def forward(self, input):
        # The forward pass processes the input through several stages:
        # 1. Global feature generation
        # 2. Local feature generation
        # 3. Point Transformer for feature fusion
        # 4. Part segmentation and final classification

        #############################################
        ## Global Features
        #############################################
        xyz = input
        B, _, _ = xyz.shape
        # Normalizing the input if normal channels are used.
        if self.norm_channel:
            norm = None
            xyz = xyz[:, :3, :]
        else:
            norm = None

        x_global = input.unsqueeze(dim=1)
        print(type(x_global))
        # print(x_global.shape)
        # Processing through global feature generation layers.
        for i, global_conv in enumerate(self.global_cnn):
            bn = self.global_bn[i]
            conv = global_conv(x_global)
            norm = bn(conv)
            x_global = self.actv_fn(norm)

        x_global = x_global.squeeze(dim=2)
        # print(x_global.shape)
        x_global = x_global.permute(2, 0, 1)
        # print(x_global.shape)
        x_global = self.global_selfattention(x_global)
        x_global = x_global.permute(1, 2, 0)
        # print(x_global.shape)

        l1_xyz, l1_points = self.sa1(xyz, x_global)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        out = torch.cat([l2_xyz, l2_points], dim=1)

        #############################################
        ## Local Features
        #############################################

        x_local = input.unsqueeze(dim=1)
        print(x_local.shape)
        # Processing through local feature generation layers.
        for i, sort_conv in enumerate(self.sort_cnn):
            bn = self.sort_bn[i]
            x_local = self.actv_fn(bn(sort_conv(x_local)))
        x_local = x_local.transpose(2, 1)
        print(x_local.shape)
        x_local_sorted = torch.cat(
            [sortnet(x_local, input)[0] for sortnet in self.sortnets], dim=-1
        )
        print(x_local.shape)
        #############################################
        ## Point Transformer
        #############################################

        source = out.permute(2, 0, 1)
        target = x_local_sorted.permute(2, 0, 1)
        embedding = self.transformer_model(source, target)

        #############################################
        ## Part Segmentation
        #############################################
        x_interp = x_global.permute(2, 0, 1)
        input_feat = self.interp_decoder(x_interp, embedding)
        input_feat = input_feat.permute(1, 2, 0)

        # Final fully connected layers for classification.
        x = self.actv_fn(self.bn1(self.conv1(input_feat)))
        x = self.drop1(x)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        output = x.permute(0, 2, 1)

        return output, None



class SortNet(nn.Module):
    def __init__(self, num_feat, input_dims, actv_fn=F.relu, feat_dims=256, top_k=16):
        """This function returns a sorted Tensor of Points. The Points are sorted
        according to a 1D score which is generated using rFF's (row-wise Feed Forward).

        Arguments:
            config {config.Config} -- Config class holding network parameters
            num_feat {int} -- number of features (dims) per point
            device {torch.device} -- Device to run (CPU or GPU)

        Keyword Arguments:
            mode {str} -- Mode to create score (default: {"max"})
        """
        super(SortNet, self).__init__()

        self.num_feat = num_feat
        self.actv_fn = actv_fn
        self.input_dims = input_dims

        self.top_k = top_k
        self.d_model = 512
        self.radius = 0.3
        self.max_radius_points = 64

        self.input_selfattention_layer = nn.TransformerEncoderLayer(
            self.num_feat, nhead=8
        )
        self.input_selfattention = nn.TransformerEncoder(
            self.input_selfattention_layer, num_layers=2
        )

        self.feat_channels = [64, 16, 1]
        self.feat_generator = RFFNet(self.feat_channels, num_feat)
        self.feat_generator.apply(init_weights)
        self.feat_bn = nn.ModuleList(
            [
                nn.BatchNorm2d(num_features=self.feat_channels[i])
                for i in range(len(self.feat_channels))
            ]
        )

        self.radius_ch = [128, 256, self.d_model]
        self.radius_cnn = create_rFF3d(self.radius_ch, self.max_radius_points + 1, 6)
        self.radius_cnn.apply(init_weights)
        self.radius_bn = nn.ModuleList(
            [
                nn.BatchNorm3d(num_features=self.radius_ch[i])
                for i in range(len(self.radius_ch))
            ]
        )

        dim_flatten = self.d_model * self.top_k
        self.flatten_linear_ch = [dim_flatten, 1024, self.d_model]
        self.flatten_linear = nn.ModuleList(
            [
                nn.Linear(
                    in_features=self.flatten_linear_ch[i],
                    out_features=self.flatten_linear_ch[i + 1],
                )
                for i in range(len(self.flatten_linear_ch) - 1)
            ]
        )
        self.flatten_linear.apply(init_weights)
        self.flatten_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.flatten_linear_ch[i + 1])
                for i in range(len(self.flatten_linear_ch) - 1)
            ]
        )

    def forward(self, sortvec, input):

        top_k = self.top_k
        sortvec_feat = sortvec
        feat_dim = input.shape[1]

        sortvec_att = sortvec.squeeze(dim=1)
        sortvec_att = sortvec_att.permute(2, 0, 1)
        sortvec_att = self.input_selfattention(sortvec_att)
        sortvec_att = sortvec_att.permute(1, 2, 0)
        sortvec = sortvec_att.unsqueeze(dim=1)

        for i, conv in enumerate(self.feat_generator):
            bn = self.feat_bn[i]
            sortvec = self.actv_fn(bn(conv(sortvec)))

        topk = torch.topk(sortvec, k=top_k, dim=-1)
        indices = topk.indices.squeeze()
        sorted_input = torch.zeros((sortvec_feat.shape[0], feat_dim, top_k)).to(
            input.device
        )

        if top_k == 1:
            indices = indices.unsqueeze(dim=-1)

        sorted_input = index_points(input.permute(0, 2, 1), indices).permute(0, 2, 1)

        all_points = input.permute(0, 2, 1)
        query_points = sorted_input.permute(0, 2, 1)

        radius_indices = query_ball_point(
            self.radius,
            self.max_radius_points,
            all_points[:, :, :3],
            query_points[:, :, :3],
        )

        radius_points = index_points(all_points, radius_indices)

        radius_centroids = query_points.unsqueeze(dim=-2)
        radius_grouped = torch.cat([radius_centroids, radius_points], dim=-2).unsqueeze(
            dim=1
        )

        for i, radius_conv in enumerate(self.radius_cnn):
            bn = self.radius_bn[i]
            radius_grouped = self.actv_fn(bn(radius_conv(radius_grouped)))

        radius_grouped = radius_grouped.squeeze()
        sorted_idx = indices
        sorted_input = radius_grouped

        sorted_input = torch.flatten(sorted_input, start_dim=1)

        for i, linear in enumerate(self.flatten_linear):
            bn = self.flatten_bn[i]
            sorted_input = self.actv_fn(bn(linear(sorted_input)))

        sorted_input = sorted_input.unsqueeze(dim=-1)

        return sorted_input, sorted_idx


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, last_dim=256, dropout=0.1, activation=F.relu):
        super(PTransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, 512)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(512, last_dim)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))

        return tgt


class PTransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    def __init__(self, decoder_layer, num_layers, last_layer, norm=None):
        super(PTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.last_layer = last_layer
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for i in range(self.num_layers):
            output = self.layers[i](
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm:
            output = self.norm(output)

        output = self.last_layer(output, memory)

        return output