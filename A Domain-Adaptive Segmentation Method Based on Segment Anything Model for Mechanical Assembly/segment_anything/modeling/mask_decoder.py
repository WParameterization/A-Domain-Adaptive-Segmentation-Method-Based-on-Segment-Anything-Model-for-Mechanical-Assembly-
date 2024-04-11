# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels,  out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm,)
        self.conv2 = Conv2dReLU( out_channels,out_channels,kernel_size=3,padding=1, use_batchnorm=use_batchnorm,)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x




class Conv2dReLU(nn.Sequential):
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, stride=1,  use_batchnorm=True,):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size,stride=stride, padding=padding,  bias=not (use_batchnorm), )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)



class MaskDecoder(nn.Module):
    def __init__(  self, *, transformer_dim: int, transformer: nn.Module, num_multimask_outputs: int = 3,  activation: Type[nn.Module] = nn.GELU, iou_head_depth: int = 3,iou_head_hidden_dim: int = 256,) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        head_channels = 512
        decoder_channels = (256, 128, 64, 32)
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        self.n_skip=3

        if self.n_skip != 0:
            skip_channels = [512, 256, 64, 16]
            for i in range(4-self.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0
        else:
            skip_channels=[0,0,0,0]


        blocks = [DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)]
        self.blocks = nn.ModuleList(blocks)
        self.conv_more = Conv2dReLU(256, head_channels, kernel_size=1, use_batchnorm=True, )

        self.transformer_dim = transformer_dim
        self.transformer = transformer#

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)#
        self.num_mask_tokens = num_multimask_outputs #+ 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential( nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),LayerNorm2d(transformer_dim // 4),activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),activation(),)
        self.output_hypernetworks_mlps = nn.ModuleList([MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)for i in range(self.num_mask_tokens)] )

        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth )

    def forward( self, image_embeddings: torch.Tensor,image_pe: torch.Tensor, sparse_prompt_embeddings: torch.Tensor,dense_prompt_embeddings: torch.Tensor,  multimask_output: bool,res_features: torch.Tensor ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """


        masks, iou_pred = self.predict_masks(image_embeddings=image_embeddings,image_pe=image_pe, sparse_prompt_embeddings=sparse_prompt_embeddings, dense_prompt_embeddings=dense_prompt_embeddings,res_features=res_features )

        # Select the coroutputrect mask or masks for
        # if multimask_output:
        #     mask_slice = slice(1, None)
        # else:
        #     mask_slice = slice(0, 1)
        # masks = masks[:, mask_slice, :, :]
        # iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks( self, image_embeddings: torch.Tensor, image_pe: torch.Tensor, sparse_prompt_embeddings: torch.Tensor, dense_prompt_embeddings: torch.Tensor,res_features: torch.Tensor ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens


        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)

        #output_tokens1 = output_tokens.unsqueeze(0).expand(2, -1, -1)
        #print(output_tokens1.shape, sparse_prompt_embeddings.size(0))
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)


        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings


        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        b, c, h, w = src.shape

        # Run the transformer

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)


        src1 = self.conv_more(src)#

        for i, decoder_block in enumerate(self.blocks):#features
            if res_features is not None:
                skip = res_features[i] if (i < self.n_skip) else None
            else:
                skip = None
            src1 = decoder_block(src1, skip=skip)#
            #print(i,decoder_block)
            b1, c1, h1, w1 = src1.shape  # [h, token_num, h, w]

       # upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))#dianji  limian de mlp
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, c, token_num]
        #b, c, h, w = upscaled_embedding.shape  # [h, token_num, h, w]


       # masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        masks1 = (hyper_in @ src1.view(b1, c1, h1 * w1)).view(b, -1, h1, w1)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)#


        return masks1, iou_pred


class MaskDecoder_line(nn.Module):
    def __init__(  self, *, transformer_dim: int, transformer: nn.Module, num_multimask_outputs: int = 3,  activation: Type[nn.Module] = nn.GELU, iou_head_depth: int = 3,iou_head_hidden_dim: int = 256,) -> None:

        super().__init__()
        head_channels = 512
        decoder_channels = (256, 128, 64, 32)
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        self.n_skip=3

        if self.n_skip != 0:
            skip_channels = [512, 256, 64, 16]
            for i in range(4-self.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0
        else:
            skip_channels=[0,0,0,0]


        blocks = [DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)]
        self.blocks = nn.ModuleList(blocks)
        self.conv_more = Conv2dReLU(256, head_channels, kernel_size=1, use_batchnorm=True, )
        self.conv_more_final = Conv2dReLU(9, 1, kernel_size=1, use_batchnorm=True, )

        self.transformer_dim = transformer_dim
        self.transformer = transformer#

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)#
        self.num_mask_tokens = num_multimask_outputs #+ 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential( nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),LayerNorm2d(transformer_dim // 4),activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),activation(),)
        self.output_hypernetworks_mlps = nn.ModuleList([MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)for i in range(self.num_mask_tokens)] )

        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth )

    def forward( self, image_embeddings: torch.Tensor,image_pe: torch.Tensor, sparse_prompt_embeddings: torch.Tensor,dense_prompt_embeddings: torch.Tensor,  multimask_output: bool,res_features: torch.Tensor ) -> Tuple[torch.Tensor, torch.Tensor]:

        masks, iou_pred = self.predict_masks(image_embeddings=image_embeddings,image_pe=image_pe, sparse_prompt_embeddings=sparse_prompt_embeddings, dense_prompt_embeddings=dense_prompt_embeddings,res_features=res_features )

        return masks, iou_pred

    def predict_masks( self, image_embeddings: torch.Tensor, image_pe: torch.Tensor, sparse_prompt_embeddings: torch.Tensor, dense_prompt_embeddings: torch.Tensor,res_features: torch.Tensor ) -> Tuple[torch.Tensor, torch.Tensor]:


        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings

        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)

        src1 = self.conv_more(src)#
        for i, decoder_block in enumerate(self.blocks):#features
            if res_features is not None:
                skip = res_features[i] if (i < self.n_skip) else None
            else:
                skip = None
            src1 = decoder_block(src1, skip=skip)#
            b1, c1, h1, w1 = src1.shape  # [h, token_num, h, w]

       # upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))#
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, c, token_num]
        #b, c, h, w = upscaled_embedding.shape  # [h, token_num, h, w]
        masks1 = (hyper_in @ src1.view(b1, c1, h1 * w1)).view(b, -1, h1, w1)
        masks1 = self.conv_more_final(masks1)

        #iou_pred = self.iou_prediction_head(iou_token_out)


        return masks1,#, iou_pred



# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
