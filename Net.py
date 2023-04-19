import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def swish(x):
    return x * torch.sigmoid(x)
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}
class Mlp(nn.Module):
    def __init__(self,inc):
        super(Mlp, self).__init__()
        self.hidden_size = inc
        self.transformer_dim = inc * 4
        self.dropout_rate = 0.1
        self.fc1 = Linear(self.hidden_size, self.transformer_dim)
        self.fc2 = Linear(self.transformer_dim, self.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(self.dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
class Attention(nn.Module):
    def __init__(self,inc):
        super(Attention, self).__init__()
        self.hidden_size = inc
        self.num_attention_heads = 16
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(self.hidden_size, self.all_head_size)
        self.key = Linear(self.hidden_size, self.all_head_size)
        self.value = Linear(self.hidden_size, self.all_head_size)

        self.out = Linear(self.hidden_size, self.hidden_size)
        self.attn_dropout = Dropout()
        self.proj_dropout = Dropout()

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output
class Block(nn.Module):
    def __init__(self,inc):
        super(Block, self).__init__()
        self.hidden_size = inc
        self.attention_norm = LayerNorm(inc, eps=1e-6)
        self.ffn_norm = LayerNorm(inc, eps=1e-6)
        self.ffn = Mlp(inc)
        self.attn = Attention(inc)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x
class Encoder(nn.Module):
    def __init__(self,inc):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(inc, eps=1e-6)
        for _ in range(6):
            layer = Block(inc)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, inc,patch):
        super(Embeddings, self).__init__()
        self.in_channels = inc
        patch_size = (patch, patch)
        self.patch_embeddings = Conv2d(in_channels=16,
                                       out_channels=2 * inc,
                                       kernel_size=patch_size,
                                       stride=patch_size)

        self.dropout = Dropout(0.1)


    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        num_patch = x.shape[1]
        position_embeddings = nn.Parameter(torch.zeros(1, num_patch, x.shape[2]))
        position_embeddings = position_embeddings.to(device)
        embeddings = x + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
class Transformer(nn.Module):
    def __init__(self,inc,patch):
        super(Transformer, self).__init__()
        self.in_channels = inc
        self.patch_size = patch
        self.embeddings = Embeddings(inc,patch)
        self.encoder = Encoder(2*inc)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded


class encoder_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(encoder_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

    

    

class decoder_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(decoder_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class TransUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(TransUNet, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.vis = Transformer(inc=256,patch=16)
        CH_FOLD2 = 1
        self.Conv1 = encoder_block(ch_in=16,ch_out=int(32*CH_FOLD2))
        self.Conv2 = encoder_block(ch_in=int(32*CH_FOLD2),ch_out=int(64*CH_FOLD2))
        self.Conv3 = encoder_block(ch_in=int(64*CH_FOLD2),ch_out=int(128*CH_FOLD2))
        self.Conv4 = encoder_block(ch_in=int(128*CH_FOLD2),ch_out=int(256*CH_FOLD2))
        self.Conv5 = encoder_block(ch_in=int(256*CH_FOLD2),ch_out=int(512*CH_FOLD2))
        

        self.Up5 = decoder_conv(ch_in=int(512), ch_out=int(256))
        self.Up_conv5 = encoder_block(ch_in=int(512), ch_out=int(256))

        self.Up4 = decoder_conv(ch_in=int(256), ch_out=int(128))
        self.Up_conv4 = encoder_block(ch_in=int(256), ch_out=int(128))

        self.Up3 = decoder_conv(ch_in=int(128), ch_out=int(64))
        self.Up_conv3 = encoder_block(ch_in=int(128), ch_out=int(64))

        self.Up2 = decoder_conv(ch_in=int(64), ch_out=int(32))
        self.Up_conv2 = encoder_block(ch_in=int(64), ch_out=int(32))

        self.Conv_1x1 = nn.Conv2d(int(32), output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        #x  B 16 512 512
        trans_input = self.vis(x)
        B, n_patch, hidden = trans_input.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        trans_input = trans_input.permute(0, 2, 1)
        trans_input = trans_input.contiguous().view(B, hidden, h, w) #B 32 L L
        
        #encoder
        x1 = self.Conv1(x) 
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)  
       
        # decoding + concat path
        d5 = self.Up5(trans_input)
        d5 = torch.cat((x4, d5), dim=1)  
        d5 = self.Up_conv5(d5)        

        d4 = self.Up4(d5)            
        d4 = torch.cat((x3, d4), dim=1) 
        d4 = self.Up_conv4(d4)        

        d3 = self.Up3(d4)             
        d3 = torch.cat((x2, d3), dim=1) 
        d3 = self.Up_conv3(d3)       

        d2 = self.Up2(d3)              
        d2 = torch.cat((x1, d2), dim=1)  
        d2 = self.Up_conv2(d2)          

        d1 = self.Conv_1x1(d2)        
        d1 = d1.squeeze(1)

        return torch.transpose(d1, -1, -2) * d1



