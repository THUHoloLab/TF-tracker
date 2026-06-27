import torch.nn.functional as F
import torch.nn.init

from models.common import *
from models.template import Template
from utils.losses import *


class FPNEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=512, recurrent=False):
        super(FPNEncoder, self).__init__()

        self.conv_bottom_0 = ConvBlock(
            in_channels=in_channels,
            out_channels=32,
            n_convs=2,
            kernel_size=1,
            padding=0,
            downsample=False,
        )
        self.conv_bottom_1 = ConvBlock(
            in_channels=32,
            out_channels=64,
            n_convs=2,
            kernel_size=5,
            padding=0,
            downsample=False,
        )
        self.conv_bottom_2 = ConvBlock(
            in_channels=64,
            out_channels=128,
            n_convs=2,
            kernel_size=5,
            padding=0,
            downsample=False,
        )
        self.conv_bottom_3 = ConvBlock(
            in_channels=128,
            out_channels=256,
            n_convs=2,
            kernel_size=3,
            padding=0,
            downsample=True,
        )
        self.conv_bottom_4 = ConvBlock(
            in_channels=256,
            out_channels=out_channels,
            n_convs=2,
            kernel_size=3,
            padding=0,
            downsample=False,
        )

        self.recurrent = recurrent
        if self.recurrent:
            self.conv_rnn = ConvLSTMCell(out_channels, out_channels, 1)

        self.conv_lateral_3 = nn.Conv2d(
            in_channels=256, out_channels=out_channels, kernel_size=1, bias=True
        )
        self.conv_lateral_2 = nn.Conv2d(
            in_channels=128, out_channels=out_channels, kernel_size=1, bias=True
        )
        self.conv_lateral_1 = nn.Conv2d(
            in_channels=64, out_channels=out_channels, kernel_size=1, bias=True
        )
        self.conv_lateral_0 = nn.Conv2d(
            in_channels=32, out_channels=out_channels, kernel_size=1, bias=True
        )

        self.conv_dealias_3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.conv_dealias_2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.conv_dealias_1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.conv_dealias_0 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.conv_out = nn.Sequential(
            ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                n_convs=1,
                kernel_size=3,
                padding=1,
                downsample=False,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
        )

        self.conv_bottleneck_out = nn.Sequential(
            ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                n_convs=1,
                kernel_size=3,
                padding=1,
                downsample=False,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
        )

    def reset(self):
        if self.recurrent:
            self.conv_rnn.reset()

    def forward(self, x):
        """
        :param x:
        :return: (highest res feature map, lowest res feature map)
        """

        # Bottom-up pathway
        c0 = self.conv_bottom_0(x)  # 31x31
        c1 = self.conv_bottom_1(c0)  # 23x23
        c2 = self.conv_bottom_2(c1)  # 15x15
        c3 = self.conv_bottom_3(c2)  # 5x5
        c4 = self.conv_bottom_4(c3)  # 1x1

        # Top-down pathway (with lateral cnx and de-aliasing)
        p4 = c4
        p3 = self.conv_dealias_3(
            self.conv_lateral_3(c3)
            + F.interpolate(p4, (c3.shape[2], c3.shape[3]), mode="bilinear")
        )
        p2 = self.conv_dealias_2(
            self.conv_lateral_2(c2)
            + F.interpolate(p3, (c2.shape[2], c2.shape[3]), mode="bilinear")
        )
        p1 = self.conv_dealias_1(
            self.conv_lateral_1(c1)
            + F.interpolate(p2, (c1.shape[2], c1.shape[3]), mode="bilinear")
        )
        p0 = self.conv_dealias_0(
            self.conv_lateral_0(c0)
            + F.interpolate(p1, (c0.shape[2], c0.shape[3]), mode="bilinear")
        )

        if self.recurrent:
            p0 = self.conv_rnn(p0)

        return self.conv_out(p0), self.conv_bottleneck_out(c4)

class ResBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim,
                               kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim,
                               kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, input):
        out = self.conv1(input)
        out = F.relu(out)
        out = self.conv2(out)
        out = input + out
        return out
class ResEncoder(nn.Module):
    def __init__(self, in_dim=1, conv_dim = 32, out_dim=32, num_blocks=4):
        super(ResEncoder, self).__init__()
        self.inputConv = nn.Conv2d(in_channels=in_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.ResBlocks = nn.Sequential(*[ResBlock(conv_dim) for i in range(num_blocks)])

        self.outputConv = nn.Conv2d(in_channels=conv_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, input):
        out = self.inputConv(input)
        out = self.ResBlocks(out)
        out = F.relu(out)
        out = self.outputConv(out)
        return out
class JointEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(JointEncoder, self).__init__()

        self.conv1 = ConvBlock(
            in_channels=in_channels, out_channels=32, kernel_size=3,padding=1,n_convs=2, downsample=True
        )
        self.conv2 = ConvBlock(
            in_channels=32, out_channels=64, kernel_size=3,padding=1,n_convs=2, downsample=True
        )
        self.convlstm0 = ConvLSTMCell(64, 64, 3)
        self.conv3 = ConvBlock(
            in_channels=64, out_channels=128, kernel_size=3,padding=1,n_convs=2, downsample=True
        )
        self.conv4 = ConvBlock(
            in_channels=128,
            out_channels=128,
            kernel_size=3,padding=1,
            n_convs=1,
            downsample=False,
        )

        '''self.conv1_heatmap = ConvBlock(
            in_channels=in_channels, out_channels=32, kernel_size=3, padding=1, n_convs=2, downsample=True
        )
        self.conv2_heatmap = ConvBlock(
            in_channels=32, out_channels=64, kernel_size=3, padding=1, n_convs=2, downsample=True
        )'''

        # Transformer Addition
        self.flatten = nn.Flatten()
        embed_dim = 128
        num_heads = 8
        self.multihead_attention0 = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

        self.prev_x_res = None # t-1
        #self.prev_prev_x_res = None  # t-2
        self.gates = nn.Conv2d(
                    in_channels=2*embed_dim,
                    out_channels=embed_dim,
                    kernel_size=3,padding=1)
        self.ls_layer = LayerScale(embed_dim)

        # Attention Mask Transformer
        self.fusion_layer0 = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1),  # 通道数减半
            nn.LeakyReLU(0.1),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),  # 通道数恢复
            nn.LeakyReLU(0.1),
        )

    def reset(self):
        self.convlstm0.reset()
        self.prev_x_res = None
        #self.prev_prev_x_res = None

    def forward(self, x, attn_mask=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.convlstm0(x)
        x = self.conv3(x)
        x = self.conv4(x)

        #heatmap_pre = self.conv1_heatmap(heatmap_pre)
        #heatmap_pre = self.conv2_heatmap(heatmap_pre)


        if self.prev_x_res is None:
            self.prev_x_res = Variable(torch.zeros_like(x))
        '''if self.prev_prev_x_res is None:
            self.prev_prev_x_res = torch.zeros_like(x)'''

        #x = self.fusion_layer0(torch.cat((x, self.prev_x_res, self.prev_prev_x_res), 1))
        x = self.fusion_layer0(torch.cat((x, self.prev_x_res), 1))

        x_attn = x.detach()
        x_attn = x_attn.flatten(2)  # [B, C, H*W]
        x_attn = x_attn.permute(2, 0, 1)  # [H*W, B, C]
        if self.training:
            x_attn = self.multihead_attention0(
                query=x_attn, key=x_attn, value=x_attn, attn_mask=attn_mask.bool()
            )[0].squeeze(0)
        else:
            x_attn = self.multihead_attention0(query=x_attn, key=x_attn, value=x_attn)[
                0
            ].squeeze(0)
        x_attn = x_attn.permute(1, 2, 0)  # [B, C, H*W]
        x_attn = x_attn.view(x.shape)  # [B, C, H, W]
        x = x + self.ls_layer(x_attn)

        gate_weight_1 = torch.sigmoid(self.gates(torch.cat((self.prev_x_res, x), 1)))
        x = self.prev_x_res * gate_weight_1 + x * (1 - gate_weight_1)
        #gate_weight_2 = torch.sigmoid(self.gates(torch.cat((self.prev_prev_x_res, x), 1)))
        #x = self.prev_x_res * gate_weight_1 + self.prev_prev_x_res * gate_weight_2 + x * (1 - gate_weight_1-gate_weight_2)

        #self.prev_prev_x_res = self.prev_x_res
        self.prev_x_res = x

        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        # gamma形状为 [1, C, 1, 1] 用于广播到所有空间位置
        self.gamma = nn.Parameter(init_values * torch.ones(1, dim, 1, 1))

    def forward(self, x):
        # x: [B, C, H, W]
        # gamma: [1, C, 1, 1] → 可以广播到 [B, C, H, W]
        gamma = self.gamma
        return x.mul_(gamma) if self.inplace else x * gamma


class TrackerNetC(Template):
    def __init__(
        self,
        representation="time_surfaces_1",
        max_unrolls=16,
        n_vis=8,
        feature_dim=1024,
        patch_size=31,
        init_unrolls=1,
        input_channels=None,
        **kwargs,
    ):
        super(TrackerNetC, self).__init__(
            representation=representation,
            max_unrolls=max_unrolls,
            init_unrolls=init_unrolls,
            n_vis=n_vis,
            patch_size=patch_size,
            **kwargs,
        )
        # Configuration
        self.grayscale_ref = True
        if not isinstance(input_channels, type(None)):
            self.channels_in_per_patch = input_channels

        # Architecture
        self.feature_dim = feature_dim
        self.redir_dim = self.feature_dim

        self.reference_encoder = ResEncoder(1, self.feature_dim, self.feature_dim)
        self.target_encoder = FPNEncoder(self.channels_in_per_patch, self.feature_dim)
        '''self.heatmap_encoder = nn.Conv2d(1, self.feature_dim, kernel_size=3,
                                         padding=1)  # FPNEncoder(1, self.feature_dim)'''
        # Correlation3 had k=1, p=0
        self.weight_ref = ResEncoder(2, 32, 32, num_blocks=2)
        self.weight_heatmap = ConvBlock(in_channels=2, out_channels=32, kernel_size=3,padding=1,n_convs=3, downsample=False)
        self.weight_corr = ResEncoder(self.feature_dim, 32, 32, num_blocks=2)
        self.weight_all = ResEncoder(32*3, 64, 64, num_blocks=2)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.joint_encoder = JointEncoder(
            in_channels=3 * self.feature_dim, out_channels=128
        )
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0)
        )
        self.heatmap_predictor = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        )
        self.heatmap_predictor[-1].bias.data.fill_(-2.19)
        self.flatten = nn.Flatten()

        # Operational
        self.loss = L1Truncated()
        self.name = f"corr_{self.representation}"

        # Persistent Tensors
        self.f_ref, self.d_ref = None, None

        self.correlation_maps = []
        self.inputs = []
        self.refs = []

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.fc_out.weight)

    def reset(self, _):
        self.d_ref, self.f_ref = None, None
        self.joint_encoder.reset()

    def forward(self, x, heatmap, i_unroll, attn_mask=None):
        '''heatmap_pre_pre = heatmap[:,:1]
        heatmap_pre = heatmap[:, 1:2]
        # Feature Extraction
        f0, _ = self.target_encoder(x[:, : self.channels_in_per_patch, :, :])
        if isinstance(self.f_ref, type(None)):
            self.f_ref = self.reference_encoder(
                x[:, self.channels_in_per_patch :, :, :]
            )
        f_heatmap_pre = self.heatmap_encoder(heatmap_pre)
        f_heatmap_pre_pre = self.heatmap_encoder(heatmap_pre_pre)
        ref_corr=self.ref_redir(x[:, self.channels_in_per_patch :, :, :])
        target_corr = self.target_redir(x[:, : self.channels_in_per_patch, :, :])
        f_corr_pre = f_heatmap_pre*ref_corr*target_corr
        f_corr_pre_pre = f_heatmap_pre_pre * ref_corr * target_corr
        f_corr = self.sigmoid(self.corr(torch.cat([f_corr_pre,f_corr_pre_pre], dim=1)))'''
        heatmap_pre_pre = heatmap[:, :1]
        heatmap_pre = heatmap[:, 1:2]
        heatmap_init = heatmap[:, 2:3]

        # Feature Extraction
        f0, _ = self.target_encoder(x[:, : self.channels_in_per_patch, :, :])
        if isinstance(self.f_ref, type(None)):
            self.f_ref = self.reference_encoder(
                x[:, self.channels_in_per_patch:, :, :]
            )

        corr_ref = self.weight_ref(torch.cat([heatmap_init, x[:, self.channels_in_per_patch:, :, :]], dim=1))
        corr_heatmap = self.weight_heatmap(torch.cat([heatmap_pre_pre, heatmap_pre], dim=1))
        i_tensor = torch.tensor(
            float(i_unroll),
            device=corr_heatmap.device,
            dtype=corr_heatmap.dtype,
        )
        corr_heatmap_w = torch.sigmoid(-0.2 * (i_tensor - 15.0))
        corr_ref_event = self.weight_corr(f0*self.f_ref)
        f_corr = self.weight_all(torch.cat([corr_ref, corr_heatmap_w*corr_heatmap,corr_ref_event], dim=1))
        f_corr = self.sigmoid(f_corr)

        self.H, self.W = x.shape[2:4]
        # Feature re-direction

        f = torch.cat([f_corr, self.f_ref, f0], dim=1)
        #heatmap = heatmap.unsqueeze(1)
        f = self.joint_encoder(f, attn_mask)
        offset = self.offset_predictor(f)
        heatmap = self.heatmap_predictor(f)
        heatmap = self.sigmoid(heatmap)
        offset=F.interpolate(offset, scale_factor=8, mode='bilinear', align_corners=False)
        heatmap = F.interpolate(heatmap, scale_factor=8, mode='bilinear', align_corners=False)
        # ======================
        # offset padding
        # ======================
        _, _, h_off, w_off = offset.shape
        pad_h = max(self.H - h_off, 0)
        pad_w = max(self.W - w_off, 0)
        if pad_h > 0 or pad_w > 0:
            offset = F.pad(offset, (0, pad_w, 0, pad_h), mode='replicate')
        # ======================
        # heatmap padding
        # ======================
        _, _, h_hm, w_hm = heatmap.shape
        pad_h = max(self.H - h_hm, 0)
        pad_w = max(self.W - w_hm, 0)
        if pad_h > 0 or pad_w > 0:
            heatmap = F.pad(heatmap, (0, pad_w, 0, pad_h), mode='replicate')

        return offset, heatmap
