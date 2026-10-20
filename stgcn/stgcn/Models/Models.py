import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from stgcn.Utils import Graph


class AttentionModule(nn.Module):
    """Attention module for spatial-temporal features."""
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)


class GraphConvolution(nn.Module):
    """The basic module for applying a graph convolution."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels,
                              out_channels * kernel_size,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)
        
        # Add attention mechanism
        self.attention = AttentionModule(out_channels)

    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        x = self.attention(x)
        return x.contiguous()


class st_gcn(nn.Module):
    """Applies a spatial temporal graph convolution over an input graph sequence."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1

        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = GraphConvolution(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels,
                      (kernel_size[0], 1),
                      (stride, 1),
                      padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x)


class StreamSpatialTemporalGraph(nn.Module):
    """Enhanced Spatial temporal graph convolutional networks."""
    def __init__(self, in_channels, graph_args, num_class=None,
                 edge_importance_weighting=True, **kwargs):
        super().__init__()
        # Load graph.
        graph = Graph(**graph_args)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # Networks.
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        
        # Enhanced network architecture with more layers and attention
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 512, kernel_size, 2, **kwargs),  # Additional layer
            st_gcn(512, 512, kernel_size, 1, **kwargs)   # Additional layer
        ))

        # Enhanced edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # Global attention mechanism
        self.global_attention = AttentionModule(512)

        if num_class is not None:
            self.cls = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_class, kernel_size=1)
            )
        else:
            self.cls = lambda x: x

    def forward(self, x):
        # data normalization.
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N, C, T, V)

        # forward.
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = gcn(x, self.A * importance)

        # Apply global attention
        x = self.global_attention(x)
        
        x = F.avg_pool2d(x, x.size()[2:])
        x = self.cls(x)
        x = x.view(x.size(0), -1)

        return x


class TwoStreamSpatialTemporalGraph(nn.Module):
    """Enhanced Two inputs spatial temporal graph convolutional networks."""
    def __init__(self, graph_args, num_class, edge_importance_weighting=True,
                 **kwargs):
        super().__init__()
        self.pts_stream = StreamSpatialTemporalGraph(3, graph_args, None,
                                                     edge_importance_weighting,
                                                     **kwargs)
        self.mot_stream = StreamSpatialTemporalGraph(2, graph_args, None,
                                                     edge_importance_weighting,
                                                     **kwargs)

        # Enhanced fusion network
        self.fusion = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_class)
        )

    def forward(self, inputs):
        out1 = self.pts_stream(inputs[0])
        out2 = self.mot_stream(inputs[1])

        concat = torch.cat([out1, out2], dim=-1)
        out = self.fusion(concat)

        return torch.sigmoid(out)