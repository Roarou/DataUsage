import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
import torch.nn.functional as F
from torch_geometric.nn import PointConv, fps, radius, global_max_pool

class SAModule(torch.nn.Module):
    """ PointNet++ Set Abstraction (SA) Module """
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio  # sampling ratio
        self.r = r  # radius of neighborhood
        self.conv = PointConv(nn)  # applying a shared MLP to each neighborhood

    def forward(self, x, pos, batch):
        # FPS sampling on 'pos' with respect to batch
        idx = fps(pos, batch, ratio=self.ratio)

        # Radius search in 'pos' with respect to the centroids at 'pos[idx]'
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)

        # Constructing edge indices and applying PointConv
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)

        # Max pooling over each neighborhood and updating 'batch' and 'pos'
        x = global_max_pool(x, batch[idx])
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class SpineSegmentationNet(torch.nn.Module):
    """ The PointNet++ model for spine segmentation """
    def __init__(self):
        super(Net, self).__init__()

        # Encoder part using SAModule
        self.sa1_module = SAModule(0.5, 0.2, Seq(Lin(3, 64), ReLU(), Lin(64, 64), ReLU(), Lin(64, 128)))
        self.sa2_module = SAModule(0.25, 0.4, Seq(Lin(128, 128), ReLU(), Lin(128, 128), ReLU(), Lin(128, 256)))

        # Decoder part using linear layers
        self.lin1 = Lin(256, 256)
        self.lin2 = Lin(256, 128)
        self.lin3 = Lin(128, 1)  # single output for binary segmentation

    def forward(self, data):
        # Forward pass through the encoder part
        x, pos, batch = data.x, data.pos, data.batch
        x1, pos1, batch1 = self.sa1_module(x, pos, batch)
        x2, pos2, batch2 = self.sa2_module(x1, pos1, batch1)

        # Forward pass through the decoder part
        x = self.lin1(x2)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        return F.sigmoid(x)  # return probabilities for each point being part of the spine