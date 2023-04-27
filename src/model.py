"""
Main file containing core LGVAVideoQA class code.
"""
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat
from utils.graph import Graph
from utils.attn import EdgeTransformer, ResidualCrossAttentionBlock, get_mask

@dataclass
class LGVAConfig:
    # Language-Guided Visual Aggregation params
    split: str = 'train'
    n_ca_heads: int = 12
    n_et_layers: int = 2
    n_et_heads: int = 5
    n_gnn_layers: int = 2
    n_frames: int = 16
    n_bboxes: int = 10
    ca_dropout: float = 0.1
    et_dropout: float = 0.3
    d_input: int = 768 # size of the input vision-language embeddings
    @classmethod
    def from_args(cls, args):
        return cls(n_ca_heads=args.n_ca_heads,
                   n_et_layers=args.n_et_layers,
                   n_et_heads=args.n_et_heads,
                   n_gnn_layers=args.n_gnn_layers,
                   n_frames=args.n_frames,
                   ca_dropout=args.ca_dropout,
                   et_dropout=args.et_dropout,
                   d_input=args.d_input,
                   split=args.split
                   )



class LGVAModel(nn.Module):
    """
    Takes as input a sequence of image-language encoding and outputs weighted frames over the inputs,
    to help analyze downstream discriminative video-language tasks.
    """

    def __init__(self, config: LGVAConfig):
        super().__init__()
        self.config = config
        self.etrans = EdgeTransformer(config)
        self.gnn    = Graph(dim_in=config.d_input, dim_hidden=config.d_input//2, dim_out=config.d_input, num_layers=config.n_gnn_layers, dropout=config.et_dropout)
        self.lca    = ResidualCrossAttentionBlock(d_model=config.d_input, n_head=config.n_ca_heads, dropout=config.ca_dropout)
        self.gca    = ResidualCrossAttentionBlock(d_model=config.d_input, n_head=config.n_ca_heads, dropout=config.ca_dropout)

        self.gcn_atten_pool_region = nn.Sequential(
            nn.Linear(config.d_input, config.d_input // 2),
            nn.Tanh(),
            nn.Linear(config.d_input // 2, 1),
            nn.Softmax(dim=-2))

        self.gcn_atten_pool_frame = nn.Sequential(
            nn.Linear(config.d_input, config.d_input // 2),
            nn.Tanh(),
            nn.Linear(config.d_input // 2, 1),
            nn.Softmax(dim=-2))



    def forward(self, item_dict):
        """
        Performs the Frame Aggregation operation on the input embeddings.
        Returns weighted visual embeddings.
        item_dict['video_features']:            torch.tensor of shape (B, L, D_in) with frame embeddings of size D_in
        item_dict['text_caption_features']:     torch.tensor of shape (B, L, D_in) with caption embeddings of size D_in
        item_dict['bbox_features']:             torch.tensor of shape (B, L, M, D_in) with region embeddings of size D_in
        item_dict['text_query_token_features']: torch.tensor of shape (B, L_tokens, D_in) with query token embeddings
        item_dict['text_query_features']:       torch.tensor of shape (B, D_in) with global query embeddings
        """

        vFeature = rearrange(item_dict['video_features'], 'b t c -> t b c')
        cFeature = rearrange(item_dict['text_caption_features'], 'b t c -> t b c')
        qFeature = rearrange(item_dict['text_query_features'].unsqueeze(dim=1), 'b n c -> n b c')
        tFeature = rearrange(item_dict['text_query_token_features'], 'b n c -> n b c')

        rFeature = item_dict['bbox_features'][:, :, 0, :, :]
        batch_size, num_frame, region_pframe, feat_dim = rFeature.shape
        rFeature = rearrange(rFeature, 'b t o c -> (t o) b c')
        rFeature = self.lca(rFeature, tFeature, tFeature)
        rFeature = rearrange(rFeature, '(t o) b c -> (b t) o c', t=num_frame)

        # Adj Note Relationship
        R = self.gnn.build_graph(rFeature)
        R = R.view(batch_size * 4, 4, region_pframe * region_pframe)
        graph_mask = get_mask(torch.tensor([4] * batch_size * 4, dtype=torch.long), 4).cuda()
        R = self.etrans(x=R, attn_mask=graph_mask)[0]
        R = R.view(batch_size * 16, region_pframe, region_pframe)
        R = F.softmax(R, dim=-1)

        # Graph Neural Network
        X_o = self.gnn(rFeature, R)
        X_o += rFeature

        # Region Attention Pooling
        att_region = self.gcn_atten_pool_region(X_o)
        gcn_att_pool_region = torch.sum(X_o * att_region, dim=1)
        rFeature = rearrange(gcn_att_pool_region, '(b t) c-> t b c', t=num_frame)

        # Global Cross Attention
        vFeature = vFeature + cFeature + rFeature
        vFeature = self.gca(vFeature, qFeature, qFeature)

        # Frame Attention Pooling
        vFeature = rearrange(vFeature, 't b d-> b t d')
        att_frame = self.gcn_atten_pool_frame(vFeature)
        vFeature = torch.sum(vFeature * att_frame, dim=1)

        return vFeature


