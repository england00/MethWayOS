import torch
import torch.nn as nn
import torch.nn.functional as F
from src.surv_path.original_modules.cross_attention import FeedForward, MMAttentionLayer


# SurvPath link
# https://github.com/mahmoodlab/SurvPath


def SNN_Block(dim1, dim2, dropout=0.25):
    """
    Multilayer Perception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))

''' SurvPath Definition '''
class SurvPath(nn.Module):
    def __init__(self,
                 meth_sizes: [],
                 rnaseq_sizes: [],
                 n_classes: int = 4,
                 dropout: float = 0.25,
                 methylation_islands_statistics: bool = False):
        super(SurvPath, self).__init__()

        # Parameters
        self.n_classes = n_classes
        self.projection_dimensionality = 256                                  # [128, 256, 512]
        self.hidden = [256, 256]                                              # [128, 128] [256, 256], [512, 512]
        self.gene_groups = len(rnaseq_sizes)

        # Gene Expression encoder
        sig_networks = []
        for input_dimension in rnaseq_sizes:
            fc_omic = [SNN_Block(dim1=input_dimension, dim2=self.hidden[0])]
            for i, _ in enumerate(self.hidden[1:]):
                fc_omic.append(SNN_Block(dim1=self.hidden[i], dim2=self.hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.gene_expression_signature_networks = nn.ModuleList(sig_networks)

        # Methylation encoder
        self.methylation_islands_statistics = methylation_islands_statistics
        if self.methylation_islands_statistics:
            fc_omic = [SNN_Block(dim1=5, dim2=self.hidden[0])]
            for i, _ in enumerate(self.hidden[1:]):
                fc_omic.append(SNN_Block(dim1=self.hidden[i], dim2=self.hidden[i + 1], dropout=0.25))
            self.methylation_signature_networks = nn.Sequential(*fc_omic)
        else:
            sig_networks = []
            for input_dimension in meth_sizes:
                fc_omic = [SNN_Block(dim1=input_dimension, dim2=self.hidden[0])]
                for i, _ in enumerate(self.hidden[1:]):
                    fc_omic.append(SNN_Block(dim1=self.hidden[i], dim2=self.hidden[i + 1], dropout=0.25))
                sig_networks.append(nn.Sequential(*fc_omic))
            self.methylation_signature_networks = nn.ModuleList(sig_networks)

        # SurvPath Cross Attention
        self.identity = nn.Identity()
        self.cross_attender = MMAttentionLayer(
            dim=self.projection_dimensionality,
            dim_head=int(self.projection_dimensionality / 2),
            heads=1,
            residual=False,
            dropout=dropout,
            num_pathways = self.gene_groups,
        )

        # Logits
        self.feed_forward = FeedForward(int(self.projection_dimensionality / 2), dropout=dropout)
        self.layer_norm = nn.LayerNorm(int(self.projection_dimensionality / 2))
        self.to_logits = nn.Sequential(
            nn.Linear(self.projection_dimensionality, int(self.projection_dimensionality // 4)),
            nn.ReLU(),
            nn.Linear(int(self.projection_dimensionality / 4), self.n_classes)
        )

    def forward(self, islands, genes, inference: bool = False):
        # Methylation Fully connected layer for each signature
        if self.methylation_islands_statistics:
            h_meth = [self.methylation_signature_networks(sig_feat.float()) for sig_feat in islands]
        else:
            h_meth = [self.methylation_signature_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(islands)]
        # H_bag: (Mx1xd_k) --> (1xMxd_k)
        # M --> columns number in meth signature, d_k --> embedding dimension
        h_meth_bag = torch.stack(h_meth).squeeze(1).unsqueeze(0)                        # (1xMxd_k)

        # Gene Expression Fully connected layers for each group signature
        g_rnaseq = [self.gene_expression_signature_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(genes)]
        # G_bag: (Nx1xd_k) --> (1xNxd_k)
        # N --> columns number in rnaseq signature, d_k --> embedding dimension
        g_rnaseq_bag = torch.stack(g_rnaseq).squeeze(1).unsqueeze(0)                    # (1xNxd_k)

        # Cross-Attention results
        tokens = torch.cat([g_rnaseq_bag, h_meth_bag], dim=1)                   # (1x[N+M]xd_k)
        tokens = self.identity(tokens)
        self_attention_rnaseq = []
        cross_attention_rnaseq = []
        cross_attention_meth = []
        if inference:
            # multimodal_embedding: (1x[N+M]x[d_k/2])
            # self_attention_rnaseq: (NxN)
            # cross_attention_rnaseq: (NxM)
            # cross_attention_meth: (MxN)
            # N --> columns number in rnaseq signature, M --> columns number in meth signature, d_k --> embedding dimension
            multimodal_embedding, self_attention_rnaseq, cross_attention_rnaseq, cross_attention_meth = self.cross_attender(x=tokens, mask=None, return_attention=True)
        else:
            # multimodal_embedding: (1x[N+M]x[d_k/2])
            multimodal_embedding = self.cross_attender(x=tokens, mask=None, return_attention=False)

        # Feedforward and Layer Normalization
        multimodal_embedding = self.feed_forward(multimodal_embedding)                  # (1x[N+M]x[d_k/2])
        multimodal_embedding = self.layer_norm(multimodal_embedding)                    # (1x[N+M]x[d_k/2])

        # Modality Specific Mean
        rnaseq_postSA_embedding = multimodal_embedding[:, :self.gene_groups, :]         # (1xNx[d_k/2])
        rnaseq_postSA_embedding = torch.mean(rnaseq_postSA_embedding, dim=1)            # (1x[d_k/2])
        meth_postSA_embed = multimodal_embedding[:, self.gene_groups:, :]               # (1xMx[d_k/2])
        meth_postSA_embed = torch.mean(meth_postSA_embed, dim=1)                        # (1x[d_k/2])

        # Modalities Aggregation
        embedding = torch.cat([rnaseq_postSA_embedding, meth_postSA_embed], dim=1)  # (1x[d_k])

        # Survival Layer
        # size   --> (1, n_classes)
        # domain --> R
        logits = self.to_logits(embedding)
        # hazards: probability of patient death in interval j
        # size   --> (1, n_classes)
        # domain --> [0, 1]
        hazards = torch.sigmoid(logits)
        # survs: probability of patient survival after time t
        # size   --> (1, n_classes)
        # domain --> [0, 1]
        surv = torch.cumprod(1 - hazards, dim=1)
        # Y: predicted probability distribution
        # size   --> (1, n_classes)
        # domain --> [0, 1] (probability distribution)
        Y = F.softmax(logits, dim=1)

        attention_scores = {}
        if inference:
            attention_scores = {'self_attention_rnaseq': self_attention_rnaseq, 'cross_attention_rnaseq': cross_attention_rnaseq, 'cross_attention_meth': cross_attention_meth}

        return hazards, surv, Y, attention_scores

    def get_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
