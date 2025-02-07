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
    def __init__(self, meth_sizes: [], rnaseq_sizes: [], n_classes: int = 4, dropout: float = 0.25):
        super(SurvPath, self).__init__()

        # Parameters
        self.n_classes = n_classes
        self.projection_dimensionality = 128
        self.hidden = [128, 128]
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
        # Gene Expression Fully connected layers for each group signature
        g_rnaseq = [self.gene_expression_signature_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(genes)] ### each omic signature goes through it's own FC layer
        g_rnaseq_bag = torch.stack(g_rnaseq)
        print('g_rnaseq_bag: ', g_rnaseq_bag.shape)

        # Methylation Fully connected layer for each signature
        h_meth = [self.methylation_signature_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(islands)]
        h_meth_bag = torch.stack(h_meth)
        print('h_meth_bag: ', h_meth_bag.shape)

        # Cross-Attention results
        tokens = torch.cat([g_rnaseq_bag, h_meth_bag], dim=0)
        print('tokens: ', tokens.shape)
        tokens = self.identity(tokens)
        print('tokens identity: ', tokens.shape)
        multimodal_embedding, self_attention_rnaseq, cross_attention_rnaseq, cross_attention_meth = self.cross_attender(x=tokens, mask=None, return_attention=True)
        print('multimodal_embedding: ', multimodal_embedding.shape)
        print('self_attention_rnaseq: ', self_attention_rnaseq)
        print('cross_attention_rnaseq: ', cross_attention_rnaseq)
        print('cross_attention_meth: ', cross_attention_meth)

        # Feedforward and Layer Normalization
        multimodal_embedding = self.feed_forward(multimodal_embedding)
        multimodal_embedding = self.layer_norm(multimodal_embedding)

        # Modality Specific Mean
        rnaseq_postSA_embedding = multimodal_embedding[:, :self.gene_groups, :]
        rnaseq_postSA_embedding = torch.mean(rnaseq_postSA_embedding, dim=1)
        meth_postSA_embed = multimodal_embedding[:, self.gene_groups:, :]
        meth_postSA_embed = torch.mean(meth_postSA_embed, dim=1)

        # Modalities Aggregation
        embedding = torch.cat([rnaseq_postSA_embedding, meth_postSA_embed], dim=1)

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

        print('hazards: ', hazards.shape)
        print('surv: ', surv.shape)

        attention_scores = {'self_attention_rnaseq': self_attention_rnaseq, 'cross_attention_rnaseq': cross_attention_rnaseq, 'cross_attention_meth': cross_attention_meth}

        return hazards, surv, Y, attention_scores

    def get_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
