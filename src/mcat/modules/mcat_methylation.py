import torch
import torch.nn as nn
import torch.nn.functional as F
from src.mcat.modules.blocks import AttentionNetGated
from src.mcat.modules.fusion import BilinearFusion, ConcatFusion, GatedConcatFusion


# MCAT link
# https://github.com/mahmoodlab/MCAT/blob/master/Model%20Computation%20%2B%20Complexity%20Overview.ipynb


''' MultimodalCoAttentionTransformer Definition '''
class MultimodalCoAttentionTransformer(nn.Module):
    def __init__(self, meth_sizes: [], rnaseq_sizes: [], model_size: str = 'medium', n_classes: int = 4, dropout: float = 0.25, fusion: str = 'concat', device: str = 'cpu'):
        super(MultimodalCoAttentionTransformer, self).__init__()
        self.n_classes = n_classes
        if model_size == 'small':
            self.model_sizes = [128, 128]
        elif model_size == 'medium':
            self.model_sizes = [256, 256]
        elif model_size == 'big':
            self.model_sizes = [512, 512]

        # Methylation Encoder
        meth_encoders = []
        for meth_size in meth_sizes:
            fc = nn.Sequential(
                nn.Sequential(
                    nn.Linear(meth_size, self.model_sizes[0]),
                    nn.ELU(),
                    nn.AlphaDropout(p=dropout, inplace=False)),
                nn.Sequential(
                    nn.Linear(self.model_sizes[0], self.model_sizes[1]),
                    nn.ELU(),
                    nn.AlphaDropout(p=dropout, inplace=False))
            )
            meth_encoders.append(fc)
        self.H = nn.ModuleList(meth_encoders)

        # Gene Expression Encoder
        rnaseq_encoders = []
        for rnaseq_size in rnaseq_sizes:
            fc = nn.Sequential(
                nn.Sequential(
                    nn.Linear(rnaseq_size, self.model_sizes[0]),
                    nn.ELU(),
                    nn.AlphaDropout(p=dropout, inplace=False)),
                nn.Sequential(
                    nn.Linear(self.model_sizes[0], self.model_sizes[1]),
                    nn.ELU(),
                    nn.AlphaDropout(p=dropout, inplace=False))
            )
            rnaseq_encoders.append(fc)
        self.G = nn.ModuleList(rnaseq_encoders)

        # Genomic-Guided Co-Attention
        self.co_attention = nn.MultiheadAttention(embed_dim=self.model_sizes[1], num_heads=1)

        # Path Transformer (T_H)
        path_encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_sizes[1], nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)

        # Methylation Global Attention Pooling (rho_H)
        self.path_attention_head = AttentionNetGated(n_classes=1, input_dim=self.model_sizes[1], hidden_dim=self.model_sizes[1])
        self.path_rho = nn.Sequential(*[nn.Linear(self.model_sizes[1], self.model_sizes[1]), nn.ReLU(), nn.Dropout(dropout)])

        # Gene Expression Transformer (T_G)
        rnaseq_encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_sizes[1], nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.rnaseq_transformer = nn.TransformerEncoder(rnaseq_encoder_layer, num_layers=2)

        # Genomic Global Attention Pooling (rho_G)
        self.rnaseq_attention_head = AttentionNetGated(n_classes=1, input_dim=self.model_sizes[1], hidden_dim=self.model_sizes[1])
        self.rnaseq_rho = nn.Sequential(*[nn.Linear(self.model_sizes[1], self.model_sizes[1]), nn.ReLU(), nn.Dropout(dropout)])

        # Fusion Layer
        self.fusion = fusion
        if self.fusion == 'concat':
            self.fusion_layer = ConcatFusion(dims=[self.model_sizes[1], self.model_sizes[1]],
                                             hidden_size=self.model_sizes[1], output_size=self.model_sizes[1]).to(device=device)
        elif self.fusion == 'bilinear':
            self.fusion_layer = BilinearFusion(dim1=self.model_sizes[1], dim2=self.model_sizes[1], output_size=self.model_sizes[1])
        elif self.fusion == 'gated_concat':
            self.fusion_layer = GatedConcatFusion(dims=[self.model_sizes[1], self.model_sizes[1]],
                                                  hidden_size=self.model_sizes[1], output_size=self.model_sizes[1]).to(device=device)
        else:
            raise RuntimeError(f'Fusion mechanism {self.fusion} not implemented')

        # Classifier
        self.classifier = nn.Linear(self.model_sizes[1], n_classes)

    def forward(self, islands, genes, inference: bool = False):
        # M Methylation Fully connected layer
        H_meth = [self.H[index].forward(island.type(torch.float32)) for index, island in enumerate(islands)]
        # H_bag: (Mxd_k)
        H_bag = torch.stack(H_meth).squeeze(1)

        # N Gene Expression Fully connected layers
        G_rnaseq = [self.G[index].forward(rnaseq.type(torch.float32)) for index, rnaseq in enumerate(genes)]
        # G_bag: (Nxd_k)
        G_bag = torch.stack(G_rnaseq).squeeze(1)

        # Co-Attention results
        # H_coattn: Genomic-Guided and Methylation-level Embeddings (Nxd_k)
        # A_coattn: Co-Attention Matrix (NxM)
        H_coattn, A_coattn = self.co_attention(query=G_bag, key=H_bag, value=H_bag, need_weights=inference)

        # Set-Based MIL Transformers
        # Attention is permutation-equivariant, so dimensions are the same (Nxd_k)
        path_trans = self.path_transformer(H_coattn)
        omic_trans = self.rnaseq_transformer(G_bag)

        # Global Attention Pooling
        A_path, h_path = self.path_attention_head(path_trans.squeeze(1))
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        # h_path: final Methylation embeddings (dk)
        h_path = self.path_rho(h_path).squeeze()

        A_omic, h_omic = self.rnaseq_attention_head(omic_trans.squeeze(1))
        A_omic = torch.transpose(A_omic, 1, 0)
        h_omic = torch.mm(F.softmax(A_omic, dim=1), h_omic)
        # h_omic: final omics embeddings (dk)
        h_omic = self.rnaseq_rho(h_omic).squeeze()

        # Fusion Layer
        # h: final representation (dk)
        h = self.fusion_layer(h_path, h_omic)

        # Survival Layer

        # logits: classifier output
        # size   --> (1, 4)
        # domain --> R
        logits = self.classifier(h).unsqueeze(0)
        # hazards: probability of patient death in interval j
        # size   --> (1, 4)
        # domain --> [0, 1]
        hazards = torch.sigmoid(logits)
        # survs: probability of patient survival after time t
        # size   --> (1, 4)
        # domain --> [0, 1]
        survs = torch.cumprod(1 - hazards, dim=1)
        # Y: predicted probability distribution
        # size   --> (1, 4)
        # domain --> [0, 1] (probability distribution)
        Y = F.softmax(logits, dim=1)

        attention_scores = {'coattn': A_coattn, 'path': A_path, 'omic': A_omic}

        return hazards, survs, Y, attention_scores

    def get_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
