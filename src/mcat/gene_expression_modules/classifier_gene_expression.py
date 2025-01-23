import torch
import torch.nn as nn
import torch.nn.functional as F
from src.mcat.original_modules.blocks import AttentionNetGated


# MCAT link
# https://github.com/mahmoodlab/MCAT/blob/master/Model%20Computation%20%2B%20Complexity%20Overview.ipynb


''' ClassifierGeneExpression Definition '''
class ClassifierGeneExpression(nn.Module):
    def __init__(self, rnaseq_sizes: [], model_size: str = 'medium', n_classes: int = 4, dropout: float = 0.25):
        super(ClassifierGeneExpression, self).__init__()
        self.n_classes = n_classes
        if model_size == 'small':
            self.model_sizes = [128, 128]
        elif model_size == 'medium':
            self.model_sizes = [256, 256]
        elif model_size == 'big':
            self.model_sizes = [512, 512]

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

        # Gene Expression Transformer (T_G)
        rnaseq_encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_sizes[1], nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.rnaseq_transformer = nn.TransformerEncoder(rnaseq_encoder_layer, num_layers=2)

        # Genomic Global Attention Pooling (rho_G)
        self.rnaseq_attention_head = AttentionNetGated(n_classes=1, input_dim=self.model_sizes[1], hidden_dim=self.model_sizes[1])
        self.rnaseq_rho = nn.Sequential(*[nn.Linear(self.model_sizes[1], self.model_sizes[1]), nn.ReLU(), nn.Dropout(dropout)])

        # Classifier
        self.classifier = nn.Linear(self.model_sizes[1], n_classes)

    def forward(self, genes):
        # N Gene Expression Fully connected layers
        G_rnaseq = [self.G[index].forward(rnaseq.type(torch.float32)) for index, rnaseq in enumerate(genes)]
        # G_bag: (Nxd_k)
        G_bag = torch.stack(G_rnaseq).squeeze(1)

        # Set-Based MIL Transformers
        # Attention is permutation-equivariant, so dimensions are the same (Nxd_k)
        rnaseq_trans = self.rnaseq_transformer(G_bag)

        # Global Attention Pooling
        A_rnaseq, h_rnaseq = self.rnaseq_attention_head(rnaseq_trans.squeeze(1))
        A_rnaseq = torch.transpose(A_rnaseq, 1, 0)
        h_rnaseq = torch.mm(F.softmax(A_rnaseq, dim=1), h_rnaseq)
        # h_omic: final omics embeddings (dk)
        h_rnaseq = self.rnaseq_rho(h_rnaseq).squeeze()

        # Survival Layer
        # logits: classifier output
        # size   --> (1, n_classes)
        # domain --> R
        logits = self.classifier(h_rnaseq).unsqueeze(0)
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

        attention_scores = {'omic': A_rnaseq}

        return hazards, surv, Y, attention_scores

    def get_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
