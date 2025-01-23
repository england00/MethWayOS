import torch
import torch.nn as nn
import torch.nn.functional as F
from src.mcat.original_modules.blocks import AttentionNetGated


# MCAT link
# https://github.com/mahmoodlab/MCAT/blob/master/Model%20Computation%20%2B%20Complexity%20Overview.ipynb


''' ClassifierMethylation Definition '''
class ClassifierMethylation(nn.Module):
    def __init__(self, meth_sizes: [], model_size: str = 'medium', n_classes: int = 4, dropout: float = 0.25):
        super(ClassifierMethylation, self).__init__()
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

        # Path Transformer (T_H)
        path_encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_sizes[1], nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.meth_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)

        # Methylation Global Attention Pooling (rho_H)
        self.meth_attention_head = AttentionNetGated(n_classes=1, input_dim=self.model_sizes[1], hidden_dim=self.model_sizes[1])
        self.meth_rho = nn.Sequential(*[nn.Linear(self.model_sizes[1], self.model_sizes[1]), nn.ReLU(), nn.Dropout(dropout)])

        # Classifier
        self.classifier = nn.Linear(self.model_sizes[1], n_classes)

    def forward(self, islands):
        # M Methylation Fully connected layer
        H_meth = [self.H[index].forward(island.type(torch.float32)) for index, island in enumerate(islands)]
        # H_bag: (Mxd_k)
        H_bag = torch.stack(H_meth).squeeze(1)

        # Set-Based MIL Transformers
        # Attention is permutation-equivariant, so dimensions are the same (Nxd_k)
        meth_trans = self.meth_transformer(H_bag)

        # Global Attention Pooling
        A_meth, h_meth = self.meth_attention_head(meth_trans.squeeze(1))
        A_meth = torch.transpose(A_meth, 1, 0)
        h_meth = torch.mm(F.softmax(A_meth, dim=1), h_meth)
        # h_meth: final Methylation embeddings (dk)
        h_meth = self.meth_rho(h_meth).squeeze()

        # Survival Layer
        # logits: classifier output
        # size   --> (1, 4)
        # domain --> R
        logits = self.classifier(h_meth).unsqueeze(0)
        # hazards: probability of patient death in interval j
        # size   --> (1, 4)
        # domain --> [0, 1]
        hazards = torch.sigmoid(logits)
        # survs: probability of patient survival after time t
        # size   --> (1, 4)
        # domain --> [0, 1]
        surv = torch.cumprod(1 - hazards, dim=1)
        # Y: predicted probability distribution
        # size   --> (1, 4)
        # domain --> [0, 1] (probability distribution)
        Y = F.softmax(logits, dim=1)

        attention_scores = {'path': A_meth}

        return hazards, surv, Y, attention_scores

    def get_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
