import torch
import torch.nn as nn
import torch.nn.functional as F
from src.mcat.original_modules.blocks import AttentionNetGated


# MCAT link
# https://github.com/mahmoodlab/MCAT/blob/master/Model%20Computation%20%2B%20Complexity%20Overview.ipynb


''' SingleModalTransformer Definition '''
class SingleModalTransformer(nn.Module):
    def __init__(self, encoder_sizes: [], model_size: str = 'medium', n_classes: int = 4, dropout: float = 0.25):
        super(SingleModalTransformer, self).__init__()
        self.n_classes = n_classes
        if model_size == 'small':
            self.model_sizes = [128, 128]
        elif model_size == 'medium':
            self.model_sizes = [256, 256]
        elif model_size == 'big':
            self.model_sizes = [512, 512]

        # Input Encoder
        encoders = []
        for rnaseq_size in encoder_sizes:
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
            encoders.append(fc)
        self.E = nn.ModuleList(encoders)

        # Transformer (T)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_sizes[1], nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Global Attention Pooling (rho)
        self.attention_head = AttentionNetGated(n_classes=1, input_dim=self.model_sizes[1], hidden_dim=self.model_sizes[1], dropout_p=dropout)
        self.rho = nn.Sequential(*[nn.Linear(self.model_sizes[1], self.model_sizes[1]), nn.ReLU(), nn.Dropout(dropout)])

        # Classifier
        self.classifier = nn.Linear(self.model_sizes[1], n_classes)

    def forward(self, data):
        # N Gene Expression Fully connected layers
        E = [self.E[index].forward(rnaseq.type(torch.float32)) for index, rnaseq in enumerate(data)]
        # E_bag: (Nxd_k)
        E_bag = torch.stack(E).squeeze(1)

        # Set-Based MIL Transformers
        # Attention is permutation-equivariant, so dimensions are the same (Nxd_k)
        data_trans = self.transformer(E_bag)

        # Global Attention Pooling
        A_data, h_data = self.attention_head(data_trans.squeeze(1))
        A_data = torch.transpose(A_data, 1, 0)
        h_data = torch.mm(F.softmax(A_data, dim=1), h_data)
        # h_omic: final omics embeddings (dk)
        h_data = self.rho(h_data).squeeze()

        # Survival Layer
        # logits: classifier output
        # size   --> (1, n_classes)
        # domain --> R
        logits = self.classifier(h_data).unsqueeze(0)
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

        attention_scores = {'omic': A_data}

        return hazards, surv, Y, attention_scores

    def get_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
