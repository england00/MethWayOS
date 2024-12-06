import torch
import torch.nn as nn


def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))


## MAIN
if __name__ == "__main__":
    x_omic = [torch.randn(dim) for dim in [100, 200, 300, 400, 500, 600]]
    print(x_omic)

    omic_sizes = [100, 200, 300, 400, 500, 600]
    model_size_omic = 'small'
    dropout = 0.25
    size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}

    ### Constructing Genomic SNN
    hidden = size_dict_omic[model_size_omic]
    sig_networks = []
    for input_dim in omic_sizes:
        fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
        sig_networks.append(nn.Sequential(*fc_omic))
    sig_networks = nn.ModuleList(sig_networks)

    print("*** 1. Bag-Level Representation (FC Processing) ***")
    h_omic = [sig_networks[idx].forward(sig_feat) for idx, sig_feat in
              enumerate(x_omic)]  ### each omic signature goes through it's own FC layer
    h_omic_bag = torch.stack(h_omic).unsqueeze(1)  ### omic embeddings are stacked (to be used in co-attention)
    print("Genomic Embeddings (G_bag before GCA):\n", h_omic_bag.shape)
    print()

