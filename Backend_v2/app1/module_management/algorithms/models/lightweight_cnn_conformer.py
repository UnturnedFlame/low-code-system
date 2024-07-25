import torch
from app1.module_management.algorithms.models.wenet.conformer_test.encoder_cat_ffn import ConformerEncoder
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from speechbrain.lobes.models.ECAPA_TDNN import BatchNorm1d


class Conformer(torch.nn.Module):
    def __init__(self, n_mels=80, num_blocks=6, output_size=256, embedding_dim=192, input_layer="conv2d2",
                 pos_enc_layer_type="rel_pos"):

        macaron_style = True
        use_mfa = True

        super(Conformer, self).__init__()
        print("input_layer: {}".format(input_layer))
        print("pos_enc_layer_type: {}".format(pos_enc_layer_type))
        self.conformer = ConformerEncoder(input_size=n_mels, num_blocks=num_blocks,
                                          output_size=output_size, input_layer=input_layer,
                                          pos_enc_layer_type=pos_enc_layer_type, macaron_style=macaron_style,
                                          use_mfa=use_mfa)
        if use_mfa:
            self.pooling = AttentiveStatisticsPooling(output_size * num_blocks)  # ASP初始化为通道数
            self.bn = BatchNorm1d(input_size=output_size * num_blocks * 2)
            self.fc = torch.nn.Linear(output_size * num_blocks * 2, embedding_dim)

        else:
            self.pooling = AttentiveStatisticsPooling(output_size)
            self.bn = BatchNorm1d(input_size=output_size * 2)
            self.fc = torch.nn.Linear(output_size * 2, embedding_dim)

    def forward(self, feat):  # (B,C,F,T)
        feat = feat.squeeze(1).permute(0, 2, 1)  # (B,T,F)
        lens = torch.ones(feat.shape[0]).to(feat.device)
        lens = torch.round(lens * feat.shape[1]).int()
        x, masks = self.conformer(feat, lens)
        x = x.permute(0, 2, 1)
        x = self.pooling(x)  # ASP
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = x.squeeze(1)
        return x


def cnn_conformer(n_mels=80, num_blocks=6, output_size=256,
                  embedding_dim=192, input_layer="conv2d", pos_enc_layer_type="rel_pos"):
    model = Conformer(n_mels=n_mels, num_blocks=num_blocks, output_size=output_size,
                      embedding_dim=embedding_dim, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type)
    return model
