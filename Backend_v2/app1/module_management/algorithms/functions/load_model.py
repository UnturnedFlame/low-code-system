# import torch
# from pytorch_lightning import LightningModule, Trainer
# from argparse import ArgumentParser
# from app1.module_management.algorithms.functions.utils import MelSpectrogram, FbankAug
# from app1.module_management.algorithms.models.loss import amsoftmax
# import numpy as np
#
#
# class Task(LightningModule):
#     def __init__(
#             self,
#             learning_rate: float = 0.2,
#             weight_decay: float = 1.5e-6,
#             batch_size: int = 32,
#             num_workers: int = 10,
#             max_epochs: int = 1000,
#             **kwargs
#     ):
#         super().__init__()
#         self.save_hyperparameters()
#         self.mel_trans = MelSpectrogram(sample_rate=48000)
#         self.specaug = FbankAug()
#
#         from app1.module_management.algorithms.models.lightweight_cnn_conformer import cnn_conformer
#
#         if self.hparams.encoder_name == "lightweight_cnn_conformer":
#             print("num_blocks is {}".format(self.hparams.num_blocks))
#             self.encoder = cnn_conformer(embedding_dim=self.hparams.embedding_dim,
#                                          num_blocks=self.hparams.num_blocks, input_layer=self.hparams.input_layer,
#                                          pos_enc_layer_type=self.hparams.pos_enc_layer_type)
#
#         else:
#             raise ValueError("encoder name error")
#
#         self.loss_fun = amsoftmax(embedding_dim=self.hparams.embedding_dim, num_classes=7205)
#
#     def forward(self, x):
#         feature = self.mel_trans(x)
#         embedding = self.encoder(feature)
#         return embedding
#
#     @staticmethod
#     def add_model_specific_args(parent_parser):
#         parser = ArgumentParser(parents=[parent_parser], add_help=False)
#         (args, _) = parser.parse_known_args()
#
#         parser.add_argument("--num_workers", default=6, type=int)
#
#         parser.add_argument("--embedding_dim", default=192, type=int)
#
#         parser.add_argument("--num_blocks", type=int, default=5)  # 5
#
#         parser.add_argument("--input_layer", type=str, default="conv2d2")
#         parser.add_argument("--pos_enc_layer_type", type=str, default="rel_pos")
#
#         parser.add_argument("--checkpoint_path", type=str,
#                             default="app1/module_management/algorithms/models/checkpoints/lightweight_cnn_conformer"
#                                     "/augepoch=22_cosine_eer=0"
#                                     ".65_as_norm(0.61).ckpt")
#         parser.add_argument("--encoder_name", type=str, default="lightweight_cnn_conformer")
#
#         return parser
#
#
# def load_model_with_pytorch_lightning():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # Loading model
#     parser = ArgumentParser()
#     # trainer args
#     parser = Trainer.add_argparse_args(parser)
#     parser = Task.add_model_specific_args(parser)
#     args = parser.parse_args([])
#     lightning_model = Task(**args.__dict__)
#     lightning_model.eval()
#     state_dict = torch.load(args.checkpoint_path, map_location=device)["state_dict"]
#     lightning_model.load_state_dict(state_dict)
#     lightning_model.to(device)
#     print("load weight from {}".format(args.checkpoint_path))
#     # 预热声纹模型 12 22 32 40 60
#     lightning_model(torch.FloatTensor(np.zeros((1, 10000))).to(device))
#
#     # 启动Elasticsearch环境
#     # es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
#     # threshold = 0.75
#
#     return lightning_model
#
#
# if __name__ == "__main__":
#     model = load_model_with_pytorch_lightning()
