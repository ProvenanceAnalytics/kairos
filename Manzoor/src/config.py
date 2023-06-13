import argparse


def parse_encoder(parser, arg_str=None):
    enc_parser = parser.add_argument_group()

    enc_parser.add_argument('--dataset', type=str,
                            help='Dataset')
    enc_parser.add_argument('--batch_size', type=int,
                            help='Training batch size')

    enc_parser.add_argument('--nei_size', type=int,
                            help='neighborhoods size')
    enc_parser.add_argument('--emb_size', type=int,
                            help='gnn embedding size')
    enc_parser.add_argument('--state_size', type=int,
                            help='node state size')
    enc_parser.add_argument('--dropout', type=float,
                            help='Dropout rate')
    enc_parser.add_argument('--opt', type=str,
                            help='optimizer')
    enc_parser.add_argument('--lr', type=float,
                            help='learning rate')
    enc_parser.add_argument('--model_path', type=str,
                            help='path to save/load model')

    enc_parser.add_argument('--db_name',
                            help='name of database')
    enc_parser.add_argument('--db_host',
                            help='postgres database host ip')
    enc_parser.add_argument('--db_user',
                            help='database user name')
    enc_parser.add_argument('--db_passwd',
                            help='database password')

    enc_parser.add_argument('--node_enc_size',
                            help='database password')

    enc_parser.set_defaults(dataset='manzoor',
                            batch_size=1024,
                            nei_size=20,
                            emb_size=100,
                            state_size=100,
                            dropout=0.0,
                            opt='adam',
                            lr=1e-5,
                            model_path="./model.pt",
                            db_name='manzoor_db',
                            db_host='10.26.52.10',
                            db_user='psql',
                            db_passwd='ssys821',
                            node_enc_size=8,
                            )
