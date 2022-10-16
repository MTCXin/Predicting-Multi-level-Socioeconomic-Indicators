import argparse


class Parser():

    def __init__(self, description):
        '''
           arguments parser
        '''
        self.parser = argparse.ArgumentParser(description=description)
        self.args = None
        self._parse()

    def _parse(self):
        # dataset
        self.parser.add_argument(
            '--dataset', type=str, default="population",
            choices=['population', 'economic_act', 'resident_cons'],
            help='name of dataset (default: population)')
        self.parser.add_argument(
            '--batch_size', type=int, default=128,
            help='batch size for training and validation (default: 32)')
        self.parser.add_argument(
            '--fold_idx', type=int, default=0,
            help='the index(<10) of fold in 10-fold validation.')
        self.parser.add_argument(
            '--filename', type=str, default="./res/GIN-fixedH0att-firm-20.txt",
            help='output file')

        # device
        self.parser.add_argument(
            '--disable-cuda', action='store_true',
            help='Disable CUDA')
        self.parser.add_argument(
            '--device', type=int, default=0,
            help='which gpu device to use (default: 0)')

        # net
        self.parser.add_argument(
            '--num_layers', type=int, default=5,
            help='number of layers (default: 5)')
        self.parser.add_argument(
            '--num_mlp_layers', type=int, default=2,
            help='number of MLP layers(default: 2). 1 means linear model.')
        self.parser.add_argument(
            '--hidden_dim', type=int, default=128,
            help='number of hidden units (default: 64)')

        # graph
        self.parser.add_argument(
            '--graph_pooling_type', type=str,
            default="sum", choices=["sum", "mean", "max"],
            help='type of graph pooling: sum, mean or max')
        self.parser.add_argument(
            '--neighbor_pooling_type', type=str,
            default="sum", choices=["sum", "mean", "max"],
            help='type of neighboring pooling: sum, mean or max')
        self.parser.add_argument(
            '--learn_eps', action="store_true",
            help='learn the epsilon weighting')

        # learning
        self.parser.add_argument(
            '--seed', type=int, default=0,
            help='random seed (default: 0)')
        self.parser.add_argument(
            '--epochs', type=int, default=12000,
            help='number of epochs to train (default: 400)')
        self.parser.add_argument(
            '--lr', type=float, default=0.005,
            help='learning rate (default: 0.01)')
        self.parser.add_argument(
            '--final_dropout', type=float, default=0,
            help='final layer dropout (default: 0.5)')
        # 
        self.parser.add_argument(
            '--dim_nfeats', type=int, default=128,
            help='dimention of data feature (default: 128)')
        self.parser.add_argument(
            '--gclasses', type=int, default=4,
            help='final classification number(default: 4)')
        self.parser.add_argument(
            '--TRAIN_SIZE', type=float, default=0.75,
            help='partition of data for training (default: 0.75)')
        self.parser.add_argument(
            '--TEST_SIZE', type=float, default=0.25,
            help='partition of data for testing (default: 0.25)')
        self.parser.add_argument(
            '--ABORT_ZERO', type=float, default=0.5,
            help='original data includes some region with 0 value indicator (default: 0.5)')
        # done
        self.args = self.parser.parse_args()