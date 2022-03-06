import argparse

from Loader import Loader

def parse_args():
    '''
    python Run.py --epochs 100 --batch_size 256 --lr 0.01 --dropout_rate 0.5
    '''
    parser = argparse.ArgumentParser(description="GNN")
    parser.add_argument("--epochs", default=100)
    parser.add_argument("--batch_size", default=256)
    parser.add_argument("--lr", default=0.01)
    parser.add_argument("--dropout_rate", default=0.5)
    
    return parser.parse_args()

class Run:
    def __init__(self):
        loader = Loader()
        
