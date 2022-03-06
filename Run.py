import argparse

from Loader import Loader
from Model import *

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
        self.loader = Loader()
        
    def run_experiment(model, x_train, y_train, args):
        # Compile the model.
        model.compile(
            optimizer=keras.optimizers.Adam(args.lr),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
        )
        # Create an early stopping callback.
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_acc", patience=50, restore_best_weights=True
        )
        # Fit the model.
        history = model.fit(
            x=x_train,
            y=y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=0.15,
            callbacks=[early_stopping],
        )

        return history
    
    def run(self,args):
        citations, papers, x_train, x_test, y_train, y_test, train_data, test_data = self.loader.load_dataset()
        feature_names = self.loader.feature_names
        num_classes = self.loader.num_classes
        graph_info = self.loader.graph_info
        
        self.gnn_model = GNNNodeClassifier(
            graph_info=graph_info,
            num_classes=num_classes,
            hidden_units=[32,32],
            dropout_rate=args.dropout_rate,
            name="gnn_model",
        )

        # print("GNN output shape:", gnn_model([1, 10, 100]))
        # gnn_model.summary()
        
        x_train = train_data.paper_id.to_numpy()
        history = self.run_experiment(self.gnn_model, x_train, y_train)
        
    def test(self):
        citations, papers, x_train, x_test, y_train, y_test, train_data, test_data = self.loader.load_dataset()
        _, test_accuracy = self.gnn_model.evaluate(x=x_test, y=y_test, verbose=0)
        print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")

if __name__=="__main__":
    
    args=parse_args()
    run = Run()
    gnn_moel = run.run(args)
    
    