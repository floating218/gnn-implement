import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Loader:
    def __init__(self):
        pass
    
    def load_dataset(self):    
        
        '''
        citations: [target논문인덱스, source논문인덱스]
        papers: [논문인덱스, 1424개 단어 포함 여부, 주제(subject)]
        train_data: papers 데이터 중 50% 샘플링
        test_data: papers 데이터 중 50% 샘플링
        x_train: train_data 중, 논문인덱스와 subject를 제외한 피쳐
        y_train: train_data 중 subject에 해당하는 레이블
        '''
        
        zip_file = keras.utils.get_file(
            fname="cora.tgz",
            origin="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
            extract=True,
        )
        data_dir = os.path.join(os.path.dirname(zip_file), "cora")
        
        citations = pd.read_csv(
            os.path.join(data_dir, "cora.cites"),
            sep="\t",
            header=None,
            names=["target", "source"],
        )
        
        column_names = ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
        papers = pd.read_csv(
            os.path.join(data_dir, "cora.content"), 
            sep="\t", 
            header=None, 
            names=column_names,
        )
        
        class_values = sorted(papers["subject"].unique())
        class_idx = {name: id for id, name in enumerate(class_values)}
        paper_idx = {name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))}

        papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
        citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
        citations["target"] = citations["target"].apply(lambda name: paper_idx[name])
        papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])
        
        train_data, test_data = [], []

        for _, group_data in papers.groupby("subject"):
            # Select around 50% of the dataset for training.
            random_selection = np.random.rand(len(group_data.index)) <= 0.5
            train_data.append(group_data[random_selection])
            test_data.append(group_data[~random_selection])

        train_data = pd.concat(train_data).sample(frac=1)
        test_data = pd.concat(test_data).sample(frac=1)

        print("citations data shape:", citations.shape)
        print("papers data shape:", papers.shape)
        print("Train data shape:", train_data.shape)
        print("Test data shape:", test_data.shape)
        
        feature_names = set(papers.columns) - {"paper_id", "subject"}
        num_features = len(feature_names)
        num_classes = len(class_idx)

        # Create train and test features as a numpy array.
        x_train = train_data[feature_names].to_numpy()
        x_test = test_data[feature_names].to_numpy()
        
        # Create train and test targets as a numpy array.
        y_train = train_data["subject"]
        y_test = test_data["subject"]
        
        # Create an edges array (sparse adjacency matrix) of shape [2, num_edges].
        edges = citations[["source", "target"]].to_numpy().T
        # Create an edge weights array of ones.
        edge_weights = tf.ones(shape=edges.shape[1])
        # Create a node features array of shape [num_nodes, num_features].
        node_features = tf.cast(
            papers.sort_values("paper_id")[feature_names].to_numpy(), dtype=tf.dtypes.float32
        )
        # Create graph info tuple with node_features, edges, and edge_weights.
        graph_info = (node_features, edges, edge_weights)

        print("Edges shape:", edges.shape)
        print("Nodes shape:", node_features.shape)
        
        self.feature_names = feature_names #단어 피쳐 컬럼 이름
        self.num_features = num_features #단어 피쳐 개수
        self.num_classes = num_classes #주제subject의 가짓수
        self.graph_info = graph_info
        self.node_features = node_features #논문 인덱스 순으로 정리된 상태의 feature 매트릭스
        self.edges = edges
        self.class_values = class_values
        
        return citations, papers, x_train, x_test, y_train, y_test, train_data, test_data
    
    