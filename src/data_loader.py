import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os


class TabularDataset(Dataset):
    def __init__(
        self,
        data,
        categorical_features,
        numerical_features,
        boolean_features,
        categorical_encoders=None,
        numerical_scaler=None,
        is_train=True,
    ):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.boolean_features = boolean_features
        
        # 复制数据避免修改原数据
        self.data = data.copy()
        
        # 处理布尔特征
        self._process_boolean_features()
        
        # 编码类别特征
        if is_train:
            self.categorical_encoders = {}
            for col in categorical_features:
                le = LabelEncoder()
                self.data[col] = self.data[col].astype(str)
                self.data[col] = le.fit_transform(self.data[col])
                self.categorical_encoders[col] = le
        else:
            self.categorical_encoders = categorical_encoders
            for col in categorical_features:
                le = self.categorical_encoders[col]
                self.data[col] = self.data[col].astype(str)
                # 处理训练集中未出现的类别
                known_classes = set(le.classes_)
                self.data[col] = self.data[col].apply(
                    lambda x: x if x in known_classes else le.classes_[0]
                )
                self.data[col] = le.transform(self.data[col])
        
        # 标准化数值特征
        if is_train:
            self.numerical_scaler = StandardScaler()
            if numerical_features:
                self.data[numerical_features] = self.numerical_scaler.fit_transform(
                    self.data[numerical_features]
                )
        else:
            self.numerical_scaler = numerical_scaler
            if numerical_features:
                self.data[numerical_features] = self.numerical_scaler.transform(
                    self.data[numerical_features]
                )
        
        # 提取数据
        self.categorical_data = (
            self.data[categorical_features].values if categorical_features else None
        )
        self.numerical_data = (
            self.data[numerical_features].values.astype(np.float32)
            if numerical_features
            else None
        )
        self.boolean_data = (
            self.data[boolean_features].values if boolean_features else None
        )

    def _process_boolean_features(self):
        """处理布尔特征：'true'->0, 'false'->1, 'null'->2"""
        bool_map = {"true": 0, "false": 1, "null": 2}
        for col in self.boolean_features:
            self.data[col] = self.data[col].astype(str).str.lower()
            self.data[col] = self.data[col].apply(lambda x: bool_map.get(x, 2))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {}
        
        if self.categorical_data is not None:
            item["categorical"] = torch.tensor(self.categorical_data[idx], dtype=torch.long)
        else:
            item["categorical"] = torch.tensor([], dtype=torch.long)
            
        if self.numerical_data is not None:
            item["numerical"] = torch.tensor(self.numerical_data[idx], dtype=torch.float32)
        else:
            item["numerical"] = torch.tensor([], dtype=torch.float32)
            
        if self.boolean_data is not None:
            item["boolean"] = torch.tensor(self.boolean_data[idx], dtype=torch.long)
        else:
            item["boolean"] = torch.tensor([], dtype=torch.long)
            
        return item


class SAINTDataLoader:
    def __init__(self, config):
        self.config = config
        self.categorical_features = config["features"]["categorical_features"]
        self.numerical_features = config["features"]["numerical_features"]
        self.boolean_features = config["features"]["boolean_features"]
        
        self.categorical_encoders = None
        self.numerical_scaler = None

    def load_parquet(self, data_path):
        """加载 Parquet 文件"""
        return pd.read_parquet(data_path)

    def get_dataloaders(self, data_path, val_size=0.2, batch_size=256, num_workers=4):
        """获取训练和验证 DataLoader"""
        # 加载数据
        df = self.load_parquet(data_path)
        
        # 划分训练集和验证集
        if val_size > 0:
            val_size = int(len(df) * val_size)
            train_df = df.iloc[:-val_size].reset_index(drop=True)
            val_df = df.iloc[-val_size:].reset_index(drop=True)
        else:
            train_df = df
            val_df = None
        
        # 创建数据集
        train_dataset = TabularDataset(
            train_df,
            self.categorical_features,
            self.numerical_features,
            self.boolean_features,
            is_train=True,
        )
        
        # 保存编码器
        self.categorical_encoders = train_dataset.categorical_encoders
        self.numerical_scaler = train_dataset.numerical_scaler
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        
        val_loader = None
        if val_df is not None:
            val_dataset = TabularDataset(
                val_df,
                self.categorical_features,
                self.numerical_features,
                self.boolean_features,
                categorical_encoders=self.categorical_encoders,
                numerical_scaler=self.numerical_scaler,
                is_train=False,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
        
        return train_loader, val_loader

    def get_inference_dataloader(self, data_path, batch_size=256, num_workers=4):
        """获取推理 DataLoader"""
        df = self.load_parquet(data_path)
        
        dataset = TabularDataset(
            df,
            self.categorical_features,
            self.numerical_features,
            self.boolean_features,
            categorical_encoders=self.categorical_encoders,
            numerical_scaler=self.numerical_scaler,
            is_train=False,
        )
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        
        return loader

    def get_cardinalities(self):
        """获取类别特征的基数（类别数量）"""
        cardinalities = []
        for col in self.categorical_features:
            le = self.categorical_encoders[col]
            cardinalities.append(len(le.classes_))
        return cardinalities

    def save_preprocessors(self, save_path):
        """保存预处理器"""
        preprocessors = {
            "categorical_encoders": self.categorical_encoders,
            "numerical_scaler": self.numerical_scaler,
        }
        with open(save_path, "wb") as f:
            pickle.dump(preprocessors, f)

    def load_preprocessors(self, load_path):
        """加载预处理器"""
        with open(load_path, "rb") as f:
            preprocessors = pickle.load(f)
        self.categorical_encoders = preprocessors["categorical_encoders"]
        self.numerical_scaler = preprocessors["numerical_scaler"]
