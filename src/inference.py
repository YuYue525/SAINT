import torch
import pandas as pd
import numpy as np
import argparse
import os
import yaml

from data_loader import SAINTDataLoader
from saint_model import SAINT


def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    cardinalities = checkpoint["cardinalities"]
    
    model = SAINT(
        num_categorical_features=len(config["features"]["categorical_features"]),
        num_numerical_features=len(config["features"]["numerical_features"]),
        num_boolean_features=len(config["features"]["boolean_features"]),
        cardinalities=cardinalities,
        embedding_dim=config["model"]["embedding_dim"],
        depth=config["model"]["depth"],
        heads=config["model"]["heads"],
        mlp_dim=config["model"]["mlp_dim"],
        dropout=config["model"]["dropout"],
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, config


def generate_embeddings(model, data_loader, device):
    """生成 embeddings"""
    all_embeddings = []
    
    with torch.no_grad():
        for batch in data_loader:
            # 将数据移到设备
            cat = batch["categorical"].to(device) if batch["categorical"].numel() > 0 else None
            num = batch["numerical"].to(device) if batch["numerical"].numel() > 0 else None
            bool_ = batch["boolean"].to(device) if batch["boolean"].numel() > 0 else None
            
            # 前向传播获取 embeddings
            embeddings = model(cat, num, bool_, return_embeddings=True)
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings with SAINT model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to parquet data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--preprocessors_path", type=str, default=None, 
                        help="Path to preprocessors (default: same dir as model)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save embeddings")
    parser.add_argument("--config", type=str, default=None, 
                        help="Path to config file (if not in model checkpoint)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    
    # 加载模型和配置
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    print("Loading model...")
    model, config = load_model(args.model_path, device)
    
    # 如果提供了单独的 config 文件，使用它
    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    
    # 准备数据加载器
    print("Preparing data loader...")
    data_loader = SAINTDataLoader(config)
    
    # 加载预处理器
    if args.preprocessors_path is None:
        # 默认从模型同目录加载
        preprocessors_path = os.path.join(os.path.dirname(args.model_path), "preprocessors.pkl")
    else:
        preprocessors_path = args.preprocessors_path
    
    if os.path.exists(preprocessors_path):
        data_loader.load_preprocessors(preprocessors_path)
    else:
        raise FileNotFoundError(f"Preprocessors not found at {preprocessors_path}")
    
    # 获取推理 DataLoader
    inference_loader = data_loader.get_inference_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
    )
    
    # 生成 embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(model, inference_loader, device)
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # 保存 embeddings
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存为 parquet 格式
    embeddings_df = pd.DataFrame(
        embeddings,
        columns=[f"emb_{i}" for i in range(embeddings.shape[1])]
    )
    embeddings_df.to_parquet(args.output_path, index=False)
    
    print(f"Embeddings saved to: {args.output_path}")
    
    # 也可以选择保存为 numpy 格式
    npy_path = args.output_path.replace(".parquet", ".npy")
    np.save(npy_path, embeddings)
    print(f"Embeddings also saved as numpy to: {npy_path}")


if __name__ == "__main__":
    main()
