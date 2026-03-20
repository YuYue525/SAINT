import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import argparse
import os
from tqdm import tqdm
import numpy as np

from data_loader import SAINTDataLoader
from saint_model import SAINT, NTXentLoss


def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def augment_batch(batch):
    """对批次数据进行增强（用于对比学习）"""
    # 这里可以添加特征级别的增强
    # 比如对数值特征添加噪声，对类别特征随机掩码等
    # 简单版本：返回两个相同的批次（后期可以改进）
    return batch, batch


def train_epoch(model, train_loader, criterion, optimizer, device, temperature):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        batch1, batch2 = augment_batch(batch)
        
        # 将数据移到设备
        cat1 = batch1["categorical"].to(device) if batch1["categorical"].numel() > 0 else None
        num1 = batch1["numerical"].to(device) if batch1["numerical"].numel() > 0 else None
        bool1 = batch1["boolean"].to(device) if batch1["boolean"].numel() > 0 else None
        
        cat2 = batch2["categorical"].to(device) if batch2["categorical"].numel() > 0 else None
        num2 = batch2["numerical"].to(device) if batch2["numerical"].numel() > 0 else None
        bool2 = batch2["boolean"].to(device) if batch2["boolean"].numel() > 0 else None
        
        # 前向传播
        _, z1 = model(cat1, num1, bool1)
        _, z2 = model(cat2, num2, bool2)
        
        # 计算损失
        loss = criterion(z1, z2)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            batch1, batch2 = augment_batch(batch)
            
            # 将数据移到设备
            cat1 = batch1["categorical"].to(device) if batch1["categorical"].numel() > 0 else None
            num1 = batch1["numerical"].to(device) if batch1["numerical"].numel() > 0 else None
            bool1 = batch1["boolean"].to(device) if batch1["boolean"].numel() > 0 else None
            
            cat2 = batch2["categorical"].to(device) if batch2["categorical"].numel() > 0 else None
            num2 = batch2["numerical"].to(device) if batch2["numerical"].numel() > 0 else None
            bool2 = batch2["boolean"].to(device) if batch2["boolean"].numel() > 0 else None
            
            # 前向传播
            _, z1 = model(cat1, num1, bool1)
            _, z2 = model(cat2, num2, bool2)
            
            # 计算损失
            loss = criterion(z1, z2)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser(description="Train SAINT model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to parquet data")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设备配置
    device = config["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    print(f"Using device: {device}")
    
    # 准备数据
    data_loader = SAINTDataLoader(config)
    train_loader, val_loader = data_loader.get_dataloaders(
        data_path=args.data_path,
        val_size=config["data"]["val_size"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )
    
    # 保存预处理器
    data_loader.save_preprocessors(os.path.join(args.output_dir, "preprocessors.pkl"))
    
    # 获取类别特征基数
    cardinalities = data_loader.get_cardinalities()
    
    # 创建模型
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
    
    # 损失函数和优化器
    criterion = NTXentLoss(temperature=config["training"]["temperature"])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],
    )
    
    # 训练循环
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {config['training']['epochs']} epochs...")
    
    for epoch in range(config["training"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        
        # 训练
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            config["training"]["temperature"]
        )
        train_losses.append(train_loss)
        
        # 验证
        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        else:
            val_loss = train_loss
            print(f"Train Loss: {train_loss:.4f}")
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": config,
                    "cardinalities": cardinalities,
                },
                os.path.join(args.output_dir, "saint_model_best.pth"),
            )
            print(f"Saved best model with val_loss: {best_val_loss:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": config,
                    "cardinalities": cardinalities,
                },
                os.path.join(args.output_dir, f"saint_model_epoch_{epoch + 1}.pth"),
            )
    
    # 保存最终模型
    torch.save(
        {
            "epoch": config["training"]["epochs"] - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "cardinalities": cardinalities,
        },
        os.path.join(args.output_dir, "saint_model_final.pth"),
    )
    
    # 保存训练历史
    np.save(os.path.join(args.output_dir, "train_losses.npy"), np.array(train_losses))
    if val_losses:
        np.save(os.path.join(args.output_dir, "val_losses.npy"), np.array(val_losses))
    
    print("\nTraining completed!")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
