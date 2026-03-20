import pandas as pd
import numpy as np
import argparse
import os


def generate_sample_data(n_samples=10000, seed=42):
    """生成示例表格型数据"""
    np.random.seed(seed)
    
    data = {}
    
    # 类别型特征
    data["device_model"] = np.random.choice(
        ["iPhone 12", "iPhone 13", "iPhone 14", "Samsung Galaxy S21", "Samsung Galaxy S22", "Xiaomi 12"],
        size=n_samples
    )
    
    data["os_version"] = np.random.choice(
        ["iOS 15", "iOS 16", "iOS 17", "Android 11", "Android 12", "Android 13"],
        size=n_samples
    )
    
    data["country"] = np.random.choice(
        ["CN", "US", "JP", "UK", "DE", "FR", "IN"],
        size=n_samples
    )
    
    data["carrier"] = np.random.choice(
        ["China Mobile", "China Unicom", "China Telecom", "Verizon", "AT&T", "Docomo"],
        size=n_samples
    )
    
    # 数值型特征
    data["boot_time"] = np.random.exponential(scale=1000, size=n_samples)  # 开机时间（秒）
    data["battery_level"] = np.random.uniform(0, 100, size=n_samples)  # 电池电量
    data["screen_time"] = np.random.gamma(shape=2, scale=60, size=n_samples)  # 屏幕使用时间（分钟）
    data["app_usage_count"] = np.random.poisson(lam=20, size=n_samples)  # app使用次数
    data["network_latency"] = np.random.lognormal(mean=3, sigma=0.5, size=n_samples)  # 网络延迟
    
    # 布尔型特征
    data["is_jailbroken"] = np.random.choice(["true", "false", "null"], size=n_samples, p=[0.05, 0.9, 0.05])
    data["is_rooted"] = np.random.choice(["true", "false", "null"], size=n_samples, p=[0.03, 0.92, 0.05])
    data["has_vpn"] = np.random.choice(["true", "false", "null"], size=n_samples, p=[0.15, 0.8, 0.05])
    data["is_emulator"] = np.random.choice(["true", "false", "null"], size=n_samples, p=[0.02, 0.93, 0.05])
    
    # 创建 DataFrame
    df = pd.DataFrame(data)
    
    # 注入一些黑产模式（用于后续聚类分析）
    # 创建一些异常样本
    n_fraud = int(n_samples * 0.05)
    fraud_indices = np.random.choice(n_samples, size=n_fraud, replace=False)
    
    # 黑产特征：短开机时间 + 高app使用次数 + 越狱 + VPN
    df.loc[fraud_indices, "boot_time"] = np.random.exponential(scale=60, size=n_fraud)
    df.loc[fraud_indices, "app_usage_count"] = np.random.poisson(lam=100, size=n_fraud)
    df.loc[fraud_indices, "is_jailbroken"] = np.random.choice(["true", "false"], size=n_fraud, p=[0.8, 0.2])
    df.loc[fraud_indices, "has_vpn"] = np.random.choice(["true", "false"], size=n_fraud, p=[0.9, 0.1])
    df.loc[fraud_indices, "is_emulator"] = np.random.choice(["true", "false"], size=n_fraud, p=[0.7, 0.3])
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate sample tabular data")
    parser.add_argument("--output_path", type=str, default="data/sample_data.parquet", 
                        help="Path to save sample data")
    parser.add_argument("--n_samples", type=int, default=10000, 
                        help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {args.n_samples} samples...")
    df = generate_sample_data(n_samples=args.n_samples, seed=args.seed)
    
    # 保存为 parquet
    df.to_parquet(args.output_path, index=False)
    print(f"Sample data saved to: {args.output_path}")
    
    # 打印数据预览
    print("\nData preview:")
    print(df.head())
    print(f"\nData shape: {df.shape}")
    
    # 打印统计信息
    print("\nCategorical features:")
    for col in ["device_model", "os_version", "country", "carrier"]:
        print(f"\n{col}:")
        print(df[col].value_counts().head())
    
    print("\nNumerical features:")
    numerical_cols = ["boot_time", "battery_level", "screen_time", "app_usage_count", "network_latency"]
    print(df[numerical_cols].describe())
    
    print("\nBoolean features:")
    for col in ["is_jailbroken", "is_rooted", "has_vpn", "is_emulator"]:
        print(f"\n{col}:")
        print(df[col].value_counts())


if __name__ == "__main__":
    main()
