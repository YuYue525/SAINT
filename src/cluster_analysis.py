import pandas as pd
import numpy as np
import argparse
import os
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_embeddings(embeddings_path):
    """加载 embeddings"""
    if embeddings_path.endswith(".parquet"):
        df = pd.read_parquet(embeddings_path)
        embeddings = df.values
    elif embeddings_path.endswith(".npy"):
        embeddings = np.load(embeddings_path)
    else:
        raise ValueError("Unsupported file format. Use .parquet or .npy")
    return embeddings


def cluster_kmeans(embeddings, n_clusters=8, random_state=42):
    """使用 K-Means 聚类"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # 计算评估指标
    silhouette = silhouette_score(embeddings, labels)
    calinski = calinski_harabasz_score(embeddings, labels)
    
    print(f"K-Means Clustering (n_clusters={n_clusters}):")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Calinski-Harabasz Score: {calinski:.4f}")
    
    return labels, kmeans


def cluster_dbscan(embeddings, eps=0.5, min_samples=5):
    """使用 DBSCAN 聚类"""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(embeddings)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples}):")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Number of noise points: {n_noise}")
    
    if n_clusters > 1:
        non_noise_mask = labels != -1
        if np.sum(non_noise_mask) > 0:
            silhouette = silhouette_score(embeddings[non_noise_mask], labels[non_noise_mask])
            print(f"  Silhouette Score (excluding noise): {silhouette:.4f}")
    
    return labels, dbscan


def visualize_embeddings(embeddings, labels, output_dir, method="tsne"):
    """可视化 embeddings"""
    print(f"Generating {method.upper()} visualization...")
    
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == "pca":
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # 创建可视化
    plt.figure(figsize=(12, 10))
    
    # 绘制散点图
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=labels, 
        cmap="tab10", 
        alpha=0.6,
        s=50
    )
    
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"SAINT Embeddings Visualization ({method.upper()})", fontsize=16)
    plt.xlabel(f"{method.upper()} 1", fontsize=12)
    plt.ylabel(f"{method.upper()} 2", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    output_path = os.path.join(output_dir, f"embeddings_{method}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Visualization saved to: {output_path}")
    
    return embeddings_2d


def analyze_clusters(original_data_path, labels, output_dir):
    """分析聚类结果"""
    print("Analyzing clusters...")
    
    # 加载原始数据
    df = pd.read_parquet(original_data_path)
    df["cluster"] = labels
    
    # 保存带聚类标签的数据
    output_path = os.path.join(output_dir, "data_with_clusters.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Data with cluster labels saved to: {output_path}")
    
    # 打印每个聚类的统计信息
    print("\nCluster statistics:")
    cluster_stats = df.groupby("cluster").size().sort_values(ascending=False)
    print(cluster_stats)
    
    # 分析异常聚类（小聚类可能是黑产）
    print("\nPotential fraud clusters (small clusters):")
    small_clusters = cluster_stats[cluster_stats < len(df) * 0.05]
    if len(small_clusters) > 0:
        print(small_clusters)
        
        # 详细分析小聚类
        for cluster_id in small_clusters.index:
            print(f"\n--- Cluster {cluster_id} ---")
            cluster_data = df[df["cluster"] == cluster_id]
            
            # 显示数值特征统计
            numerical_cols = cluster_data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                print("\nNumerical features:")
                print(cluster_data[numerical_cols].describe())
            
            # 显示类别特征分布
            categorical_cols = cluster_data.select_dtypes(exclude=[np.number]).columns
            categorical_cols = [col for col in categorical_cols if col != "cluster"]
            for col in categorical_cols[:3]:  # 只显示前几个
                print(f"\n{col}:")
                print(cluster_data[col].value_counts().head())
    else:
        print("No small clusters found")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Cluster analysis of SAINT embeddings")
    parser.add_argument("--embeddings_path", type=str, required=True, 
                        help="Path to embeddings file (.parquet or .npy)")
    parser.add_argument("--original_data_path", type=str, required=True,
                        help="Path to original data file (for cluster analysis)")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory")
    parser.add_argument("--method", type=str, default="kmeans",
                        choices=["kmeans", "dbscan"], help="Clustering method")
    parser.add_argument("--n_clusters", type=int, default=8,
                        help="Number of clusters for K-Means")
    parser.add_argument("--eps", type=float, default=0.5,
                        help="Epsilon for DBSCAN")
    parser.add_argument("--min_samples", type=int, default=5,
                        help="Min samples for DBSCAN")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization")
    parser.add_argument("--visualize_method", type=str, default="tsne",
                        choices=["tsne", "pca"], help="Visualization method")
    
    args = parser.parse_args()
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载 embeddings
    print("Loading embeddings...")
    embeddings = load_embeddings(args.embeddings_path)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # 聚类
    if args.method == "kmeans":
        labels, model = cluster_kmeans(embeddings, n_clusters=args.n_clusters)
    else:
        labels, model = cluster_dbscan(embeddings, eps=args.eps, min_samples=args.min_samples)
    
    # 可视化
    if args.visualize:
        visualize_embeddings(embeddings, labels, args.output_dir, method=args.visualize_method)
    
    # 分析聚类结果
    analyze_clusters(args.original_data_path, labels, args.output_dir)
    
    print("\nCluster analysis completed!")


if __name__ == "__main__":
    main()
