import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.pipeline import RunConfig, TestRunner
from src.attacks.corruption.common import CORRUPTIONS

def plot_degradation_line(df, out_path, title):
    """Vẽ đồ thị đường (Line plot) mAP theo mức độ nhiễu (Severity)."""
    plt.figure(figsize=(10, 6))
    
    # Lấy danh sách các models
    models = df['Model'].unique()
    colors = sns.color_palette("husl", len(models))
    markers = ['o', 's', '^', 'D', 'p', 'v', '<', '>']
    
    for i, model in enumerate(models):
        model_df = df[df['Model'] == model]
        # Gom nhóm theo Severity và tính trung bình mAP trên các attack (nếu có nhiều attack)
        grouped = model_df.groupby('Severity')['AP'].mean().reset_index()
        
        # Sắp xếp theo Severity
        grouped = grouped.sort_values('Severity')
        x_vals = grouped['Severity']
        y_vals = grouped['AP']
        
        plt.plot(x_vals, y_vals, marker=markers[i % len(markers)], linewidth=2.5, markersize=8, color=colors[i], label=model)
        
        # Ghi chú giá trị ở mốc Clean (Severity=0)
        clean_val = grouped[grouped['Severity'] == 0]['AP'].values
        if len(clean_val) > 0:
            plt.annotate(f"{clean_val[0]:.2f}", (0, clean_val[0]), textcoords="offset points", xytext=(-15, 5), ha='center', color=colors[i], fontweight='bold')
            
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Mức độ nhiễu (Severity: 0 = Clean)", fontsize=11)
    plt.ylabel("Average Precision (AP)", fontsize=11)
    plt.xticks(sorted(df['Severity'].unique()))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Đã lưu biểu đồ Degradation tại: {out_path}")

def plot_heatmap(df, out_path, title):
    """Vẽ Heatmap thể hiện phần trăm Degradation theo Attack và Severity."""
    models = df['Model'].unique()
    for model in models:
        model_df = df[(df["Model"] == model) & (df["Severity"] > 0)]
        if model_df.empty:
            continue
            
        pivot = model_df.pivot(index="Attack", columns="Severity", values="Degradation (%)")
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", vmin=0, vmax=100)
        plt.title(f"{title} - {model}", fontsize=14, pad=15)
        plt.ylabel("Attack Type")
        plt.xlabel("Severity Level")
        plt.tight_layout()
        
        heatmap_path = out_path.replace(".png", f"_{model}.png")
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        print(f"✅ Đã lưu Heatmap tại: {heatmap_path}")


def main():
    parser = argparse.ArgumentParser(description="Công cụ tập trung để chạy thử nghiệm và đánh giá độ bền (Robustness) của YOLO models.")
    parser.add_argument("--models", nargs="+", default=["yolo11s"], help="Danh sách mô hình cần test (VD: yolo11n yolo11s)")
    parser.add_argument("--attacks", nargs="+", default=["gaussian_noise"], help="Danh sách nhiễu cần test. Dùng 'all' để test toàn bộ Nhóm A.")
    parser.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5], help="Các mức độ nhiễu (Severity)")
    parser.add_argument("--limit", type=int, default=10, help="Số lượng ảnh test tối đa (để chạy nhanh)")
    parser.add_argument("--plot-type", choices=["line", "heatmap", "both"], default="both", help="Loại biểu đồ muốn vẽ (line, heatmap, hoặc cả hai)")
    args = parser.parse_args()

    models = args.models
    attacks_to_run = CORRUPTIONS if "all" in args.attacks else args.attacks
    severities = args.severities

    print(f"=== ROBUSTNESS EVALUATION PIPELINE ===")
    print(f"Models: {models}")
    print(f"Attacks ({len(attacks_to_run)}): {attacks_to_run}")
    print(f"Severities: {severities}")
    print(f"Limit: {args.limit} images")
    print("======================================\n")

    os.makedirs("results", exist_ok=True)
    all_results = []

    for model in models:
        print(f"\n[{model.upper()}] Bắt đầu Pipeline (Bao gồm chạy baseline Clean)...")
        
        config = RunConfig(
            model=model,
            dataset="kitti_2d",
            attacks=attacks_to_run,
            severities=severities,
            limit=args.limit
        )
        report = TestRunner().run(config)
        ap_clean = report.ap_clean
        print(f"[{model.upper()}] Baseline AP (Clean): {ap_clean:.4f}")

        # Thêm kết quả Clean vào dataset (Severity = 0)
        all_results.append({
            "Model": model.upper(),
            "Attack": "None (Clean)",
            "Severity": 0,
            "AP": ap_clean,
            "Degradation (%)": 0.0
        })

        for attack in attacks_to_run:
            for s in severities:
                cell = next((c for c in report.cells if c.attack == attack and c.severity == s), None)
                if cell:
                    ap = cell.ap
                    degradation = ((ap_clean - ap) / ap_clean * 100) if ap_clean > 0 else 0
                    all_results.append({
                        "Model": model.upper(),
                        "Attack": attack,
                        "Severity": s,
                        "AP": ap,
                        "Degradation (%)": degradation
                    })

    # Lưu kết quả thô
    df = pd.DataFrame(all_results)
    csv_path = "results/robustness_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Đã lưu kết quả đo đạc thô vào: {csv_path}")

    # Vẽ biểu đồ
    print("\nĐang tạo biểu đồ...")
    attack_label = "Tất cả nhiễu Nhóm A" if "all" in args.attacks else ", ".join(attacks_to_run)
    
    if args.plot_type in ["line", "both"]:
        line_title = f"Robustness Degradation\n(Attacks: {attack_label} | Images: {args.limit})"
        plot_degradation_line(df, "results/plot_robustness_line.png", line_title)
        
    if args.plot_type in ["heatmap", "both"]:
        heat_title = f"Degradation (%) Heatmap\n(Limit: {args.limit})"
        plot_heatmap(df, "results/plot_robustness_heatmap.png", heat_title)

if __name__ == "__main__":
    main()
