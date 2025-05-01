import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

PHANOTATE_DIR = 'phanotate_out'
INPUT_DIRS = ['pre_train', 'fine_tuned']
OUTPUT_DIR = 'phanotate_analysis_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_phanotate_files():
    """Parse Phanotate tabular output for all context lengths and both models."""
    data = []
    for model in INPUT_DIRS:
        model_dir = os.path.join(PHANOTATE_DIR, model)
        if not os.path.isdir(model_dir):
            continue
        for file in os.listdir(model_dir):
            if file.endswith('.txt'):
                # Extract context length from filename
                match = re.search(r'(\d+)_\d+', file)
                context = match.group(1) if match else 'unknown'
                txt_path = os.path.join(model_dir, file)
                with open(txt_path, 'r') as f:
                    header_seen = False
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            if 'START' in line and 'STOP' in line and 'SCORE' in line:
                                header_seen = True
                            continue
                        if header_seen:
                            cols = line.split()
                            if len(cols) >= 5:
                                try:
                                    strand = cols[2]
                                    score = float(cols[4])
                                    data.append({
                                        'Model': model,
                                        'Context_Length': context,
                                        'Strand': strand,
                                        'Log_Likelihood': score
                                    })
                                except Exception:
                                    continue
    return pd.DataFrame(data)

def summarize_and_plot(df):
    if df.empty:
        print("No Phanotate data found!")
        return

    df['Context_Length'] = pd.to_numeric(df['Context_Length'], errors='coerce')
    df = df.dropna(subset=['Context_Length'])
    df = df.sort_values('Context_Length')

    # Average log likelihood per model/context
    ll_data = df.groupby(['Model', 'Context_Length'])['Log_Likelihood'].mean().reset_index()

    # 1. Log Likelihood Plot (scientific notation)
    plt.figure(figsize=(12, 7))
    sns.lineplot(x='Context_Length', y='Log_Likelihood', hue='Model', marker='o', data=ll_data)
    plt.title('Average Log Likelihood by Context Length', fontsize=16)
    plt.xlabel('Context Length', fontsize=14)
    plt.ylabel('Average Log Likelihood', fontsize=14)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    for model in ll_data['Model'].unique():
        sub = ll_data[ll_data['Model'] == model]
        for _, row in sub.iterrows():
            plt.text(row['Context_Length'], row['Log_Likelihood'], f"{row['Log_Likelihood']:.1e}", fontsize=9, ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'log_likelihood.png'), dpi=300)
    plt.close()

    # 2. Log Likelihood Difference Plot (fine_tuned - pre_train)
    pivot = ll_data.pivot_table(index='Context_Length', columns='Model', values='Log_Likelihood', aggfunc='mean')
    if 'fine_tuned' in pivot.columns and 'pre_train' in pivot.columns:
        pivot['difference'] = pivot['fine_tuned'] - pivot['pre_train']
        plt.figure(figsize=(10, 6))
        plt.plot(pivot.index, pivot['difference'], marker='o', color='purple')
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.title('Difference in Log Likelihood (Fine-tuned minus Pre-trained)', fontsize=16)
        plt.xlabel('Context Length', fontsize=14)
        plt.ylabel('Log Likelihood Difference', fontsize=14)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        for idx, row in pivot.iterrows():
            plt.text(idx, row['difference'], f"{row['difference']:.1e}", fontsize=9, ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'log_likelihood_difference.png'), dpi=300)
        plt.close()

    # 3. Strand Distribution Plot
    strand_counts = df.groupby(['Model', 'Context_Length', 'Strand']).size().reset_index(name='Count')
    total_counts = df.groupby(['Model', 'Context_Length']).size().reset_index(name='Total')
    strand_counts = strand_counts.merge(total_counts, on=['Model', 'Context_Length'])
    strand_counts['Percentage'] = (strand_counts['Count'] / strand_counts['Total'] * 100).round(1)
    plus_strand = strand_counts[strand_counts['Strand'] == '+']
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x='Context_Length', y='Percentage', hue='Model', data=plus_strand)
    plt.title('Phanotate: Percentage of Genes on + Strand', fontsize=16)
    plt.xlabel('Context Length', fontsize=14)
    plt.ylabel('Percentage on + Strand (%)', fontsize=14)
    plt.axhline(y=50, color='gray', linestyle='--')
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width()/2., p.get_height()), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'strand_distribution.png'), dpi=300)
    plt.close()

    # Save summary
    ll_data.to_csv(os.path.join(OUTPUT_DIR, 'log_likelihood_summary.csv'), index=False)
    plus_strand.to_csv(os.path.join(OUTPUT_DIR, 'plus_strand_summary.csv'), index=False)

if __name__ == "__main__":
    print("Parsing Phanotate files...")
    phanotate_df = parse_phanotate_files()
    print(f"Found {len(phanotate_df)} Phanotate CDS features")
    summarize_and_plot(phanotate_df)
    print(f"Analysis complete! Results saved in {OUTPUT_DIR}")
