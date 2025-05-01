import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

# Configuration
INPUT_DIRS = ['pre_train', 'fine_tuned']
PROKKA_DIR = 'prokka_out'
PHANOTATE_DIR = 'phanotate_out'
OUTPUT_DIR = 'annotation_final_out'

# Create output structure
os.makedirs(os.path.join(OUTPUT_DIR, 'tables'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'plots'), exist_ok=True)

def parse_prokka_gff_files():
    """Parse Prokka GFF files to extract CDS length and strand info"""
    data = []
    
    for model in INPUT_DIRS:
        model_dir = os.path.join(PROKKA_DIR, model)
        if not os.path.exists(model_dir):
            continue
            
        for root, dirs, files in os.walk(model_dir):
            if 'phage_annot.gff' in files:
                gff_path = os.path.join(root, 'phage_annot.gff')
                
                # Extract context length from directory name
                dir_name = os.path.basename(root)
                match = re.search(r'(\d+)_\d+$', dir_name)
                context = match.group(1) if match else 'unknown'
                
                # Parse GFF file
                with open(gff_path, 'r') as f:
                    for line in f:
                        if line.startswith('#'):
                            continue
                        parts = line.strip().split('\t')
                        if len(parts) >= 8 and parts[2] == 'CDS':
                            start = int(parts[3])
                            end = int(parts[4])
                            strand = parts[6]  # '+' or '-'
                            length = end - start + 1
                            
                            data.append({
                                'Model': model,
                                'Context_Length': context,
                                'Tool': 'Prokka',
                                'Strand': strand,
                                'Length': length
                            })
    
    return pd.DataFrame(data)

def parse_phanotate_files():
    """Parse Phanotate output files with correct format"""
    data = []
    
    for model in INPUT_DIRS:
        model_dir = os.path.join(PHANOTATE_DIR, model)
        if not os.path.exists(model_dir):
            print(f"Directory not found: {model_dir}")
            continue
            
        for file in os.listdir(model_dir):
            if file.endswith('.txt'):
                txt_path = os.path.join(model_dir, file)
                
                # Extract context length from filename
                match = re.search(r'(\d+)_\d+', file)
                context = match.group(1) if match else 'unknown'
                
                with open(txt_path, 'r') as f:
                    header_seen = False
                    for line in f:
                        line = line.strip()
                        
                        # Skip comment lines and empty lines
                        if not line or line.startswith('#'):
                            # Detect the header line to know the format
                            if '#START' in line and 'STOP' in line and 'SCORE' in line:
                                header_seen = True
                            continue
                        
                        if header_seen:
                            cols = line.split()
                            if len(cols) >= 5:  # Must have at least 5 columns as per the format
                                try:
                                    start = int(cols[0])
                                    stop = int(cols[1])
                                    frame = cols[2]  # This is FRAME in your file, convert to strand
                                    strand = '+' if frame.isdigit() else frame
                                    contig = cols[3]
                                    score = float(cols[4])
                                    
                                    # Calculate length
                                    length = abs(stop - start) + 1
                                    
                                    data.append({
                                        'Model': model,
                                        'Context_Length': context,
                                        'Tool': 'Phanotate',
                                        'Strand': strand,
                                        'Length': length,
                                        'Log_Likelihood': score
                                    })
                                except (ValueError, IndexError) as e:
                                    print(f"Error parsing line in {txt_path}: {line}")
                                    continue
    
    return pd.DataFrame(data)

def generate_comparison_tables(prokka_df, phanotate_df):
    """Create comprehensive comparison tables"""
    all_data = pd.concat([prokka_df, phanotate_df])
    
    # Ensure context length is numeric for proper ordering
    all_data['Context_Length'] = pd.to_numeric(all_data['Context_Length'], errors='coerce')
    
    # CDS Length statistics
    length_stats = all_data.groupby(['Model', 'Tool', 'Context_Length']).agg({
        'Length': ['mean', 'median', 'min', 'max', 'count']
    }).reset_index()
    
    # Strand distribution
    strand_counts = all_data.groupby(['Model', 'Tool', 'Context_Length', 'Strand']).size().reset_index(name='Count')
    strand_pivot = pd.pivot_table(strand_counts, 
                                 values='Count',
                                 index=['Model', 'Tool', 'Context_Length'],
                                 columns=['Strand'], 
                                 fill_value=0)
    
    # Calculate strand percentages
    if '+' in strand_pivot.columns and '-' in strand_pivot.columns:
        strand_pivot['Total'] = strand_pivot['+'] + strand_pivot['-']
        strand_pivot['Percent_Plus'] = (strand_pivot['+'] / strand_pivot['Total'] * 100).round(1)
        strand_pivot['Percent_Minus'] = (strand_pivot['-'] / strand_pivot['Total'] * 100).round(1)
    
    # Log likelihood stats for Phanotate
    if 'Log_Likelihood' in phanotate_df.columns:
        ll_stats = phanotate_df.groupby(['Model', 'Context_Length']).agg({
            'Log_Likelihood': ['mean', 'median', 'min', 'max']
        }).reset_index()
        ll_stats.to_csv(os.path.join(OUTPUT_DIR, 'tables', 'log_likelihood_stats.csv'))
    
    # Save tables
    length_stats.to_csv(os.path.join(OUTPUT_DIR, 'tables', 'cds_length_stats.csv'))
    strand_pivot.to_csv(os.path.join(OUTPUT_DIR, 'tables', 'strand_distribution.csv'))
    
    return length_stats, strand_pivot

def visualize_results(prokka_df, phanotate_df):
    """Generate comprehensive visualizations"""
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Combine data for some plots
    all_data = pd.concat([prokka_df, phanotate_df])
    all_data['Context_Length'] = pd.to_numeric(all_data['Context_Length'], errors='coerce')
    
    # 1. CDS Count Comparison by Context Length
    cds_counts = all_data.groupby(['Tool', 'Model', 'Context_Length']).size().reset_index(name='Count')
    plt.figure(figsize=(12, 7))
    for tool in ['Prokka', 'Phanotate']:
        tool_data = cds_counts[cds_counts['Tool'] == tool]
        plt.subplot(1, 2, 1 if tool == 'Prokka' else 2)
        sns.barplot(x='Context_Length', y='Count', hue='Model', data=tool_data)
        plt.title(f'{tool} CDS Count by Context Length', fontsize=14)
        plt.xlabel('Context Length', fontsize=12)
        plt.ylabel('Number of CDS', fontsize=12)
        plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plots', 'cds_count_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. CDS Length Distribution
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='Context_Length', y='Length', hue='Model', data=all_data, showfliers=False)
    plt.title('CDS Length Distribution by Context Length', fontsize=16)
    plt.xlabel('Context Length', fontsize=14)
    plt.ylabel('CDS Length (bp)', fontsize=14)
    plt.savefig(os.path.join(OUTPUT_DIR, 'plots', 'cds_length_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Strand Distribution for Prokka and Phanotate
    for tool in ['Prokka', 'Phanotate']:
        tool_data = all_data[all_data['Tool'] == tool]
        
        # Skip if no data for this tool
        if tool_data.empty:
            continue
            
        strand_counts = tool_data.groupby(['Model', 'Context_Length', 'Strand']).size().reset_index(name='Count')
        total_counts = tool_data.groupby(['Model', 'Context_Length']).size().reset_index(name='Total')
        strand_counts = strand_counts.merge(total_counts, on=['Model', 'Context_Length'])
        strand_counts['Percentage'] = (strand_counts['Count'] / strand_counts['Total'] * 100).round(1)
        
        # Plot only + strand (- strand is complementary)
        plus_strand = strand_counts[strand_counts['Strand'] == '+']
        
        plt.figure(figsize=(12, 7))
        ax = sns.barplot(x='Context_Length', y='Percentage', hue='Model', data=plus_strand)
        plt.title(f'{tool}: Percentage of CDSs on + Strand', fontsize=16)
        plt.xlabel('Context Length', fontsize=14)
        plt.ylabel('Percentage on + Strand (%)', fontsize=14)
        plt.axhline(y=50, color='gray', linestyle='--')
        
        # Add data labels
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}%", 
                       (p.get_x() + p.get_width()/2., p.get_height()), 
                       ha = 'center', va = 'bottom', fontsize=10)
            
        plt.savefig(os.path.join(OUTPUT_DIR, 'plots', f'{tool.lower()}_strand_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Log Likelihood by Context Length for Phanotate
    if 'Log_Likelihood' in phanotate_df.columns and not phanotate_df.empty:
        # Group by model and context length to get average log likelihood
        ll_data = phanotate_df.groupby(['Model', 'Context_Length']).agg({
            'Log_Likelihood': 'mean'
        }).reset_index()
        
        # Make sure context length is numeric for proper ordering
        ll_data['Context_Length'] = pd.to_numeric(ll_data['Context_Length'])
        
        plt.figure(figsize=(12, 7))
        sns.lineplot(x='Context_Length', y='Log_Likelihood', 
                     hue='Model', style='Model', markers=True, 
                     markersize=10, dashes=False, data=ll_data)
        plt.title('Average Log Likelihood by Context Length', fontsize=16)
        plt.xlabel('Context Length', fontsize=14)
        plt.ylabel('Log Likelihood Score', fontsize=14)
        
        # Add data labels
        for model in ll_data['Model'].unique():
            model_data = ll_data[ll_data['Model'] == model]
            for _, row in model_data.iterrows():
                plt.text(row['Context_Length'], row['Log_Likelihood'], 
                         f"{row['Log_Likelihood']:.2e}", 
                         fontsize=9, ha='center', va='bottom')
        
        plt.savefig(os.path.join(OUTPUT_DIR, 'plots', 'phanotate_log_likelihood.png'), dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # Parse results
    print("Parsing Prokka GFF files...")
    prokka_data = parse_prokka_gff_files()
    print(f"Found {len(prokka_data)} Prokka CDS features")
    
    print("Parsing Phanotate files...")
    phanotate_data = parse_phanotate_files()
    print(f"Found {len(phanotate_data)} Phanotate CDS features")
    
    # Generate comparison tables
    print("Generating comparison tables...")
    length_stats, strand_stats = generate_comparison_tables(prokka_data, phanotate_data)
    
    # Visualize results
    print("Creating visualizations...")
    visualize_results(prokka_data, phanotate_data)
    
    print(f"Analysis complete! Results saved in {OUTPUT_DIR}")
