import os
from Bio import SeqIO
import matplotlib.pyplot as plt
import numpy as np

# Create a new folder for plots
plot_folder = 'plots'
os.makedirs(plot_folder, exist_ok=True)

def gc_content(seq):
    """Calculate GC content percentage"""
    g = seq.count('G')
    c = seq.count('C')
    return (g + c) / len(seq) * 100

def gene_density(seq):
    """Calculate ATG start codon density (genes/kb)"""
    return seq.count('ATG') / (len(seq) / 1000)

def load_sequences(folder):
    """Load all FASTA files from a directory"""
    sequences = []
    for fname in os.listdir(folder):
        if fname.endswith('.fna'):
            try:
                # Extract context length from filename (e.g., pretrained_generate_1000_1.fna)
                parts = fname.split('_')
                context_length = int(parts[2])  # Getting the numeric part after 'generate'
                
                for record in SeqIO.parse(os.path.join(folder, fname), 'fasta'):
                    seq = str(record.seq).upper()
                    sequences.append({
                        'seq': seq,
                        'length': len(seq),
                        'context': context_length,
                        'filename': fname
                    })
            except (IndexError, ValueError) as e:
                print(f"Error parsing {fname}: {e}")
    return sequences

# Load sequences from both models
pre_train = load_sequences('pre_train')
fine_tuned = load_sequences('fine_tuned')

# Calculate metrics
def calculate_metrics(sequences):
    return [{
        'gc': gc_content(s['seq']),
        'gene_density': gene_density(s['seq']),
        'length': s['length'],
        'context': s['context'],
        'filename': s['filename']
    } for s in sequences]

pre_metrics = calculate_metrics(pre_train)
fine_metrics = calculate_metrics(fine_tuned)

# Calculate additional metrics
def calculate_kmer_diversity(seq, k=3):
    """Calculate k-mer diversity (unique k-mers / possible k-mers)"""
    if len(seq) < k:
        return 0
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    return len(set(kmers)) / min(4**k, len(kmers))

# Add k-mer diversity metrics
for metrics in [pre_metrics, fine_metrics]:
    for m in metrics:
        seq = [s['seq'] for s in pre_train + fine_tuned if s['filename'] == m['filename']][0]
        m['kmer_diversity'] = calculate_kmer_diversity(seq)

# Visualization function
def plot_comparison(metric, ylabel, formatter=lambda x: x):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by context length
    contexts = sorted({m['context'] for m in pre_metrics + fine_metrics})
    width = 0.35
    
    for i, ctx in enumerate(contexts):
        pre_vals = [m[metric] for m in pre_metrics if m['context'] == ctx]
        fine_vals = [m[metric] for m in fine_metrics if m['context'] == ctx]
        
        x = i - width/2
        ax.bar(x, np.mean(pre_vals), width, label='Pre-trained' if i==0 else None,
               color='blue', alpha=0.7)
        ax.bar(x + width, np.mean(fine_vals), width, label='Fine-tuned' if i==0 else None,
               color='orange', alpha=0.7)
    
    ax.set_xticks(np.arange(len(contexts)))
    ax.set_xticklabels([f'{ctx//1000}k' if ctx >=1000 else str(ctx) for ctx in contexts])
    ax.set_xlabel('Context Length')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} Comparison by Context Length')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f'{metric}_comparison.png'))
    plt.close()

# Plot metrics
plot_comparison('gc', 'GC Content (%)')
plot_comparison('gene_density', 'Gene Density (genes/kb)')
plot_comparison('kmer_diversity', 'Trinucleotide (3-mer) Diversity')

# Plot Sequence Length Distribution
plt.figure(figsize=(12, 6))
plt.boxplot([[m['length'] for m in pre_metrics], 
             [m['length'] for m in fine_metrics]],
            labels=['Pre-trained', 'Fine-tuned'])
plt.ylabel('Sequence Length (bp)')
plt.title('Sequence Length Distribution')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'sequence_length_distribution.png'))
plt.close()

# Nucleotide Composition Comparison
def nucleotide_comp(seq):
    return {nuc: (seq.count(nuc)/len(seq))*100 for nuc in 'ATGC'}

pre_comp = [nucleotide_comp(s['seq']) for s in pre_train]
fine_comp = [nucleotide_comp(s['seq']) for s in fine_tuned]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(4)
width = 0.35

pre_means = [np.mean([c[nuc] for c in pre_comp]) for nuc in 'ATGC']
fine_means = [np.mean([c[nuc] for c in fine_comp]) for nuc in 'ATGC']

ax.bar(x - width/2, pre_means, width, label='Pre-trained')
ax.bar(x + width/2, fine_means, width, label='Fine-tuned')
ax.set_xticks(x)
ax.set_xticklabels(['A', 'T', 'G', 'C'])
ax.set_ylabel('Nucleotide Composition (%)')
ax.set_title('Average Nucleotide Distribution')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'nucleotide_composition.png'))
plt.close()

# Dinucleotide frequency analysis
def dinucleotide_freq(seq):
    """Calculate frequencies of all dinucleotides"""
    dinucs = {}
    for i in range(len(seq)-1):
        dinuc = seq[i:i+2]
        if dinuc not in dinucs:
            dinucs[dinuc] = 0
        dinucs[dinuc] += 1
    
    total = sum(dinucs.values())
    return {d: (count/total)*100 for d, count in dinucs.items()}

pre_dinuc = [dinucleotide_freq(s['seq']) for s in pre_train]
fine_dinuc = [dinucleotide_freq(s['seq']) for s in fine_tuned]

# Get all possible dinucleotides and calculate means
all_dinucs = sorted(set().union(*[d.keys() for d in pre_dinuc + fine_dinuc]))
pre_means = []
fine_means = []

for dinuc in all_dinucs:
    pre_means.append(np.mean([d.get(dinuc, 0) for d in pre_dinuc]))
    fine_means.append(np.mean([d.get(dinuc, 0) for d in fine_dinuc]))

# Plot dinucleotide frequencies
fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(all_dinucs))
width = 0.35

ax.bar(x - width/2, pre_means, width, label='Pre-trained')
ax.bar(x + width/2, fine_means, width, label='Fine-tuned')
ax.set_xticks(x)
ax.set_xticklabels(all_dinucs, rotation=90)
ax.set_ylabel('Frequency (%)')
ax.set_title('Dinucleotide Frequency Comparison')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'dinucleotide_frequency.png'))
plt.close()

print(f"Analysis complete. All plots saved to '{plot_folder}' directory.")
