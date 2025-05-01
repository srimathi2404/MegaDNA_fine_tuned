from Bio import SeqIO
import pickle
import os
import time
from collections import Counter
import numpy as np

def load_and_analyze_genbank(file_path):
    sequences = []
    skipped_ids = []
    problematic_records = []
    analysis_data = {
        'total_records': 0,
        'feature_counts': Counter(),
        'gene_lengths': [],
        'gc_contents': [],
        'nucleotide_counts': Counter(),
        'protein_lengths': [],
        'record_details': []
    }

    print(f"\n{'='*40}")
    print(f"Processing and Analyzing: {file_path}")
    print(f"{'='*40}")
    
    start_time = time.time()
    
    for record in SeqIO.parse(file_path, "genbank"):
        analysis_data['total_records'] += 1
        record_data = {
            'id': record.id,
            'length': len(record.seq),
            'organism': record.annotations.get('organism', 'N/A'),
            'features': Counter()
        }

        try:
            if not record.seq or len(record.seq) == 0:
                skipped_ids.append(record.id)
                print(f"Skipping empty sequence: {record.id}")
                continue

            # Store valid sequence
            sequences.append(str(record.seq))

            # Record metadata
            record_data['description'] = record.description
            record_data['molecule_type'] = record.annotations.get('molecule_type', 'N/A')
            record_data['taxonomy'] = ' > '.join(record.annotations.get('taxonomy', []))

            # Feature analysis
            for feature in record.features:
                ftype = feature.type
                analysis_data['feature_counts'][ftype] += 1
                record_data['features'][ftype] += 1

                # Gene length analysis
                if ftype == 'CDS':
                    gene_len = len(feature)
                    analysis_data['gene_lengths'].append(gene_len)
                    if 'translation' in feature.qualifiers:
                        protein_len = len(feature.qualifiers['translation'][0])
                        analysis_data['protein_lengths'].append(protein_len)

            # GC content calculation
            seq = str(record.seq).upper()
            gc = (seq.count('G') + seq.count('C')) / len(seq) * 100
            analysis_data['gc_contents'].append(gc)

            # Nucleotide distribution
            nucs = Counter(seq)
            analysis_data['nucleotide_counts'] += nucs

            analysis_data['record_details'].append(record_data)

        except Exception as e:
            problematic_records.append(record.id)
            print(f"Error processing {record.id}: {str(e)}")

    elapsed_time = time.time() - start_time
    print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
    
    return sequences, skipped_ids, problematic_records, analysis_data

def generate_report(analysis_data):
    """Generate comprehensive analysis report"""
    report = []
    
    # Basic statistics
    report.append(f"\n{'='*40}")
    report.append("GenBank File Analysis Report")
    report.append(f"{'='*40}")
    report.append(f"Total records processed: {analysis_data['total_records']}")
    report.append(f"Valid sequences: {len(analysis_data['record_details'])}")
    report.append(f"Average sequence length: {np.mean([r['length'] for r in analysis_data['record_details']]):.2f} bp")
    report.append(f"Total GC content: {np.mean(analysis_data['gc_contents']):.2f}% (±{np.std(analysis_data['gc_contents']):.2f})")
    
    # Feature breakdown
    report.append("\nFeature Counts:")
    for ftype, count in analysis_data['feature_counts'].most_common():
        report.append(f"  {ftype}: {count}")

    # Gene statistics
    if analysis_data['gene_lengths']:
        report.append("\nGene Statistics (CDS features):")
        report.append(f"  Total genes: {len(analysis_data['gene_lengths'])}")
        report.append(f"  Average length: {np.mean(analysis_data['gene_lengths']):.2f} bp")
        report.append(f"  Median length: {np.median(analysis_data['gene_lengths']):.2f} bp")
        report.append(f"  Range: {min(analysis_data['gene_lengths'])}-{max(analysis_data['gene_lengths'])} bp")

    # Protein statistics
    if analysis_data['protein_lengths']:
        report.append("\nProtein Statistics:")
        report.append(f"  Total proteins: {len(analysis_data['protein_lengths'])}")
        report.append(f"  Average length: {np.mean(analysis_data['protein_lengths']):.2f} aa")
        report.append(f"  Median length: {np.median(analysis_data['protein_lengths']):.2f} aa")
        report.append(f"  Range: {min(analysis_data['protein_lengths'])}-{max(analysis_data['protein_lengths'])} aa")

    # Nucleotide distribution
    total_bases = sum(analysis_data['nucleotide_counts'].values())
    report.append("\nNucleotide Composition:")
    for base in ['A', 'T', 'C', 'G']:
        count = analysis_data['nucleotide_counts'].get(base, 0)
        report.append(f"  {base}: {count} ({count/total_bases*100:.2f}%)")

    # Record examples
    report.append("\nSample Records:")
    for i, rec in enumerate(analysis_data['record_details'][:3]):
        report.append(f"\nRecord {i+1}: {rec['id']}")
        report.append(f"  Organism: {rec['organism']}")
        report.append(f"  Length: {rec['length']} bp")
        report.append(f"  Features: {sum(rec['features'].values())} total")
        report.append(f"  Top Features: {rec['features'].most_common(3)}")

    return "\n".join(report)

def main():
    genbank_file_path = '2Mar2025_phages_downloaded_from_genbank.gb'
    
    # Create output directory
    preprocessed_dir = 'preprocessed_data'
    os.makedirs(preprocessed_dir, exist_ok=True)

    # Process and analyze file
    sequences, skipped, problematic, analysis = load_and_analyze_genbank(genbank_file_path)
    
    # Print analysis report
    print(generate_report(analysis))
    
    # Print preprocessing statistics
    print(f"\n{'='*40}")
    print(f"Loaded {len(sequences)} valid sequences")
    print(f"Skipped {len(skipped)} records")
    print(f"Encountered {len(problematic)} problematic records")
    
    # Save preprocessed data
    with open(os.path.join(preprocessed_dir, 'valid_sequences.pkl'), 'wb') as f:
        pickle.dump(sequences, f)
        
    print(f"\nPreprocessing data saved to {preprocessed_dir}/")

if __name__ == "__main__":
    main()
