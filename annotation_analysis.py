import os
import subprocess

# Input directories
INPUT_DIRS = ['pre_train', 'fine_tuned']

# Output directories
PROKKA_OUT = 'prokka_out'
PHANOTATE_OUT = 'phanotate_out'

# Number of CPUs for Prokka (set to your system's max)
CPUS = '4'

def run_prokka():
    for model_dir in INPUT_DIRS:
        input_dir = model_dir
        for fname in os.listdir(input_dir):
            if fname.endswith('.fna'):
                input_path = os.path.join(input_dir, fname)
                base_name = os.path.splitext(fname)[0]
                # Output directory for this file
                outdir = os.path.join(PROKKA_OUT, model_dir, base_name)
                os.makedirs(outdir, exist_ok=True)
                # Prokka command
                prokka_cmd = [
                    'prokka',
                    '--kingdom', 'Viruses',
                    '--prefix', 'phage_annot',
                    '--locustag', 'PHAGE',
                    '--cpus', CPUS,
                    '--mincontiglen', '0',
                    '--outdir', outdir,
                    '--force',
                    input_path
                ]
                print(f"Running Prokka for {input_path} ...")
                subprocess.run(prokka_cmd, check=True)

def run_phanotate():
    for model_dir in INPUT_DIRS:
        input_dir = model_dir
        # Output subdir for this model
        phanotate_model_out = os.path.join(PHANOTATE_OUT, model_dir)
        os.makedirs(phanotate_model_out, exist_ok=True)
        for fname in os.listdir(input_dir):
            if fname.endswith('.fna'):
                input_path = os.path.join(input_dir, fname)
                base_name = os.path.splitext(fname)[0]
                output_file = os.path.join(phanotate_model_out, f"{base_name}.txt")
                print(f"Running Phanotate for {input_path} ...")
                # If you have phanotate.py in PATH, use 'phanotate.py'
                # If you have pharokka (recommended for phage), use 'pharokka.py -g phanotate ...'
                try:
                    with open(output_file, 'w') as out_f:
                        subprocess.run(
                            ['phanotate.py', input_path],
                            stdout=out_f,
                            check=True
                        )
                except FileNotFoundError:
                    print("phanotate.py not found. If you are using pharokka, use the following command instead:")
                    print(f"pharokka.py -i {input_path} -o {phanotate_model_out} -g phanotate -t {CPUS} --force")
                    print("Or install phanotate.py and ensure it is in your PATH.")
                    break

if __name__ == "__main__":
    # Run Prokka
    run_prokka()
    # Run Phanotate
    run_phanotate()
    print("All annotations completed!")
