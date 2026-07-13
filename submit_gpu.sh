#!/bin/bash
#SBATCH --job-name=MatrixNaive
#SBATCH --output=MatrixNaive.out
#SBATCH --error=MatrixNaive.err
#SBATCH --partition=l40s        # Se esiste, usa la partizione CPU, altrimenti rimetti l40s
#SBATCH --cpus-per-task=1      # Il codice naive in Python usa un solo core
#SBATCH --mem=8G               # 32GB sono esagerati per N=256 o 512, 8GB bastano e avanzano
#SBATCH --time=00:30:00        # Mezz'ora è più che sufficiente per i test iniziali

set -e

echo "Attivazione ambiente virtuale..."
source /scratch.hpc/fabio.ciraci2/gpuAccProgramming/.venv/bin/activate

cd /scratch.hpc/fabio.ciraci2/gpuAccProgramming

echo "Esecuzione NaiveApproach.py..."

python3 -u cupyProgramming.py

TOKEN="8969231949:AAFv9wU0l4OLUZfJgM5m2KcR9WQvhrElOi4"
CHAT_ID="549421087"
MESSAGE="Execution Completed!"

curl -s -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
     -d "chat_id=$CHAT_ID" \
     -d "text=$MESSAGE" \
     -o /dev/null
