# generate_candidates_local.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import pickle
import os
from datetime import datetime
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# ============================================
# 1. R-group definitions (based on your file)
# ============================================

R10_OPTIONS = [
    "COC(/C(C)=C\\C)=O", "COC(/C(C)=C/C)=O", "COC(/C=C(C)/C)=O",
    "COC(/C=C/C)=O", "COC(/C=C/C6=CC=CC=C6)=O", "COC(CCCC)=O",
    "COC(CCCCC)=O", "COC(C6=CC=CC=C6)=O", "COC(C(CC)CC)=O",
]

R1_R2_R4_R5_R8_OPTIONS = [
    "COC(/C(C)=C\\C)=O", "COC(/C(C)=C/C)=O", "COC(/C=C(C)/C)=O",
    "COC(/C=C/C)=O", "COC(/C=C/C6=CC=CC=C6)=O", "COC(CCCC)=O",
    "COC(CCCCC)=O", "COC(C6=CC=CC=C6)=O", "COC(C(CC)CC)=O",
    "CO", "O",
    "OC(/C(C)=C\\C)=O", "OC(/C(C)=C/C)=O", "OC(/C=C(C)/C)=O",
    "OC(C)=O", "OC(/C=C/C)=O", "OC(/C=C/C6=CC=CC=C6)=O",
    "OC(CCCC)=O", "OC(CCCCC)=O", "OC(C6=CC=CC=C6)=O",
    "OC(C(CC)CC)=O", "COC(C)=O", "[H]",
]

R3_OPTIONS = ["[H]", "O"]

CORE_TEMPLATE = "C[C@]1([C@]2(CC=C3[C@@]4(CC(C)([C@H]([C@@H]([C@@]4([C@@H]([C@@H]([C@]3([C@@]2(CC[C@]1([C@@]5(C)[R10])[H])C)C)[R3])[R8])[R4])[R2])[R1])C)[H])[H])CC[C@@H]5[R5]"

# ============================================
# 2. Pruning rules (Rules 1-3) - CORRECTED based on your feedback
# ============================================

# Definition of bulky groups (Rule 3)
BULKY_PATTERNS = [
    43,
]

def is_bulky(r_smiles):
    """Check if an R-group is bulky (Rule 3)"""
    return any(pattern in r_smiles for pattern in BULKY_PATTERNS)

def passes_pruning_rules(combo):
    """
    Combined pruning rules (Rules 1+2+3) - CORRECTED
    
    CORRECT molecular topology (based on user feedback):
        R1
         |
    R2——[骨架]——R4
              |
              R8
              |
              R3
              |
              R5——R10
    
    Adjacent pairs:
    1. R1 and R2
    2. R2 and R4
    3. R4 and R8
    4. R8 and R3
    5. R5 and R10
    
    Rule 1 (Steric hindrance): Adjacent positions cannot both be bulky
    Rule 2 (H-bond pattern): When R3=OH, adjacent positions need H-bond acceptors
    Rule 3 (Bulky group definition): Implemented by is_bulky()
    """
    # === Rule 1: Steric hindrance check for ALL adjacent pairs ===
    
    # Adjacent pair 1: R1 and R2
    if is_bulky(combo['r1']) and is_bulky(combo['r2']):
        return False
    
    # Adjacent pair 2: R2 and R4
    if is_bulky(combo['r2']) and is_bulky(combo['r4']):
        return False
    
    # Adjacent pair 3: R4 and R8
    if is_bulky(combo['r4']) and is_bulky(combo['r8']):
        return False
    
    # Adjacent pair 4: R8 and R3
    if is_bulky(combo['r8']) and is_bulky(combo['r3']):
        return False
    
    # Adjacent pair 5: R5 and R10
    if is_bulky(combo['r5']) and is_bulky(combo['r10']):
        return False
    
    # === Rule 2: H-bond pattern check ===
    # When R3 is OH (represented as "O")
    if combo['r3'] == "O":
        # R3 is adjacent to R8 only
        # So check R8 for carbonyl as H-bond acceptor
        has_carbonyl_r8 = 'C=O' in combo['r8']
        
        if not has_carbonyl_r8:
            return False
    
    # Passed all rules
    return True

def generate_smiles(combo):
    """Generate full SMILES from R-group combination"""
    smiles = CORE_TEMPLATE.replace("[R10]", combo['r10'])
    smiles = smiles.replace("[R1]", combo['r1'])
    smiles = smiles.replace("[R2]", combo['r2'])
    smiles = smiles.replace("[R3]", combo['r3'])
    smiles = smiles.replace("[R4]", combo['r4'])
    smiles = smiles.replace("[R5]", combo['r5'])
    smiles = smiles.replace("[R8]", combo['r8'])
    return smiles

def smiles_to_fingerprint(smiles):
    """Generate fingerprint (using new RDKit API, no warnings)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 3, 2048)
    return fp

# ============================================
# 3. Generator class
# ============================================

class SaponinGenerator:
    def __init__(self):
        self.total_count = 0      # total combinations processed
        self.passed_count = 0     # combinations passing pruning
        
    def __iter__(self):
        """Generator: iterate through all combinations, yield only those passing pruning"""
        for r10 in R10_OPTIONS:
            for r1 in R1_R2_R4_R5_R8_OPTIONS:
                for r2 in R1_R2_R4_R5_R8_OPTIONS:
                    for r3 in R3_OPTIONS:
                        for r4 in R1_R2_R4_R5_R8_OPTIONS:
                            for r5 in R1_R2_R4_R5_R8_OPTIONS:
                                for r8 in R1_R2_R4_R5_R8_OPTIONS:
                                    self.total_count += 1
                                    
                                    combo = {
                                        'r10': r10, 'r1': r1, 'r2': r2,
                                        'r3': r3, 'r4': r4, 'r5': r5, 'r8': r8
                                    }
                                    
                                    if passes_pruning_rules(combo):
                                        self.passed_count += 1
                                        yield combo

# ============================================
# 4. Main execution
# ============================================

def main():
    print("="*60)
    print("Saponin Virtual Library Generator (Rules 1-3)")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print topology for verification
    print("\nMolecular topology (adjacent pairs):")
    print("  ✓ R1 - R2")
    print("  ✓ R2 - R4")
    print("  ✓ R4 - R8")
    print("  ✓ R8 - R3")
    print("  ✓ R5 - R10")
    print()
    
    # Create output directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Initialize generator
    generator = SaponinGenerator()
    
    # Batch processing setup
    batch_size = 10000
    current_batch = []
    batch_count = 0
    
    # Progress bar
    pbar = tqdm(desc="Generating candidates", unit=" molecules")
    
    for combo in generator:
        # Generate SMILES
        smiles = generate_smiles(combo)
        current_batch.append(smiles)
        
        # Save batch when full
        if len(current_batch) >= batch_size:
            # Save batch to file
            filename = f"outputs/batch_{batch_count:06d}.txt"
            with open(filename, 'w') as f:
                f.write('\n'.join(current_batch))
            
            # Update progress
            batch_count += 1
            pbar.update(len(current_batch))
            pbar.set_postfix({
                'Total': f"{generator.total_count/1e6:.1f}M",
                'Pass': f"{generator.passed_count/generator.total_count*100:.2f}%"
            })
            
            # Clear batch
            current_batch = []
            
            # Save checkpoint every 10 batches
            if batch_count % 10 == 0:
                checkpoint = {
                    'total_count': generator.total_count,
                    'passed_count': generator.passed_count,
                    'batch_count': batch_count,
                    'time': datetime.now().isoformat()
                }
                with open(f"checkpoints/checkpoint_{batch_count:06d}.pkl", 'wb') as f:
                    pickle.dump(checkpoint, f)
    
    # Save final batch
    if current_batch:
        filename = f"outputs/batch_{batch_count:06d}.txt"
        with open(filename, 'w') as f:
            f.write('\n'.join(current_batch))
        pbar.update(len(current_batch))
    
    pbar.close()
    
    print(f"\nCompletion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total combinations processed: {generator.total_count:,}")
    print(f"Combinations passing pruning: {generator.passed_count:,}")
    print(f"Pass rate: {generator.passed_count/generator.total_count*100:.2f}%")
    print(f"Output files: in outputs/ directory")
    print(f"Number of batch files: {batch_count}")

if __name__ == "__main__":
    main()