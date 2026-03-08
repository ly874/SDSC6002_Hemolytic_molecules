"""
Saponin Detection Module - Using Robust SMILES Processing
=========================================================
This module analyzes molecules using the same robust method that
successfully processed all 624 training molecules.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import pandas as pd
import numpy as np
from tabulate import tabulate
import warnings

warnings.filterwarnings('ignore')

# ============================================
# 1. REUSE YOUR WORKING CODE - Same robust parser
# ============================================

# Create Morgan generator (reusable)
morgan_gen = GetMorganGenerator(radius=3, fpSize=2048)


def robust_mol_from_smiles(smiles):
    """Robust SMILES parser that handles nitro groups - EXACTLY from your working code"""
    if not isinstance(smiles, str) or smiles == "":
        return None
    # Fix nitro groups
    smiles = smiles.replace('N(=O)O', '[N+](=O)[O-]')
    smiles = smiles.replace('N(=O)[O]', '[N+](=O)[O-]')

    # First attempt: standard parsing
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol

    # Second attempt: parse without sanitization
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None

    try:
        # Skip kekulization only
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        )
        return mol
    except:
        return None


def process_smiles_list(smiles_list):
    """
    Process SMILES list using the same method that successfully handled your training data.
    Returns valid molecules and their indices.
    """
    valid_mols = []
    valid_indices = []
    failed_indices = []

    for idx, smi in enumerate(smiles_list):
        mol = robust_mol_from_smiles(smi)
        if mol is not None:
            try:
                # Test if fingerprint can be generated
                fp = morgan_gen.GetFingerprint(mol)
                valid_mols.append(mol)
                valid_indices.append(idx)
            except:
                failed_indices.append(idx)
        else:
            failed_indices.append(idx)

    return valid_mols, valid_indices, failed_indices


# ============================================
# 2. Saponin Detector Class
# ============================================

class SaponinDetector:
    """Comprehensive saponin detector using robust molecular processing"""

    def __init__(self, verbose=False):
        self.verbose = verbose

    def calculate_features(self, mol):
        """Calculate all relevant features for saponin detection"""
        if mol is None:
            return None

        features = {}

        # Basic molecular properties
        features['Molecular_Weight'] = Descriptors.MolWt(mol)
        features['Num_Atoms'] = mol.GetNumAtoms()
        features['Num_Heavy_Atoms'] = mol.GetNumHeavyAtoms()

        # Ring analysis
        features['Num_Rings'] = rdMolDescriptors.CalcNumRings(mol)
        features['Num_Aromatic_Rings'] = rdMolDescriptors.CalcNumAromaticRings(mol)
        features['Num_Aliphatic_Rings'] = features['Num_Rings'] - features['Num_Aromatic_Rings']

        # Elemental composition
        features['Oxygen_Count'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
        features['Nitrogen_Count'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
        features['Carbon_Count'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)

        # Hydroxyl groups
        oh_pattern = Chem.MolFromSmarts('[OH]')
        features['Hydroxyl_Count'] = len(mol.GetSubstructMatches(oh_pattern)) if oh_pattern else 0

        # Glycosidic bonds (C-O-C-O pattern - characteristic of sugar chains)
        glycosidic_pattern = Chem.MolFromSmarts('[C][O][C][O]')
        features['Glycosidic_Bonds'] = len(mol.GetSubstructMatches(glycosidic_pattern)) if glycosidic_pattern else 0

        # Chiral centers (sugars have many chiral centers)
        features['Chiral_Centers'] = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))

        # Fused rings (saponin characteristic)
        features['Has_Fused_Rings'] = 1 if features['Num_Rings'] >= 4 else 0

        # O/C ratio
        features['O_C_Ratio'] = features['Oxygen_Count'] / max(features['Carbon_Count'], 1)

        return features

    def calculate_saponin_score(self, features):
        """Calculate composite score indicating likelihood of being a saponin"""
        score = 0
        reasons = []

        # Molecular weight (saponins typically >600 Da)
        mw = features['Molecular_Weight']
        if mw > 1000:
            score += 35
            reasons.append(f"MW: {mw:.1f} > 1000 (+35)")
        elif mw > 800:
            score += 30
            reasons.append(f"MW: {mw:.1f} > 800 (+30)")
        elif mw > 600:
            score += 20
            reasons.append(f"MW: {mw:.1f} > 600 (+20)")
        elif mw > 400:
            score += 10
            reasons.append(f"MW: {mw:.1f} > 400 (+10)")
        else:
            reasons.append(f"MW: {mw:.1f} < 400 (0)")

        # Ring count (saponins have 4+ rings)
        rings = features['Num_Rings']
        if rings >= 6:
            score += 25
            reasons.append(f"Rings: {rings} ≥ 6 (+25)")
        elif rings >= 4:
            score += 20
            reasons.append(f"Rings: {rings} ≥ 4 (+20)")
        elif rings >= 3:
            score += 10
            reasons.append(f"Rings: {rings} ≥ 3 (+10)")

        # Oxygen content (saponins have many oxygens from sugars)
        oxygens = features['Oxygen_Count']
        if oxygens >= 15:
            score += 25
            reasons.append(f"Oxygen: {oxygens} ≥ 15 (+25)")
        elif oxygens >= 10:
            score += 20
            reasons.append(f"Oxygen: {oxygens} ≥ 10 (+20)")
        elif oxygens >= 5:
            score += 15
            reasons.append(f"Oxygen: {oxygens} ≥ 5 (+15)")

        # Hydroxyl groups (characteristic of sugars)
        hydroxyl = features['Hydroxyl_Count']
        if hydroxyl >= 8:
            score += 20
            reasons.append(f"Hydroxyl: {hydroxyl} ≥ 8 (+20)")
        elif hydroxyl >= 4:
            score += 15
            reasons.append(f"H ydroxyl: {hydroxyl} ≥ 4 (+15)")
        elif hydroxyl >= 2:
            score += 5
            reasons.append(f"H ydroxyl: {hydroxyl} ≥ 2 (+5)")

        # Glycosidic bonds (evidence of sugar chains)
        if features['Glycosidic_Bonds'] >= 2:
            score += 20
            reasons.append(f"Glycosidic bonds: {features['Glycosidic_Bonds']} ≥ 2 (+20)")
        elif features['Glycosidic_Bonds'] >= 1:
            score += 15
            reasons.append(f"Has glycosidic bonds: {features['Glycosidic_Bonds']} (+15)")

        # Chiral centers
        if features['Chiral_Centers'] >= 10:
            score += 15
            reasons.append(f"Chiral centers: {features['Chiral_Centers']} ≥ 10 (+15)")
        elif features['Chiral_Centers'] >= 5:
            score += 10
            reasons.append(f"Chiral centers: {features['Chiral_Centers']} ≥ 5 (+10)")

        # O/C ratio
        if features['O_C_Ratio'] > 0.2:
            score += 10
            reasons.append(f"O/C ratio: {features['O_C_Ratio']:.3f} > 0.2 (+10)")
        elif features['O_C_Ratio'] > 0.15:
            score += 5
            reasons.append(f"O/C ratio: {features['O_C_Ratio']:.3f} > 0.15 (+5)")

        # PENALTY: Nitrogen (typical saponins don't contain nitrogen)
        if features['Nitrogen_Count'] > 0:
            penalty = features['Nitrogen_Count'] * 5
            score -= min(penalty, 25)
            reasons.append(f"Contains N: {features['Nitrogen_Count']} (-{min(penalty, 25)})")

        return score, reasons

    def classify_saponin(self, score):
        """Classify based on score threshold"""
        if score >= 80:
            return "HIGH_CONFIDENCE_SAPONIN", "Very likely a saponin"
        elif score >= 60:
            return "LIKELY_SAPONIN", "Probably a saponin"
        elif score >= 40:
            return "POSSIBLE_SAPONIN", "Could be a saponin"
        elif score >= 20:
            return "UNLIKELY_SAPONIN", "Probably not a saponin"
        else:
            return "NOT_SAPONIN", "Not a saponin"

    def analyze_molecule(self, mol, smiles, index=None):
        """Analyze a single molecule (already processed)"""
        if mol is None:
            return None

        # Calculate features
        features = self.calculate_features(mol)

        # Calculate saponin score
        score, reasons = self.calculate_saponin_score(features)

        # Get classification
        classification, description = self.classify_saponin(score)

        # Compile results
        result = {
            'Index': index,
            'SMILES': smiles[:60] + '...' if len(smiles) > 60 else smiles,
            'MW': f"{features['Molecular_Weight']:.1f}",
            'Rings': features['Num_Rings'],
            'Oxygen': features['Oxygen_Count'],
            'Nitrogen': features['Nitrogen_Count'],
            'Hydroxyl': features['Hydroxyl_Count'],
            'Chiral': features['Chiral_Centers'],
            'Glycosidic': features['Glycosidic_Bonds'],
            'O/C_Ratio': f"{features['O_C_Ratio']:.3f}",
            'Score': score,
            'Classification': classification,
            'Description': description
        }

        if self.verbose:
            self.print_report(smiles, features, score, reasons, classification, description)

        return result

    def print_report(self, smiles, features, score, reasons, classification, description):
        """Print detailed report for a molecule"""
        print("\n" + "=" * 70)
        print("SAPONIN ANALYSIS REPORT")
        print("=" * 70)
        print(f"SMILES: {smiles[:100]}..." if len(smiles) > 100 else f"SMILES: {smiles}")
        print("-" * 70)

        # Feature table
        print("\n📊 MOLECULAR FEATURES:")
        feature_table = [
            ["Molecular Weight", f"{features['Molecular_Weight']:.1f} Da", ">600 Da"],
            ["Total Rings", features['Num_Rings'], "≥4"],
            ["Oxygen Atoms", features['Oxygen_Count'], "≥8"],
            ["Hydroxyl Groups", features['Hydroxyl_Count'], "≥3"],
            ["Nitrogen Atoms", features['Nitrogen_Count'], "0 (ideal)"],
            ["Chiral Centers", features['Chiral_Centers'], "≥5"],
            ["Glycosidic Bonds", features['Glycosidic_Bonds'], "≥1"],
            ["O/C Ratio", f"{features['O_C_Ratio']:.3f}", ">0.15"]
        ]
        print(tabulate(feature_table, headers=["Feature", "Value", "Expected"], tablefmt="grid"))

        print(f"\n📈 TOTAL SCORE: {score}/100")
        print(f"📌 CLASSIFICATION: {classification}")
        print(f"💬 {description}")
        print("=" * 70)

    def analyze_dataset(self, smiles_list):
        """
        Analyze entire dataset using robust processing
        """
        print(f"\n📊 Processing {len(smiles_list)} molecules with robust parser...")

        # Process all SMILES with your robust method
        valid_mols, valid_indices, failed_indices = process_smiles_list(smiles_list)

        print(f"✅ Successfully processed: {len(valid_mols)} molecules")
        if failed_indices:
            print(f"⚠️ Failed: {len(failed_indices)} molecules")

        # Analyze valid molecules
        results = []
        for idx, mol in zip(valid_indices, valid_mols):
            result = self.analyze_molecule(mol, smiles_list[idx], idx)
            if result:
                results.append(result)

        # Create DataFrame
        df = pd.DataFrame(results)

        # Summary statistics
        self.print_summary(df, len(smiles_list), failed_indices)

        return df, failed_indices

    def print_summary(self, df, total_molecules, failed_indices):
        """Print summary statistics"""
        print("\n" + "=" * 70)
        print("DATASET ANALYSIS SUMMARY")
        print("=" * 70)

        if len(df) == 0:
            print("\n❌ No valid molecules to analyze")
            return

        print(f"\n📈 Total molecules in dataset: {total_molecules}")
        print(f"✅ Successfully analyzed: {len(df)}")
        print(f"⚠️ Failed to process: {len(failed_indices)}")

        # Count by classification
        class_counts = df['Classification'].value_counts()
        print("\n🎯 Classification Results:")
        for class_name, count in class_counts.items():
            pct = count / len(df) * 100
            print(f"  • {class_name}: {count} ({pct:.1f}%)")

        # Statistics
        print("\n📊 Score Statistics:")
        print(f"  • Mean score: {df['Score'].mean():.1f}")
        print(f"  • Median score: {df['Score'].median():.1f}")
        print(f"  • Min score: {df['Score'].min()}")
        print(f"  • Max score: {df['Score'].max()}")

        # Feature averages
        print("\n🔬 Feature Averages:")
        print(f"  • Average MW: {pd.to_numeric(df['MW']).mean():.1f} Da")
        print(f"  • Average rings: {df['Rings'].mean():.1f}")
        print(f"  • Average oxygen: {df['Oxygen'].mean():.1f}")
        print(f"  • Nitrogen containing: {len(df[df['Nitrogen'] > 0])} molecules")


# ============================================
# 3. MAIN EXECUTION
# ============================================

def main():
    """Main function to run saponin analysis"""

    print("\n" + "=" * 70)
    print("SAPONIN DETECTION SYSTEM - USING ROBUST SMILES PARSER")
    print("=" * 70)

    # Mount Google Drive

    base_path = 'D:/LuYang/CityU Study/SDSC6002/sanopin_project/data/'

    # Load training set
    train_path = base_path + 'success_samples_train.csv'
    try:
        train_df = pd.read_csv(train_path)
        print(f"\n📂 Loaded: {train_path}")
        print(f"📊 Total molecules: {len(train_df)}")
    except FileNotFoundError:
        print(f"\n❌ File not found: {train_path}")
        return

    # Initialize detector
    detector = SaponinDetector(verbose=True)

    # Analyze first 5 molecules in detail
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS - FIRST 5 MOLECULES")
    print("=" * 70)

    # Process first 5 with robust method
    sample_smiles = train_df['SMILES'].head(5).tolist()
    valid_mols, valid_indices, _ = process_smiles_list(sample_smiles)

    for i, (mol, idx) in enumerate(zip(valid_mols, valid_indices)):
        detector.analyze_molecule(mol, sample_smiles[idx], idx)

    # Analyze entire dataset
    detector.verbose = False
    print("\n" + "=" * 70)
    print("ANALYZING ENTIRE DATASET")
    print("=" * 70)

    all_smiles = train_df['SMILES'].tolist()
    results_df, failed = detector.analyze_dataset(all_smiles)

    # Save results
    output_file = 'saponin_analysis_robust.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")

    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    if len(results_df) > 0:
        saponins = len(results_df[results_df['Classification'].str.contains('HIGH|LIKELY')])
        non_saponins = len(results_df) - saponins

        print(f"\n📊 Analysis of {len(results_df)} successfully processed molecules:")
        print(f"  • True saponins (HIGH+LIKELY): {saponins}")
        print(f"  • Non-saponins: {non_saponins}")
        print(f"  • Saponin percentage: {saponins / len(results_df) * 100:.1f}%")

        if saponins / len(results_df) * 100 > 50:
            print("\n✅ CONCLUSION: Training set contains mostly saponins.")
            print("   The screening results (0 matches) suggest the threshold 0.3 may be too high.")
        else:
            print("\n⚠️ CONCLUSION: Training set contains significant non-saponins.")
    else:
        print("\n❌ No molecules could be analyzed")

    print("\n" + "=" * 70)


# Run the main function
if __name__ == "__main__":
    main()