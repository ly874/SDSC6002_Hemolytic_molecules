"""
Training Set Similarity Analysis
================================
Calculate Tanimoto similarities between different confidence levels of saponins
to verify if non-saponins are pulling down the average.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from tabulate import tabulate
import warnings

warnings.filterwarnings('ignore')

# ============================================
# 1. Setup and Data Loading
# ============================================

print("\n" + "=" * 70)
print("TRAINING SET SIMILARITY ANALYSIS")
print("=" * 70)

base_path = 'D:/LuYang/CityU Study/SDSC6002/sanopin_project/data/'

# Load training set and classification results
train_path = base_path + 'success_samples_train.csv'
analysis_path = 'saponin_analysis_robust.csv'  # Your analysis results

try:
    train_df = pd.read_csv(train_path)
    analysis_df = pd.read_csv(analysis_path)
    print(f"✅ Loaded {len(train_df)} training molecules")
    print(f"✅ Loaded analysis results with {len(analysis_df)} classifications")
except FileNotFoundError as e:
    print(f"❌ File not found: {e}")
    exit()

# ============================================
# 2. Generate Fingerprints
# ============================================

print("\n📊 Generating fingerprints...")

morgan_gen = GetMorganGenerator(radius=3, fpSize=2048)


def smiles_to_fingerprint(smiles):
    """Convert SMILES to fingerprint using robust parser"""
    if not isinstance(smiles, str) or smiles == "":
        return None

    # Fix nitro groups
    smiles = smiles.replace('N(=O)O', '[N+](=O)[O-]')
    smiles = smiles.replace('N(=O)[O]', '[N+](=O)[O-]')

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            )
        except:
            return None

    return morgan_gen.GetFingerprint(mol)


# Generate fingerprints for all molecules
fingerprints = []
valid_indices = []

for idx, smi in enumerate(train_df['SMILES']):
    fp = smiles_to_fingerprint(smi)
    if fp is not None:
        fingerprints.append(fp)
        valid_indices.append(idx)

print(f"✅ Generated {len(fingerprints)} valid fingerprints")

# ============================================
# 3. Group by Classification
# ============================================

# Merge with analysis results
merged_df = train_df.iloc[valid_indices].copy()
merged_df['Classification'] = analysis_df['Classification'].values[:len(merged_df)]

# Define groups
high_conf = merged_df[merged_df['Classification'] == 'HIGH_CONFIDENCE_SAPONIN']
likely = merged_df[merged_df['Classification'] == 'LIKELY_SAPONIN']
possible = merged_df[merged_df['Classification'] == 'POSSIBLE_SAPONIN']
unlikely = merged_df[merged_df['Classification'] == 'UNLIKELY_SAPONIN']
not_saponin = merged_df[merged_df['Classification'] == 'NOT_SAPONIN']

# Combined groups
high_likely = merged_df[merged_df['Classification'].isin(['HIGH_CONFIDENCE_SAPONIN', 'LIKELY_SAPONIN'])]
medium = merged_df[merged_df['Classification'].isin(['POSSIBLE_SAPONIN'])]
low = merged_df[merged_df['Classification'].isin(['UNLIKELY_SAPONIN', 'NOT_SAPONIN'])]

print("\n📊 Group sizes:")
print(f"  HIGH_CONFIDENCE: {len(high_conf)}")
print(f"  LIKELY: {len(likely)}")
print(f"  POSSIBLE: {len(possible)}")
print(f"  UNLIKELY: {len(unlikely)}")
print(f"  NOT_SAPONIN: {len(not_saponin)}")
print(f"  HIGH+LIKELY (True saponins): {len(high_likely)}")
print(f"  LOW (Unlikely+Not): {len(low)}")


# ============================================
# 4. Calculate Similarities Between Groups
# ============================================

def calculate_group_similarity(group1_fps, group2_fps, sample_size=1000):
    """
    Calculate average Tanimoto similarity between two groups.
    If groups are large, sample to avoid O(n²) computation.
    """
    if len(group1_fps) == 0 or len(group2_fps) == 0:
        return 0.0

    # Sample if too large
    if len(group1_fps) > sample_size:
        idx1 = np.random.choice(len(group1_fps), sample_size, replace=False)
        group1_fps = [group1_fps[i] for i in idx1]

    if len(group2_fps) > sample_size:
        idx2 = np.random.choice(len(group2_fps), sample_size, replace=False)
        group2_fps = [group2_fps[i] for i in idx2]

    similarities = []
    for fp1 in group1_fps:
        for fp2 in group2_fps:
            sim = DataStructs.TanimotoSimilarity(fp1, fp2)
            similarities.append(sim)

    return np.mean(similarities), np.std(similarities), len(similarities)


# Get fingerprints for each group
def get_fps_by_indices(df_group, all_fps, all_indices):
    """Get fingerprints for molecules in df_group"""
    group_indices = df_group.index.tolist()
    group_fps = []
    for idx in group_indices:
        if idx in valid_indices:
            pos = valid_indices.index(idx)
            group_fps.append(fingerprints[pos])
    return group_fps


high_conf_fps = get_fps_by_indices(high_conf, fingerprints, valid_indices)
likely_fps = get_fps_by_indices(likely, fingerprints, valid_indices)
possible_fps = get_fps_by_indices(possible, fingerprints, valid_indices)
unlikely_fps = get_fps_by_indices(unlikely, fingerprints, valid_indices)
not_fps = get_fps_by_indices(not_saponin, fingerprints, valid_indices)

high_likely_fps = get_fps_by_indices(high_likely, fingerprints, valid_indices)
low_fps = get_fps_by_indices(low, fingerprints, valid_indices)

# Calculate similarities
print("\n🔬 Calculating similarities between groups...")

results = []

# High vs High
sim, std, n = calculate_group_similarity(high_conf_fps, high_conf_fps)
results.append(['HIGH vs HIGH', f"{sim:.4f}", f"±{std:.4f}", n])

# High vs Likely
sim, std, n = calculate_group_similarity(high_conf_fps, likely_fps)
results.append(['HIGH vs LIKELY', f"{sim:.4f}", f"±{std:.4f}", n])

# High vs Possible
sim, std, n = calculate_group_similarity(high_conf_fps, possible_fps)
results.append(['HIGH vs POSSIBLE', f"{sim:.4f}", f"±{std:.4f}", n])

# High vs Unlikely
sim, std, n = calculate_group_similarity(high_conf_fps, unlikely_fps)
results.append(['HIGH vs UNLIKELY', f"{sim:.4f}", f"±{std:.4f}", n])

# High vs Not
sim, std, n = calculate_group_similarity(high_conf_fps, not_fps)
results.append(['HIGH vs NOT', f"{sim:.4f}", f"±{std:.4f}", n])

# Likely vs Possible
sim, std, n = calculate_group_similarity(likely_fps, possible_fps)
results.append(['LIKELY vs POSSIBLE', f"{sim:.4f}", f"±{std:.4f}", n])

# Likely vs Unlikely
sim, std, n = calculate_group_similarity(likely_fps, unlikely_fps)
results.append(['LIKELY vs UNLIKELY', f"{sim:.4f}", f"±{std:.4f}", n])

# Likely vs Not
sim, std, n = calculate_group_similarity(likely_fps, not_fps)
results.append(['LIKELY vs NOT', f"{sim:.4f}", f"±{std:.4f}", n])

# HIGH+LIKELY vs LOW
sim, std, n = calculate_group_similarity(high_likely_fps, low_fps)
results.append(['TRUE SAPONINS vs NON-SAPONINS', f"{sim:.4f}", f"±{std:.4f}", n])

# HIGH+LIKELY internal
sim, std, n = calculate_group_similarity(high_likely_fps, high_likely_fps)
results.append(['TRUE SAPONINS internal', f"{sim:.4f}", f"±{std:.4f}", n])

# LOW internal
sim, std, n = calculate_group_similarity(low_fps, low_fps)
results.append(['NON-SAPONINS internal', f"{sim:.4f}", f"±{std:.4f}", n])

# ============================================
# 5. Print Results
# ============================================

print("\n" + "=" * 70)
print("SIMILARITY ANALYSIS RESULTS")
print("=" * 70)

df_results = pd.DataFrame(results, columns=['Comparison', 'Mean Similarity', 'Std Dev', 'Pairs'])
print(tabulate(df_results, headers='keys', tablefmt='grid', showindex=False))

# ============================================
# 6. Visualization
# ============================================

print("\n📈 Creating visualization...")

# Prepare data for plotting
plot_data = {
    'Comparison': [r[0] for r in results],
    'Similarity': [float(r[1]) for r in results],
    'Error': [float(r[2].replace('±', '')) for r in results]
}
plot_df = pd.DataFrame(plot_data)

# Filter for key comparisons
key_comparisons = [
    'TRUE SAPONINS internal',
    'TRUE SAPONINS vs NON-SAPONINS',
    'NON-SAPONINS internal',
    'HIGH vs HIGH',
    'HIGH vs NOT'
]
plot_df_filtered = plot_df[plot_df['Comparison'].isin(key_comparisons)]

plt.figure(figsize=(12, 6))
bars = plt.bar(plot_df_filtered['Comparison'], plot_df_filtered['Similarity'],
               yerr=plot_df_filtered['Error'], capsize=5, alpha=0.7,
               color=['#2ecc71', '#e74c3c', '#3498db', '#2ecc71', '#e74c3c'])

# Add threshold line
plt.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Threshold 0.3')
plt.axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, label='Threshold 0.25')

plt.ylabel('Average Tanimoto Similarity')
plt.title('Similarity Between Different Saponin Confidence Groups')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('group_similarities.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 7. Key Insight
# ============================================

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)

true_saponin_internal = float(
    df_results[df_results['Comparison'] == 'TRUE SAPONINS internal']['Mean Similarity'].values[0])
true_vs_low = float(
    df_results[df_results['Comparison'] == 'TRUE SAPONINS vs NON-SAPONINS']['Mean Similarity'].values[0])
low_internal = float(df_results[df_results['Comparison'] == 'NON-SAPONINS internal']['Mean Similarity'].values[0])

print(f"\n📊 True saponins internal similarity: {true_saponin_internal:.4f}")
print(f"📊 True saponins vs non-saponins similarity: {true_vs_low:.4f}")
print(f"📊 Non-saponins internal similarity: {low_internal:.4f}")

if true_vs_low < 0.2:
    print("\n✅ HYPOTHESIS CONFIRMED: Non-saponins are very different from true saponins!")
    print(f"   They only share {true_vs_low * 100:.1f}% similarity, which pulls down averages.")

    # Calculate the "drag" effect
    drag_effect = (453 * true_saponin_internal + 171 * true_vs_low) / 624 - true_saponin_internal
    print(f"\n📉 Drag effect: {abs(drag_effect) * 100:.2f}% reduction in average")
    print(f"   To achieve overall 0.3, saponins need {0.3 + abs(drag_effect):.3f} internal similarity")
else:
    print("\n⚠️ Non-saponins are somewhat similar to saponins - filtering may have less impact")

# Save results
df_results.to_csv('similarity_analysis_results.csv', index=False)
print("\n✅ Results saved to: similarity_analysis_results.csv")
print("✅ Plot saved to: group_similarities.png")
print("\n" + "=" * 70)