# 01_check_hardware.py (修正版)
import psutil
import multiprocessing as mp
import time
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

print("="*50)
print("硬件性能检测")
print("="*50)

# CPU核心数
cpu_count = mp.cpu_count()
print(f"CPU核心数: {cpu_count}")

# 内存
memory = psutil.virtual_memory()
print(f"总内存: {memory.total / 1024**3:.1f} GB")
print(f"可用内存: {memory.available / 1024**3:.1f} GB")

# ============================================
# 真实场景速度测试
# ============================================
print("\n" + "="*50)
print("真实场景速度测试")
print("="*50)

# 1. SMILES生成速度测试
print("\n1. 测试SMILES生成速度...")
test_combo = {
    'r10': "COC(/C(C)=C\\C)=O",
    'r1': "COC(C)=O",
    'r2': "O",
    'r3': "[H]",
    'r4': "OC(C)=O",
    'r5': "CO",
    'r8': "[H]"
}

core_template = "C[C@]1([C@]2(CC=C3[C@@]4(CC(C)([C@H]([C@@H]([C@@]4([C@@H]([C@@H]([C@]3([C@@]2(CC[C@]1([C@@]5(C)[R10])[H])C)C)[R3])[R8])[R4])[R2])[R1])C)[H])[H])CC[C@@H]5[R5]"

def generate_smiles(combo):
    smiles = core_template.replace("[R10]", combo['r10'])
    smiles = smiles.replace("[R1]", combo['r1'])
    smiles = smiles.replace("[R2]", combo['r2'])
    smiles = smiles.replace("[R3]", combo['r3'])
    smiles = smiles.replace("[R4]", combo['r4'])
    smiles = smiles.replace("[R5]", combo['r5'])
    smiles = smiles.replace("[R8]", combo['r8'])
    return smiles

# 测试SMILES生成速度
start = time.time()
n_tests = 100000
for i in range(n_tests):
    smiles = generate_smiles(test_combo)
end = time.time()
smiles_time = (end - start) / n_tests * 1000  # 毫秒
print(f"  生成10万次SMILES: {end-start:.2f}秒")
print(f"  平均每个SMILES: {smiles_time:.3f}毫秒")

# 2. 分子对象创建速度测试
print("\n2. 测试分子对象创建速度...")
test_smiles = generate_smiles(test_combo)
start = time.time()
n_tests = 10000  # 少一点，因为分子创建较慢
for i in range(n_tests):
    mol = Chem.MolFromSmiles(test_smiles)
end = time.time()
mol_time = (end - start) / n_tests * 1000
print(f"  创建1万次分子: {end-start:.2f}秒")
print(f"  平均每个分子: {mol_time:.3f}毫秒")

# 3. 指纹生成速度测试
print("\n3. 测试指纹生成速度...")
mol = Chem.MolFromSmiles(test_smiles)
start = time.time()
n_tests = 10000
for i in range(n_tests):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048)
end = time.time()
fp_time = (end - start) / n_tests * 1000
print(f"  生成1万次指纹: {end-start:.2f}秒")
print(f"  平均每个指纹: {fp_time:.3f}毫秒")

# 4. 剪枝规则判断速度测试
print("\n4. 测试剪枝规则判断速度...")

def is_bulky(r_smiles):
    bulky = ['c6ccccc6', 'C=C/C6=CC=CC=C6', 'CCCCC']
    return any(p in r_smiles for p in bulky)

def check_pruning(combo):
    # 简单的剪枝规则
    if is_bulky(combo['r1']) and is_bulky(combo['r2']):
        return False
    return True

start = time.time()
n_tests = 1000000  # 100万次
for i in range(n_tests):
    result = check_pruning(test_combo)
end = time.time()
prune_time = (end - start) / n_tests * 1000 * 1000  # 微秒
print(f"  判断100万次剪枝: {end-start:.2f}秒")
print(f"  平均每次剪枝: {prune_time:.1f}微秒")

# ============================================
# 真实时间估算
# ============================================
print("\n" + "="*50)
print("真实时间估算")
print("="*50)

total_combinations = 116_000_000

# 估算剪枝通过率（从之前的采样数据）
prune_pass_rate = 0.0432  # 4.32%
after_prune = total_combinations * prune_pass_rate

print(f"\n📊 基础数据:")
print(f"  总组合数: {total_combinations:,}")
print(f"  剪枝通过率: {prune_pass_rate*100:.2f}%")
print(f"  剪枝后分子数: {after_prune:,.0f}")

# 时间计算
# 剪枝时间（所有组合都要判断）
prune_total = total_combinations * (prune_time / 1_000_000)  # 转换为秒
print(f"\n⏱️ 剪枝阶段:")
print(f"  每个组合剪枝时间: {prune_time:.1f}微秒")
print(f"  剪枝总时间: {prune_total/3600:.2f}小时")

# 通过剪枝的分子处理时间
smiles_total = after_prune * (smiles_time / 1000)  # 转换为秒
mol_total = after_prune * (mol_time / 1000)
fp_total = after_prune * (fp_time / 1000)

print(f"\n⏱️ 通过剪枝的分子处理:")
print(f"  SMILES生成: {smiles_total/3600:.2f}小时")
print(f"  分子创建: {mol_total/3600:.2f}小时")
print(f"  指纹生成: {fp_total/3600:.2f}小时")

total_time = prune_total + smiles_total + mol_total + fp_total
print(f"\n⏱️ 总计时间: {total_time/3600:.2f}小时")

# 并行加速估算
print(f"\n⚡ 并行加速 (8核):")
print(f"  理想情况: {total_time/3600/8:.2f}小时")
print(f"  实际情况(考虑开销): {total_time/3600/8*1.2:.2f}小时")

# ============================================
# 建议
# ============================================
print("\n" + "="*50)
print("建议")
print("="*50)

if total_time/3600 < 24:
    print("✅ 可以在本地运行全量生成器")
    print("   建议: 晚上开始跑，第二天早上收结果")
elif total_time/3600 < 48:
    print("⚠️ 需要1-2天，建议分批运行")
    print("   建议: 实现断点续传，分2-3次运行")
else:
    print("❌ 时间太长，建议先用采样估计")
    print("   建议: 先跑千分之一采样，再决定")