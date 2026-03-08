#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试环境是否配置正确
"""

import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import multiprocessing as mp
import psutil

def test_environment():
    print("="*50)
    print("环境测试报告")
    print("="*50)
    
    # Python版本
    print(f"\n📌 Python版本: {sys.version}")
    
    # 测试numpy
    try:
        arr = np.array([1,2,3])
        print("✅ numpy: 正常")
    except:
        print("❌ numpy: 异常")
    
    # 测试pandas
    try:
        df = pd.DataFrame({'a': [1,2,3]})
        print("✅ pandas: 正常")
    except:
        print("❌ pandas: 异常")
    
    # 测试RDKit
    try:
        mol = Chem.MolFromSmiles('CCO')
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        print("✅ RDKit: 正常")
        print(f"   测试分子: {Chem.MolToSmiles(mol)}")
        print(f"   指纹长度: {len(fp)}")
    except Exception as e:
        print(f"❌ RDKit: 异常 - {e}")
    
    # 测试多进程
    try:
        cpu_count = mp.cpu_count()
        print(f"✅ 多进程: 正常 (CPU核心数: {cpu_count})")
    except:
        print("❌ 多进程: 异常")
    
    # 内存信息
    memory = psutil.virtual_memory()
    print(f"\n📊 系统资源:")
    print(f"   总内存: {memory.total / 1024**3:.1f} GB")
    print(f"   可用内存: {memory.available / 1024**3:.1f} GB")
    
    print("\n" + "="*50)
    print("环境测试完成")
    print("="*50)

if __name__ == "__main__":
    test_environment()