# 皂苷虚拟筛选项目

## 📋 项目概述
基于皂苷结构规则生成虚拟库（约4400万分子），筛选与训练集（624个分子）相似度 ≥0.30 的分子。

## 📊 数据集
- `ci0c00102_si_002.xlsx`原论文训练集
- `ci0c00102_si_002.xlsx`原论文测试集
- `data/success_samples_train.csv`: 624个训练集分子（原论文训练集-处理失败样本）
- `data/success_samples_test.csv`: 70个测试集分子（处理后）
- 已完成皂苷分类分析：453个真皂苷 (72.6%)，171个非皂苷 (27.4%)

## 🔬 最新发现
运行到35%进度（1540万分子）时，相似度≥0.3的分子数为0。分析发现：
- 真皂苷内部平均相似度: ~0.28
- 真皂苷 vs 非皂苷平均相似度: ~0.16
- 171个非皂苷严重拉低平均值，导致阈值0.3难以达到

## 🚀 脚本说明
- `scripts/stage2_gpu_similarity.py`: GPU加速相似度计算（支持float16精度，断点续传）
- `scripts/training_analysis.py`: 训练集皂苷分类及相似度分析

## 📈 建议方案
1. 过滤训练集，只保留453个真皂苷
2. 或使用加权平均（给非皂苷低权重）
3. 或将阈值降至0.25
运行方法：
1. 本地确认环境（test.environment.py, 01_checkhardware.py)
2. 本地生成筛选后的候选分子（02_generate_candidates_local.py）
3. Google Colab运行候选分子与训练集分子比较，找出相似者（stage2_gpu_similarity.py)
4. 其他文件用于分析训练集内部皂苷与其他分子的相似度（check_sanopin_in _trainingset.py, trainging_set_similarity_analysis.py) - 本地运行
5. HemodynamicDrug.ipynb为复现HD50回归实验的代码（Colab运行）
