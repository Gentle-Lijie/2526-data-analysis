# AI影响分类数据分析项目

## 项目描述
本项目使用Python进行数据分析，分析工作数据集中的特征与AI影响水平的相关性，并构建分类模型。

## 数据集
- `人工智能数据9.13.csv`: 包含工作信息和AI影响水平的数据集。

## 依赖包
安装依赖：
```
pip install -r requirements.txt
```

## 运行脚本
运行主分析脚本：
```
python ai_classification_model.py
```

脚本将执行以下步骤：
1. 数据加载和预处理
2. 特征工程（编码、标准化）
3. 多维度分析（相关性、PCA、可视化）
4. 特征选择
5. 模型训练（梯度提升分类器）
6. 评估和可视化

## 输出
- 控制台输出：数据集信息、特征重要性、模型准确率等
- 生成的图片文件：
  - `correlation_matrix.png`: 数值特征相关性矩阵
  - `pca_visualization.png`: PCA可视化
  - `feature_importance.png`: 特征重要性条形图
  - `confusion_matrix.png`: 混淆矩阵
  - `industry_impact.png`: 按行业AI影响分布
- 模型文件：`ai_impact_gb_model.pkl`

## 注意事项
- 脚本使用中文标签，确保matplotlib支持中文字体（Windows上可能需要安装中文字体）。
- 网格搜索可能耗时较长，可以调整参数减少时间。
- 数据集较大，运行时间约几分钟到几十分钟。

## 迁移到Windows
1. 复制整个文件夹到Windows。
2. 创建新的虚拟环境：`python -m venv .venv`
3. 激活虚拟环境：`.venv\Scripts\activate`
4. 安装依赖：`pip install -r requirements.txt`
5. 运行脚本：`python ai_classification_model.py`

## 继任Copilot说明
- 数据集：`人工智能数据9.13.csv`
- 目标：AI Impact Level (High, Moderate, Low)
- 关键特征：薪资、自动化风险、性别多样性等
- 模型：GradientBoostingClassifier
- 准确率：约33%（可进一步优化）