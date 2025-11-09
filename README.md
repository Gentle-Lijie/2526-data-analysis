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
4. 特征选择（使用多线程RandomForest）
5. 模型训练（XGBoost with GPU加速）
6. 评估和可视化

## 性能优化
- **多线程处理**：RandomForest和GridSearchCV使用并行计算
- **GPU加速**：XGBoost自动检测并使用CUDA GPU加速（如果可用）
- **优化参数**：使用预调优的参数以提高训练速度

## 输出
- 控制台输出：数据集信息、特征重要性、模型准确率等
- 生成的图片文件：

### 1. `correlation_matrix.png` - 数值特征相关性矩阵
**图表说明**: 显示数值特征之间的皮尔逊相关系数热力图
**X轴/Y轴**: 数值特征名称（Median Salary, Experience Required, Job Openings等）
**颜色含义**:
- 红色（正相关）: 特征值同时增加或减少
- 蓝色（负相关）: 一个特征增加时另一个特征减少
- 颜色深浅: 相关强度（深色=强相关，浅色=弱相关）
**数值范围**: -1到+1，绝对值越大相关性越强

### 2. `pca_visualization.png` - PCA主成分分析可视化
**图表说明**: 使用主成分分析（PCA）将高维数据降维到2D空间展示
**X轴**: 第一主成分（解释最多方差的方向）
**Y轴**: 第二主成分（解释第二多方差的方向）
**颜色**: AI影响水平（High=高影响，Moderate=中等影响，Low=低影响）
**点分布含义**: 相似的工作岗位在图中距离较近，不同AI影响水平的工作分布在不同区域

### 3. `feature_importance.png` - 特征重要性条形图
**图表说明**: 显示各特征对AI影响水平预测的重要性
**X轴**: 重要性得分（0-1之间，值越大越重要）
**Y轴**: 特征名称
**条形长度**: 特征的重要性程度
**排序**: 按重要性降序排列，最重要的特征在顶部

### 4. `confusion_matrix.png` - 混淆矩阵
**图表说明**: 显示模型预测结果与实际标签的对比
**行（Y轴）**: 实际标签（真实AI影响水平）
**列（X轴）**: 预测标签（模型预测的AI影响水平）
**单元格数值**: 该类别组合的样本数量
**对角线**: 正确预测的样本数
**非对角线**: 错误预测的样本数
**颜色深浅**: 数值大小（深色=数量多）

### 5. `industry_impact.png` - 按行业AI影响分布
**图表说明**: 堆叠条形图显示不同行业的AI影响水平分布
**X轴**: 行业名称
**Y轴**: 占比（0-100%）
**条形颜色**:
- 蓝色: Low（低影响）
- 绿色: Moderate（中等影响）
- 红色: High（高影响）
**条形高度**: 该行业的岗位总数
**颜色比例**: 每个行业内不同AI影响水平的岗位占比

- 模型文件：`ai_impact_xgb_model.pkl` (XGBoost模型，支持GPU加速)
- 分析报告：`AI_Impact_Analysis_Report.md` (详细的数据分析报告)

## 数据分析报告
详细的数据分析报告请查看：`AI_Impact_Analysis_Report.md`

报告包含：
- 关键特征向量分析
- AI影响水平预测模型评估
- 行业洞察和政策建议
- 未来研究方向
- 脚本使用中文标签，确保matplotlib支持中文字体（Windows上可能需要安装中文字体）。
- 自动检测GPU并使用CUDA加速（需要NVIDIA GPU和CUDA驱动）。
- 如果没有GPU，将自动回退到CPU模式。
- 运行时间约1-2分钟（GPU加速）或2-3分钟（CPU模式）。
- 生成的分析报告包含详细的特征重要性分析和行业洞察。

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