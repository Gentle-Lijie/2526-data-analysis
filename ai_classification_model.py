import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import warnings
from tqdm import tqdm
import joblib
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
df = pd.read_csv('人工智能数据9.13.csv')

print("数据集形状:", df.shape)
print("目标变量分布:")
print(df['AI Impact Level'].value_counts())

# 分离特征和目标
X = df.drop('AI Impact Level', axis=1)
y = df['AI Impact Level']

# 编码目标变量
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 处理分类特征
categorical_cols = ['Job Title', 'Industry', 'Job Status', 'Required Education']
numerical_cols = ['Median Salary (USD)', 'Experience Required (Years)', 'Job Openings (2024)',
                  'Projected Openings (2030)', 'Remote Work Ratio (%)', 'Automation Risk (%)', 'Gender Diversity (%)']

# OneHot编码分类特征
ohe = OneHotEncoder(sparse_output=False, drop='first')
X_cat = ohe.fit_transform(X[categorical_cols])
X_cat_df = pd.DataFrame(X_cat, columns=ohe.get_feature_names_out(categorical_cols))

# 合并数值和编码后的分类特征
X_processed = pd.concat([X[numerical_cols].reset_index(drop=True), X_cat_df], axis=1)

# 标准化数值特征
scaler = StandardScaler()
X_processed[numerical_cols] = scaler.fit_transform(X_processed[numerical_cols])

# 多维度分析：相关性矩阵
corr_matrix = X_processed[numerical_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('数值特征相关性矩阵')
plt.tight_layout()
plt.savefig('correlation_matrix.png')

# PCA分析
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap='viridis', alpha=0.5)
plt.colorbar(scatter, ticks=[0,1,2], label='AI Impact Level')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA可视化 (按AI影响水平着色)')
plt.tight_layout()
plt.savefig('pca_visualization.png')

# 特征选择：使用随机森林特征重要性
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_processed, y_encoded)

# 获取特征重要性
feature_importances = pd.DataFrame({
    'feature': X_processed.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n前10个重要特征:")
print(feature_importances.head(10))

# 可视化特征重要性
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importances.head(20))
plt.title('特征重要性 (前20个)')
plt.tight_layout()
plt.savefig('feature_importance.png')

# 选择前k个重要特征
k = 50
top_features = feature_importances.head(k)['feature'].tolist()
X_selected = X_processed[top_features]

# 划分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# 训练梯度提升模型
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.2]
}

print("开始网格搜索超参数...")
model = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', verbose=2)
model.fit(X_train, y_train)

print(f"最佳参数: {model.best_params_}")

# 预测
y_pred = model.predict(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"\n梯度提升模型准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('混淆矩阵')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.tight_layout()
plt.savefig('confusion_matrix.png')

# 保存模型
joblib.dump(model.best_estimator_, 'ai_impact_gb_model.pkl')
print("\n梯度提升模型已保存为 ai_impact_gb_model.pkl")

# 额外分析：按行业分析AI影响
industry_impact = df.groupby('Industry')['AI Impact Level'].value_counts(normalize=True).unstack()
industry_impact.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title('按行业划分的AI影响水平分布')
plt.ylabel('比例')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('industry_impact.png')

print("\n分析完成！生成的图片文件：correlation_matrix.png, pca_visualization.png, feature_importance.png, confusion_matrix.png, training_history.png, industry_impact.png")

# 加载数据
df = pd.read_csv('人工智能数据9.13.csv')

# 显示数据基本信息
print("数据集形状:", df.shape)
print("目标变量分布:")
print(df['AI Impact Level'].value_counts())

# 分离特征和目标
X = df.drop('AI Impact Level', axis=1)
y = df['AI Impact Level']

# 编码目标变量
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 处理分类特征
categorical_cols = ['Job Title', 'Industry', 'Job Status', 'Required Education']
numerical_cols = ['Median Salary (USD)', 'Experience Required (Years)', 'Job Openings (2024)',
                  'Projected Openings (2030)', 'Remote Work Ratio (%)', 'Automation Risk (%)', 'Gender Diversity (%)']

# OneHot编码分类特征
ohe = OneHotEncoder(sparse_output=False, drop='first')
X_cat = ohe.fit_transform(X[categorical_cols])
X_cat_df = pd.DataFrame(X_cat, columns=ohe.get_feature_names_out(categorical_cols))

# 合并数值和编码后的分类特征
X_processed = pd.concat([X[numerical_cols].reset_index(drop=True), X_cat_df], axis=1)

# 特征选择：使用随机森林特征重要性
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_processed, y_encoded)

# 获取特征重要性
feature_importances = pd.DataFrame({
    'feature': X_processed.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n前10个重要特征:")
print(feature_importances.head(10))

# 可视化特征重要性
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importances.head(20))
plt.title('特征重要性 (前20个)')
plt.tight_layout()
plt.savefig('feature_importance.png')
# plt.show()  # 移除以避免阻塞

# 选择前k个重要特征 (这里选择前50个)
k = 50
top_features = feature_importances.head(k)['feature'].tolist()
X_selected = X_processed[top_features]

# 划分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# 训练GradientBoosting模型
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('混淆矩阵')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
# plt.show()  # 移除以避免阻塞

# 保存模型 (使用joblib)
import joblib
joblib.dump(model, 'ai_impact_model.pkl')
print("\n模型已保存为 ai_impact_model.pkl")