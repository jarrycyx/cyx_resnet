## 基于ResNet152的图像分类

### 训练流程
1. 分割数据集（SplitDataset.py）
2. 设置数据集位置、使用gpu设备、分类数目等参数（TrainBagging.py）
3. 开始训练（MultiThreadTrain.py）
4. 验证bagging准确率（ValidBagging.py）
5. 保存测试集分类结果到csv（SaveTestResults.py）
6. 绘制训练曲线（PlotCurves.py）

### 文件目录说明

./Utils/ 训练所需的工具类/函数

./Utils/DataUtils 定义生成数据集的方式

./Utils/LogUtils 实时存储日志的工具类

./q1_data/ 提供的数据集和label

./loss/ 损失函数

./LegacyCodes 暂时不再使用的代码

./bagging 集成学习所需工具

./bagging/bag1.npy 存储“属于第一个训练集”的训练集（train.npy）中图片序号的查找表

./bagging/val.npy 存储“属于验证集”的训练集（train.npy）中图片序号的查找表（因为没有单独的验证集，从训练集中分一部分用于验证）

./bagging/SplitDataset.py 通过生成查找表的方式分割数据集为训练集1、训练集2、训练集3、验证集，训练集之间数据有重合但不完全相同

./bagging/MergeResults.py 分开训练之后，用集成学习（bagging）的方式“投票”选出最优结果

./TrainBagging.py 封装用于训练一个网络的函数和参数

./ValidBagging.py 用于训练完成后验证bagging最终准确率

./SaveTestResults.py 训练完成后保存测试集分类结果到csv文件

./MultiThreadTrain.py **顶层文件**。为了防止CPU的阻塞影响GPU多卡运算的效率，使用多线程训练

./PlotCurves.py 从log读取数据，绘制训练过程的loss和accuracy变化曲线


### 准确率
#### 20分类：
Merge Accuracy: 0.9060

Bag 0 Accuracy: 0.8862 

Bag 1 Accuracy: 0.8818 

Bag 2 Accuracy: 0.8860 

#### 100分类
Merge Accuracy: 0.8362 

Bag 0 Accuracy: 0.8042 

Bag 1 Accuracy: 0.8108 

Bag 2 Accuracy: 0.8068 

