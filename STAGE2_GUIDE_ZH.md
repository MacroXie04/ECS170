# ECS 170 Stage 2：从数据到结果

## 这份代码在哪里

这份独立实现位于本聊天的 `outputs/ECS170`，分支为 `Zz_Wang`。
初次实现时未编辑桌面原仓库。随后按你的要求，将完整实现保存为本地 Git 提交，
并同步至桌面仓库的 `Zz_Wang` 分支；未推送 GitHub，也未合并到 `main`。
开始时原仓库唯一未提交内容是未跟踪的 `data/stage_2_data/`，已保留并复制到本副本。
远程 main 核验为 `8144e7514eacc8c10bfb650927724e87614c4254`；
初始模板远程现可访问，main 为 `465d116276dd3698f73139ca0b64062c379f3e7c`。
课程模板的基础类及 Stage 1 文件保留，Stage 2 使用相同的模块分工并继承基础类。

已找到作业说明、CSV、数据 ReadMe 和源代码模板；随后根据你的提示，
在 Desktop 找到 `Stage_2-5_Report_Template.pdf`，报告已按其中的章节和 10 号字要求制作。
报告仍标记为草稿，因为组名、四位成员信息与公开可访问的 Stage 2 源码地址需要补齐。
四人组是你的项目背景，不因 PDF 的组队描述被自动改写。

## 1. 先理解数据

老师的数据按行存储，没有标题行：

```text
label,feature1,feature2,...,feature784
```

实际核查：`train.csv` 为 60,000×785，`test.csv` 为 10,000×785。
第一列标签为 0–9，其他列为 0–255 的整数。
这里的 785 是 CSV 列数，模型输入是 **784**，因为标签不能作为输入特征。

`Dataset_Loader.load_file()` 做三件事：检查内容；把第一列拆成 `y`、其余列拆成 `X`；将特征除以 255。
`X` 使用 float32（适合神经网络计算），`y` 使用 int64（交叉熵的类别索引要求）。
`np.loadtxt` 不把第一行当标题，因此不会少读一条数据。
类别映射从训练集建立，测试集若出现未见标签则报错。

`/255` 是事先确定的缩放规则，没有从训练、验证或测试数据估计参数。
如果以后使用 StandardScaler 等方法，必须只在内部训练部分 fit，再对验证/测试 transform。

## 2. 训练、验证、测试各做什么

```text
老师 train.csv（60,000）
  ├─ 固定分层划分：内部训练 54,000 → 学习模型参数
  └─ 固定分层划分：内部验证  6,000 → 选轮数、选配置

选定配置和轮数 → 新建模型 → 用完整 60,000 条重新训练
老师 test.csv（10,000）→ 最终测试预测与指标
```

分层划分尽量保持各类比例，`random_state=42` 固定行号，所有实验共用。
这是从原训练集内部留验证集，没有重新混合老师的训练和测试文件，也没有 KFold。
数据审查时读取过测试文件以核验格式、行数和哈希；其预测表现没有参与调参。

验证也属于模型开发过程，所以多次比较后的验证成绩不等同于最终泛化成绩。
最终测试结果只用于报告，不据此增加配置或反复选择种子。

## 3. 一个 batch 如何通过模型

以基准配置为例：

```text
128 条样本的 X：[128, 784]
    ↓ Linear(784, 128)：每条样本变成 128 个可学习组合
    ↓ ReLU：将负数置零，使多层网络能学习非线性关系
    ↓ Linear(128, 10)
logits：[128, 10]，每行对应 10 个类别的原始评分
    ↓ CrossEntropyLoss(logits, y)
一个平均损失标量
    ↓ loss.backward()：自动算各参数对损失的梯度
    ↓ optimizer.step()：沿减小损失的方向更新参数
```

`optimizer.zero_grad()` 放在每批计算前，防止上批梯度意外累加。
最后一批可能不足 128 条，代码按实际样本数计算平均指标。
`argmax(1)` 对每行选最大评分的位置，得到预测类别。
交叉熵直接接收 logits，内部完成 log-softmax，模型最后不再叠加 Softmax。
平方误差对照则明确先 softmax，再计算预测概率与 one-hot 真值的平方距离之和均值。
两种损失的数值尺度不同，不能按损失绝对值跨实验排名。

`fit()` 表示完整训练过程。`train()` 是 PyTorch 的模式开关，训练时启用 Dropout；
`eval()` 在验证/预测时关闭 Dropout；`no_grad()` 表示不用记录用于反向传播的计算图。
原模板 `train(X,y)` 覆盖模式开关的问题在 Stage 2 中已经消除。
每轮训练/验证指标均在该轮结束的同一组参数上以 eval 模式计算，方便比较。

## 4. 每个文件的职责

| 文件 | 作用 |
| --- | --- |
| `local_code/stage_2_code/Dataset_Loader.py` | CSV 检查、首列标签、缩放、训练类别映射、源文件哈希 |
| `local_code/stage_2_code/Method_MLP.py` | 可配置网络、fit、最佳验证模型恢复、批量预测 |
| `local_code/stage_2_code/Setting.py` | 固定内部划分、种子、新建模型、验证选型、全量重训和最终测试 |
| `local_code/stage_2_code/Evaluate.py` | 多分类 Accuracy、PRF、逐类指标和混淆矩阵 |
| `local_code/stage_2_code/Result_Saver.py` | JSON/CSV/NPZ/模型保存，防止覆盖已有实验 |
| `local_code/stage_2_code/Plot.py` | 从保存记录绘图，不重新训练 |
| `script/stage_2_script/script_mlp.py` | 命令行入口，预先声明七组配置 |
| `tests/test_stage2.py` | 数据、梯度、模式、指标和存取一致性检查 |
| `requirements-stage2.txt` | 本次实验所用核心依赖版本 |

`__init__.py` 是让目录作为 Python 包导入的文件，原有文件无需修改。
运行脚本自动定位仓库根目录，不需要更改 Stage 1 的导入路径。

## 5. 如何运行

在 `outputs/ECS170` 目录打开终端。新环境建议 Python 3.9–3.12：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-stage2.txt
python -m unittest discover -s tests -v
```

第一次复现实验应使用一个新结果目录，以免覆盖本次真实结果：

```bash
python script/stage_2_script/script_mlp.py baseline --output result/my_reproduction
python script/stage_2_script/script_mlp.py compare --output result/my_reproduction
python script/stage_2_script/script_mlp.py final --output result/my_reproduction
python script/stage_2_script/script_mlp.py plot --output result/my_reproduction
```

`compare` 会复用已有且配置相同的 baseline，其余实验各自重新初始化。
一旦已生成 selection.json，不要向同一实验目录追加配置；新的研究应使用新目录。
`final` 目录一旦存在便拒绝覆盖；不应用删除它的方式反复查看测试表现。
可用 `--data /你的数据目录` 指定 CSV 所在位置；CPU 默认两线程。
CUDA/MPS 是可选项，本次只验证了 CPU；不同设备和库版本可能有数值差异。

本聊天实际运行使用桌面 `Project_/.venv/bin/python`（Python 3.9.6），
绘图依赖独立安装在聊天的 `work/python_packages`，没有修改原课程环境。
因此你在自己终端复现时使用上面的独立 venv 安装步骤最清楚。
本次绘图使用 `PYTHONPATH=work/python_packages` 后调用课程解释器，路径相对聊天根目录。

## 6. 实验如何比较

所有配置最多 25 轮，batch_size=128，初始化和批次种子=42，验证划分种子=42。
验证损失连续 5 轮未改善便停止，每个配置恢复自己验证损失最低的那轮。
配置间按恢复后模型的验证 Accuracy 排名，平手依次比较 macro F1 和名称。
验证损失用于配置内部选轮数，配置之间不会拿交叉熵与平方误差的损失直接比较。

基准为 `[128]`、Adam、lr=0.001、交叉熵。
深度比较为 `[128]`、`[256,128]`、`[256,128,64]`，这里说的是隐藏层数，另有输出层。
注意深度比较同时改变了参数量，不能把差异全部解释为“层数的因果效果”。
SGD 对照使用 momentum=0.9、lr=0.05，与 Adam 的比较包含相应学习率设置。
Dropout=0.2 和 weight_decay=0.0001 分别对照无正则的双隐藏层配置。
单种子、小规模配置比较只能说明本次运行，不能声称统计显著或全局最优。

## 7. 结果文件怎么看

每组 `record.json` 含配置、种子、环境、数据/源代码哈希、所选轮数和完整指标。
`history.csv` 每行一轮；准确率上升意味着预测正确比例升高，损失下降表示目标函数改善。
训练持续改善、验证变差通常提示过拟合；曲线波动不等于代码一定出错。
`learning_curve.png` 虚线为该配置所选轮数；后续几轮用于判断早停，未被当成最终权重。
最终重训曲线只有训练线，没有虚构一条验证线或逐轮测试线。

`predictions.npz` 保存真值和预测；验证实验行号在 `split.npz` 中。
`model.pt` 含 state_dict、配置、输入维度、类别和预处理规则，可重新构造模型。
例如用 `torch.load(path, map_location='cpu', weights_only=True)` 读取可信的本地检查点，
再用 `Method_MLP(checkpoint['input_dim'], len(checkpoint['classes']), checkpoint['config'])`
创建模型并 `load_state_dict(checkpoint['state_dict'])`；预测输入仍须除以 255。

Accuracy 是全部预测中正确的比例。
Precision 看预测成某类的样本有多少正确；Recall 看该类真样本有多少被找回；F1 综合两者。
macro 对每类等权，weighted 按真值样本数加权；本任务完整单标签多分类的 micro PRF 与 Accuracy 一致。
混淆矩阵行是真值、列是预测；对角线是正确预测，非对角线显示常见错误。

最终实测结果和讨论放在 `STAGE2_RESULTS_ZH.md`，英文报告草稿单独保存。
报告限制 5 页；详细逐类表、所有模型及完整训练记录可留在代码结果附件中。

## 8. 已验证的训练正确性

四项自动检查验证了：不误读首列或标题、未见测试标签会报错；
梯度存在且有限、小数据损失能下降并达到完全拟合；
同种子生成相同初始参数但对象不共享存储、eval 递归关闭训练模式；
检查点预测一致，以及没有被预测到的类别仍被计入 macro 指标。
完整实验还检查每组划分一致、训练/验证不交叠、保存模型与预测一致。

技术参考：PyTorch CrossEntropyLoss 文档与 scikit-learn precision_recall_fscore_support 文档。
优先对应本次依赖版本，而不是假定最新版行为完全一致。
