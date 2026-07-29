# EEG Keyword Decoding

本项目研究以下问题：

> 能否通过完整短句上下文监督，从变长中文 EEG 序列中解码稳定的单词信息？

主训练轨道将短句 EEG 序列与上下文化词序列对齐，使用 Sinkhorn OT 学习弱监督词级对应，并在固定关键词词表上评估单词存在信息。

## 科学协议

- 主任务：短句上下文泛化。
- 外层评估：5 折分组交叉验证。
- 分组单位：同一短句 occurrence 的全部受试者 EEG，以及规范化后完全相同的重复短句。
- 内层验证：只从每个外层训练部分划分，用于早停、checkpoint 和超参数选择。
- 固定词表：Core、Main、Extended、Master 以及后续人物/地点/动作视图。
- Prototype：只能由相应外层 fold 的训练短句上下文表示构建。
- 测试：模型只接收 EEG 和冻结的训练集 prototype；测试文本只提供真值标签。

完整约束见 [docs/PROTOCOL.md](docs/PROTOCOL.md)。

## 当前阶段

阶段 0–1 已完成：

- 干净的 `src/` 包结构；
- 冻结的《小王子》高频词协议资产；
- EEG manifest；
- 从旧项目迁移的 BrainVision 读取器；
- 不依赖旧 `target_id` 或旧文本 embedding 的 manifest 接口；
- 14,034 条词 occurrence 及字符跨度；
- 外层 5 折分组交叉验证与每折内层验证集；
- 句子 occurrence DF 与独立规范化上下文组 DF 双口径资格表；
- 协议资产、分组隔离、覆盖率和可重复性审计。

尚未实现：

- BGE-M3 / MacBERT 上下文词缓存；
- EEG sequence encoder；
- Sinkhorn OT 与训练损失；
- 固定词表评估器。

后续实施顺序与验收条件见 [docs/ROADMAP.md](docs/ROADMAP.md)。

## 来源

可复用代码与数据协议迁移自：

```text
repository: language_decoding
commit: 5396c4ca8b9620cfdfba6a680e084fe003c58832
```

旧仓库只作为整句向量基线和数据来源；其 canonical `target_id`、整句检索协议和训练入口不属于本项目的任务身份。

## 本地检查

使用 `bm5060`：

```powershell
conda activate bm5060
python -m pytest -q
python scripts/audit_littleprince_hf_v1.py
python scripts/audit_nested_folds_v1.py
```

项目不保存原始 EEG 二进制文件或大型文本模型缓存。EEG manifest 中的路径继续指向本地 ChineseEEG-2 数据集。
