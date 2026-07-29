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

阶段 0–4 已完成：

- 干净的 `src/` 包结构；
- 冻结的《小王子》高频词协议资产；
- EEG manifest；
- 从旧项目迁移的 BrainVision 读取器；
- 不依赖旧 `target_id` 或旧文本 embedding 的 manifest 接口；
- 14,034 条词 occurrence 及字符跨度；
- 外层 5 折分组交叉验证与每折内层验证集；
- 句子 occurrence DF 与独立规范化上下文组 DF 双口径资格表；
- 协议资产、分组隔离、覆盖率和可重复性审计；
- 模型无关的 `ContextWordStore` 连续数组缓存契约；
- 基于完整短句的 MacBERT 最后四层上下文词缓存；
- 基于完整短句的 BGE-M3 ColBERT multi-vector 上下文词缓存；
- fast-tokenizer offset、特殊 token 隔离、缓存哈希和只读恢复审计。
- 冻结 fold 到 20,902 个有效 EEG view 的只读联结索引；
- 全局稳定的 8 被试索引与 247 词 Master 整数索引；
- 同时支持 MacBERT/BGE-M3 的变长 EEG—上下文词 Dataset；
- batch 内动态双序列 padding、bool mask 与长度契约；
- validation/test 上下文目标访问的代码级拒绝；
- Windows `num_workers=2` 下按 worker 懒建只读 memmap/cache reader；
- 保留时间维的非因果卷积 EEG sequence encoder；
- 精确 stride 4 的 mask/length 传播与 padding 严格清零；
- 逐时间点 LayerNorm 和可配置的 subject FiLM adapter；
- FP32、CUDA AMP、最大真实长度及多 worker 真实 batch smoke。

尚未实现：

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
python scripts/cache_macbert_context_words.py --smoke-only
python scripts/cache_macbert_context_words.py
python scripts/audit_context_word_cache.py --cache-dir data/cache/context_words/macbert_v1
python scripts/cache_bge_m3_context_words.py --smoke-only
python scripts/cache_bge_m3_context_words.py
python scripts/audit_context_word_cache.py --cache-dir data/cache/context_words/bge_m3_colbert_v1 --compare-cache-dir data/cache/context_words/macbert_v1
python scripts/audit_context_eeg_dataset.py
python scripts/smoke_context_eeg_dataloader.py --outer-fold 0 --role train --text-backend macbert --batch-size 4 --num-workers 0 --num-batches 3
python scripts/smoke_context_eeg_dataloader.py --outer-fold 0 --role train --text-backend bge_m3 --batch-size 4 --num-workers 2 --num-batches 3
python scripts/smoke_context_eeg_dataloader.py --outer-fold 0 --role validation --text-backend bge_m3 --batch-size 4 --num-workers 0 --num-batches 3
python scripts/smoke_context_eeg_dataloader.py --outer-fold 0 --role test --text-backend bge_m3 --batch-size 4 --num-workers 0 --num-batches 3
python scripts/audit_eeg_sequence_encoder.py
python scripts/smoke_eeg_sequence_encoder.py --outer-fold 0 --role train --batch-size 8 --num-workers 0 --device cuda --precision fp32 --num-batches 3
python scripts/smoke_eeg_sequence_encoder.py --outer-fold 0 --role train --batch-size 16 --num-workers 0 --device cuda --precision amp --num-batches 3
python scripts/smoke_eeg_sequence_encoder.py --outer-fold 0 --role train --batch-size 8 --num-workers 2 --device cuda --precision fp32 --num-batches 1 --config configs/models/eeg_sequence_conv_no_subject_v1.yaml
python scripts/smoke_eeg_sequence_encoder.py --outer-fold 0 --role train --batch-size 8 --num-workers 0 --device cuda --precision fp32 --num-batches 1 --maximum-length-batch
```

MacBERT 配置固定在
`configs/text/macbert_base.yaml`，模型与 tokenizer revision 均为
`a986e004d2a7f2a1c2f5a3edef4e20604a974ed1`。完整缓存位于
`data/cache/context_words/macbert_v1/`，其中词向量形状为
`[14034, 4, 768]`、dtype 为 `float32`；该目录由 `.gitignore` 排除。

BGE-M3 配置固定在 `configs/text/bge_m3_colbert.yaml`，使用
`FlagEmbedding==1.4.0` 的
`BGEM3FlagModel.encode(return_colbert_vecs=True)` 接口。模型与 fast
tokenizer 均固定 revision
`5617a9f61b028005a4858fdac845db406aefb181`。完整缓存位于
`data/cache/context_words/bge_m3_colbert_v1/`，词向量形状为
`[14034, 1024]`、dtype 为 `float16`；模型前向和字符重叠聚合均使用
`float32`，reader 可按需恢复为 `float32`。

项目不保存原始 EEG 二进制文件或大型文本模型缓存到 Git。EEG manifest
中的路径继续指向本地 ChineseEEG-2 数据集。

阶段 3 的 Sample/Batch 字段、访问边界、padding 约定和 worker 资源
生命周期见 [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md)。

阶段 4 的默认模型配置位于
`configs/models/eeg_sequence_conv_v1.yaml`，无 subject adapter 消融配置位于
`configs/models/eeg_sequence_conv_no_subject_v1.yaml`。encoder 只输出
`[B, T', 256]` EEG 时间序列、bool mask 和精确长度，不执行整句池化。
