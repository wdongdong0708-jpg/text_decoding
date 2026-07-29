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

阶段 0–4、阶段 5A/5B 及阶段 6A 已完成：

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
- backend 显式、metadata 校验的 MacBERT/BGE-M3 上下文词投影；
- MacBERT 共享可学习四层 scalar mix，BGE-M3 线性投影基线；
- masked cosine cost 与 balanced、log-domain、内部 FP32 的 Sinkhorn；
- 均匀有效时间/词边缘、严格零 padding transport mass 和边缘审计；
- 按实际列质量归一化的词条件 EEG transport pooling；
- 两种文本 backend 的 FP32/AMP、subject adapter 开关和最大长度真实
  inner-train batch 前后向 smoke。
- 覆盖全部 14,034 个词 occurrence 的稳定 `surface_type_index`，以及由
  `sentence_group_id + word_position + surface_form` 定义的
  `context_token_group_index`；
- expected transport cost、对称多正例 context-token InfoNCE 和固定
  Master 空间 prototype 分类三项损失；
- context-token 中同 lexical type、不同 context group 的假负例硬屏蔽，
  以及默认逐样本等权的 token reduction；
- 只接受当前 outer fold inner-train、按句内重复词→规范化句组→不同
  context group 等权聚合的 247×256 prototype builder；
- prototype bank 的可用 mask、保存/加载、文件哈希以及 fold/backend/
  cache/projector provenance 硬校验；
- MacBERT/BGE-M3、FP32/AMP、无 Master 词、重复词、多受试者同句 view、
  subject adapter 开关和最大长度真实三项损失前后向 smoke。
- 只接收 inner-train/inner-validation 的单 fold trainer，API 中不存在
  test Dataset/DataLoader；
- 每 epoch 训练前和验证前各刷新一次、与当前 projector SHA256 绑定的
  detached train-only prototype bank；
- validation 只用 EEG、EEG mask、subject index 和 train-only prototype，
  使用减去 `log(T)` 的 masked log-mean-exp 得到固定 247 列 score；
- EEG-view → sentence-occurrence → normalized-context-group 三级等权聚合，
  逐词 continuous-score AUPRC 及显式 NaN 原因；
- 唯一 early-stopping 指标
  `validation/core/context_group/macro_auprc`；
- 包含模型、prototype、optimizer/scheduler/GradScaler、RNG、DataLoader
  generator、配置和数据资产哈希的 `last.pt`/`best.pt`；
- synthetic tiny overfit、checkpoint resume、Windows multi-worker
  可重复性，以及 BGE-M3/MacBERT 的 fold 0 AMP full-validation smoke。

尚未启动或实现的是正式外层 5 折长训练、OOF 测试评估、bootstrap 和
阶段 7 消融；阶段 6A 产物均明确标记为不可用于科学报告。

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
python scripts/audit_masked_sinkhorn.py
python scripts/smoke_context_ot.py --outer-fold 0 --role train --text-backend macbert --batch-size 4 --device cuda --precision fp32 --epsilon 0.05 --iterations 50 --num-batches 3
python scripts/smoke_context_ot.py --outer-fold 0 --role train --text-backend bge_m3 --batch-size 4 --device cuda --precision amp --epsilon 0.05 --iterations 50 --num-batches 3
python scripts/smoke_context_ot.py --outer-fold 0 --role train --text-backend bge_m3 --batch-size 4 --device cuda --precision fp32 --epsilon 0.05 --iterations 50 --num-batches 1 --maximum-length-batch
python scripts/build_train_only_prototypes.py --device cpu
python scripts/audit_prototype_bank.py --outer-fold 0 --text-backend macbert
python scripts/audit_prototype_bank.py --outer-fold 0 --text-backend bge_m3
python scripts/smoke_context_ot_losses.py --outer-fold 0 --role train --text-backend macbert --batch-size 4 --device cuda --precision fp32 --num-batches 3
python scripts/smoke_context_ot_losses.py --outer-fold 0 --role train --text-backend bge_m3 --batch-size 4 --device cuda --precision amp --num-batches 3
python scripts/smoke_context_ot_losses.py --outer-fold 0 --role train --text-backend bge_m3 --batch-size 4 --device cuda --precision amp --num-batches 1 --scenario maximum_length
python scripts/synthetic_tiny_overfit.py --steps 25 --seed 42
python scripts/smoke_fold_training.py --outer-fold 0 --text-backend bge_m3 --epochs 2 --max-train-batches 10 --batch-size 8 --precision amp --device cuda
python scripts/smoke_fold_training.py --outer-fold 0 --text-backend macbert --epochs 2 --max-train-batches 10 --batch-size 8 --precision amp --device cuda
python scripts/inspect_checkpoint.py outputs/training/<backend>/outer_fold_0/<run_id>/checkpoints/last.pt
python scripts/validate_checkpoint.py outputs/training/<backend>/outer_fold_0/<run_id>/checkpoints/last.pt
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

阶段 5A 的文本投影配置位于
`configs/text/macbert_projection_v1.yaml` 和
`configs/text/bge_m3_projection_v1.yaml`；OT 与 pooling 配置位于
`configs/ot/`。该阶段只公开 expected transport cost 和独立 entropy
诊断，不把未经长度偏置分析的 entropy-regularized objective 固定为训练
loss，也不读取 validation/test context targets。

阶段 5B 的损失配置位于
`configs/losses/context_ot_three_loss_v1.yaml`，train-only prototype
配置位于 `configs/prototypes/train_only_group_balanced_v1.yaml`。默认总
损失为三项非负加权和；OT-only 只允许作为表示塌缩风险明确的消融。原型
运行产物位于 gitignored 的 `data/cache/prototypes/`，每次 projector
更新后必须显式重建，最终 checkpoint 也必须重建并冻结一次。

阶段 6A 的训练配置位于 `configs/training/`。每轮固定执行
`refresh(train-only) → train → refresh(updated projector) → text-free
validation → checkpoint → early stopping`。输出写入 gitignored 的
`outputs/training/<backend>/outer_fold_<k>/<unique_run_id>/`，不会静默覆盖
旧运行。训练入口不创建 outer-test Dataset；正式 outer-test 推理属于后续
独立命令。
