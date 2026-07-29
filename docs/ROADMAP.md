# Implementation Roadmap

本路线图以“外层 5 折分组交叉验证 + 内层验证集”的短句上下文泛化协议为唯一主轨道。每一阶段先冻结数据契约和测试，再进入下一阶段。

## 阶段 0：新仓库与可复用迁移

状态：已完成。

- 建立独立 `src/` 项目，不继承旧项目的整句 `target_id` 任务身份。
- 原样迁移高频词协议、短句标签、EEG manifest 和 BrainVision 读取器。
- 记录上游 commit、源文件 SHA256 和迁移边界。
- 不迁移旧 checkpoint、输出、模型缓存和悬空的 MEG/speech 代码。

## 阶段 1：词边界与评估划分

状态：已完成。

- 将冻结分词映射为 `[char_start, char_end)`，生成 14,034 条词 occurrence。
- 排除 27 个章节标题和 1 个无有效词短句，保留 2,809 个有效短句。
- 用完整规范化文本 SHA256 合并重复短句，形成 2,575 个原子组。
- 生成外层 5 折测试集；每个有效短句恰好成为一次 out-of-fold 测试样本。
- 每个外层训练部分再按同一组约束划出约 12.5% 作为内层验证集。
- 同时审计短句 DF 与独立上下文组 DF；固定候选词表不随 fold 缩减。

验收：生成脚本跨独立 Python 进程得到相同 SHA256；同一文本组不跨 role；Core 33 词在每折满足上下文组 `20/5/5`。

## 阶段 2：上下文词表示缓存

状态：2A MacBERT baseline 与 2B BGE-M3 ColBERT 缓存均已完成。

1. 定义连续数组缓存契约：词向量、句子 offsets、词表面形式、字符跨度、keyword ID、模型元数据。
2. 先实现 MacBERT baseline：缓存最后 4 层 `[total_words, 4, 768]`，训练时学习 scalar mix。
3. 再实现 BGE-M3 主模型：提取完整短句的 ColBERT token vectors，按字符 offset 聚合为 `[total_words, 1024]`。
4. 对全部 14,034 条词检查至少一个有效 subtoken、无特殊 token 污染、聚合可重复。
5. 缓存元数据记录模型和 tokenizer 的精确 revision、源标签 SHA256、聚合规则、dtype 和输出哈希。

验收：MacBERT 与 BGE-M3 各完成小样本人工核对及全量 offset 审计；缓存只由冻结文本生成，不包含 split 或 EEG 信息。

### 阶段 2A 完成记录

- 模型和 fast tokenizer：`hfl/chinese-macbert-base`，二者均固定 revision
  `a986e004d2a7f2a1c2f5a3edef4e20604a974ed1`。
- 输入始终为完整规范化短句；没有逐词孤立编码。
- 缓存最后四个 encoder hidden states，即第 9、10、11、12 层。
- subtoken 聚合规则为字符重叠长度加权平均；`[CLS]`、`[SEP]`、
  padding 和不属于冻结词跨度的标点不进入词向量。
- `ContextWordStore` 使用连续 `.npy` 数组保存词向量、句子索引和
  offsets、occurrence ID、词位置、表面形式、字符跨度与 keyword ID。
- 全量结果为 2,809 个短句、14,034 个词，词向量
  `[14034, 4, 768] float32`。
- 独立全量 offset 审计失败数为 0，特殊 token 重叠数为 0，同一
  tokenizer token 跨冻结词共享数为 0，最大输入长度为 28 tokens。
- 人工核对覆盖普通标点、多字词、重复词、书名号、引号/未知标点及
  最长短句；缓存 reader 可按 `text_embedding_idx` 恢复
  `[n_word, 4, 768]`。
- metadata 精确记录模型/tokenizer revision、输入与配置 SHA256、
  聚合规则、运行库版本、dtype、shape、文件大小及每个数组 SHA256。
- 缓存目录 `data/cache/context_words/macbert_v1/` 不包含 fold、split
  或 EEG 信息，并由 `.gitignore` 排除。

### 阶段 2B 完成记录

- 模型与 fast tokenizer：`BAAI/bge-m3`，二者均固定 revision
  `5617a9f61b028005a4858fdac845db406aefb181`；模型 hidden size 与
  ColBERT dimension 均为 1024。
- 使用 `FlagEmbedding==1.4.0` 的公开接口
  `BGEM3FlagModel.encode(return_dense=False, return_sparse=False,
  return_colbert_vecs=True)`；没有使用 dense 整句向量或 sparse 词权重。
- 通过 FlagEmbedding 源码和直接前向实测确认：公开 ColBERT 第 `j` 行
  对应 fast tokenizer 第 `j+1` 个 token；模型去除首个 CLS、保留 SEP，
  公开接口再去除 padding。10 条代表短句的公开接口与直接前向最大误差为
  0（容差 `1e-6`）。
- 输入始终为完整规范化短句。模型前向和字符重叠加权聚合使用
  `float32`，写盘前转换为 `float16`；reader 可恢复为 `float32`。
- 全量结果为 2,809 个短句、14,034 个词，词向量
  `[14034, 1024] float16`；向量文件 SHA256 为
  `f8b6f93e7a7fca7716a2276c1d187359c50b9239968e09869f0ddc6219c0704b`。
- 全量 offset 失败、特殊 token 重叠、空向量和非有限向量数量均为 0；
  3,625 个词使用多个 subtoken（25.83%），最大输入长度为 25 tokens，
  发现 20 个 `<unk>` token。
- BGE-M3 SentencePiece 有 1,408 个 tokenizer token 跨越两个或更多冻结词
  的字符边界。实现不改变冻结词序列，而是按每个 occurrence 的正长度字符
  重叠独立聚合；该现象不是特殊 token 或标点污染，并在 metadata 中显式记录。
- BGE-M3 与 MacBERT 缓存的 `text_embedding_idx`、sentence offsets、
  occurrence ID、词位置、表面形式、字符跨度与 keyword ID 完全一致，
  可由同一个 `ContextWordStore` 读取。
- 缓存只由冻结文本生成，不包含 fold、split 或 EEG 信息；后续训练仍必须
  用 fold mask 阻止 outer-test 表示进入训练、prototype 或 checkpoint 选择。

## 阶段 3：变长双序列数据管线

状态：已完成。

- Dataset 按 `text_embedding_idx` 联结 EEG、fold role、词 occurrence 和上下文词缓存。
- Collate 输出 `eeg/eeg_mask/context_words/word_mask/keyword_ids`。
- 一个短句的全部受试者 EEG view 只共享文本缓存，不重复进入 prototype 统计。
- 空序列在数据构建时被排除，不在训练循环中临时跳过。

验收：不同 EEG 长度和词数可同批训练；padding 改变不影响有效区域；fold role 无交叉。

### 阶段 3 完成记录

- 冻结 EEG manifest 共 21,110 行；28 个排除短句对应 208 个 view，
  2,809 个有效短句对应 20,902 个 view，每句 6–8 个 view。
- `SplitViewIndex` 不读取 EEG 数据，按 `text_embedding_idx` 联结每折
  `train/validation/test`，强制同一规范化文本组 role 原子性并验证每个
  有效短句恰好 outer-test 一次。
- fold 0–4 的 train view 数依次为
  14,650/14,630/14,624/14,641/14,620；validation 为
  2,086/2,090/2,093/2,076/2,098；test 为
  4,166/4,182/4,185/4,185/4,184。
- 全局 subject index 固定为 `sub-01`–`sub-08 -> 0`–`7`，不随 fold
  重排。Master keyword index 按冻结 rank 建立 247 维共享索引空间，
  Core/Main/Extended 大小为 33/64/100，非 Master occurrence 为 `-1`。
- `ContextEEGDataset` 的单位是一个句子 occurrence 的一个受试者 EEG
  view；EEG 只在 `__getitem__` 中按窗口懒读，不标准化、不裁剪、
  不重采样、不增强。
- train 可通过同一个 `ContextWordStore` 接口读取 MacBERT
  `[N,4,768]` 或 BGE-M3 `[N,1024]`，统一恢复为 `float32`。
  缓存数组 SHA256、metadata SHA256 和冻结 occurrence 顺序均在
  Dataset 初始化时验证。
- validation/test 无法绑定 context store，也无法请求
  `include_context_targets=True`；其 batch 中 `context_words=None`，
  不使用全零伪目标。
- Collate 只 padding 当前 batch 的 EEG 时间轴与词轴，生成
  `eeg_mask/word_mask` 和严格一致的 lengths；EEG、词位置、关键词索引
  与字符串元数据保持输入顺序。
- 200 个本地 BrainVision recording 全部为 128 通道、250 Hz；全部
  manifest 窗口在文件边界内。真实数据的 MacBERT/BGE-M3、validation、
  test 各完成 3 batch smoke test。
- Dataset pickle 时主动丢弃 ContextWordStore memmap 和 BrainVision
  reader 缓存；Windows spawn worker 各自按需重开只读资源，
  `num_workers=2` 真实数据 smoke test通过。

完整数据契约见 [DATA_PIPELINE.md](DATA_PIPELINE.md)。

## 阶段 4：EEG sequence encoder

状态：已完成。

- 从旧 `SimpleConvTimeAgg` 的卷积干线提炼可复用结构，但输出 `[B, T', 256]` 而非整句向量。
- 默认时间 stride 4，控制 16 GB 显存占用。
- 使用逐时间点 LayerNorm，避免 BatchNorm/普通 GroupNorm 跨 padding 时间位置统计。
- 保留 duration-only、mean-pooling 与旧整句向量基线的独立入口。

验收：mask 下采样正确；padding 不改变有效输出；无 NaN；正反向 smoke test 通过。

完成记录：

- 实现 1×1 shared channel projection、可选 subject FiLM、stride-2 temporal
  stem、四个 dilation 为 `[1, 2, 4, 8]` 的残差时间块、第二个 stride-2
  downsample 和 256 维逐时间点输出投影。
- 输入为 `[B, C, T]` EEG、连续前缀 bool mask 和全局 subject index；输出为
  `[B, T', 256]`、`[B, T']` bool mask 和 `[B]` 精确长度。
- 两次 same-padding stride-2 卷积给出
  `T' = ceil(ceil(T / 2) / 2) = ceil(T / 4)`；真实最大长度 1,297
  对应 325 帧。
- 每个卷积、归一化、激活、dropout 和残差步骤后重新清零无效位置；
  拒绝非连续 mask、空序列、过短输入和越界 subject index。
- 默认配置启用 8-subject FiLM；同时冻结无 subject adapter 的消融配置。
- 单元测试覆盖 padding/额外右 padding/无效值/批排列不变性、梯度、
  AMP、state-dict round trip 和真实 inner-train batch。
- RTX 5060 Ti 16 GB 上 batch 8 FP32、batch 16 AMP、最大长度 batch 和
  `num_workers=2` 真实数据 smoke 均通过。

## 阶段 5：Sinkhorn OT 与三项损失

状态：阶段 5A（文本投影 + masked balanced Sinkhorn OT）已完成；三项损失
与 train-only prototype builder 待下一阶段实现。

- masked、balanced、log-domain、内部 FP32 的 Sinkhorn。
- 均匀有效时间边缘与均匀有效词边缘，cosine cost。
- 实现 `OT-context + λ_token context-token + λ_lex prototype`。
- prototype 损失只覆盖达到 inner-train `group_df` 门槛的词；同词 occurrence 不互作负例。
- 第一版不加入关键词 BCE 主损失、词序代价或顺序损失。

验收：无效位置 transport mass 为零；行列边缘误差达标；极端相似度无 NaN；梯度可回传。

### 阶段 5A 完成记录

- MacBERT 投影从缓存 metadata 验证 `[word, encoder_layer, hidden]` 轴、
  `[9, 10, 11, 12]` 层顺序和 768 hidden size；全局共享的四层 logits
  以 0 初始化，经 softmax 得到均匀 scalar mix，再使用 LayerNorm 和
  `Linear(768, 256)`。
- BGE-M3 投影验证缓存为 ColBERT 上下文词向量及 1,024 hidden size，
  使用 LayerNorm 和 `Linear(1024, 256)`；两种 backend 对下游公开相同
  `[B, N, 256]` sequence/mask 契约。
- cosine cost 在最后一维 L2 normalize 后计算 `1 - cosine`，显式保留
  valid pair mask；不包含位置代价、对角偏好或长度修正。
- Sinkhorn 对每个样本使用有效时间上的 `1/T` 和有效词上的 `1/N`
  均匀边缘，在 log domain 固定迭代；迭代、计划、边缘、expected
  transport cost 和 entropy 诊断均为 FP32。
- transport pooling 使用每个词的实际列质量归一化
  `sum_t(P[t,j] h[t]) / sum_t(P[t,j])`，padding 词严格清零。
- 单元测试覆盖投影、cost、边缘、数值稳定、padding 不变性、等变性、
  梯度和 pooling。真实 inner-train MacBERT/BGE-M3 的 FP32/AMP、
  subject adapter 开关及 325 帧最大 EEG 输出 batch 均完成前后向
  smoke；validation/test context targets 未被访问。
- 当前只把 expected transport cost 作为公开 cost，并独立报告 entropy；
  三项最终训练 loss、prototype、partial/unbalanced/monotonic OT、位置
  cost 和顺序约束均未在本阶段引入。

## 阶段 6：逐 fold 训练与 train-only prototype

- 每个 outer fold 只在 inner-train 优化参数。
- inner-validation 只用于早停、checkpoint、损失权重、OT 参数和全局阈值。
- 每折 prototype 只由该折 inner-train 的上下文词表示生成，按 `text_embedding_idx` 计一次，并报告 context-group DF。
- outer-test 文本不得进入 prototype、训练或 checkpoint 选择。
- 固定随机种子、配置、数据哈希、模型 revision 和 checkpoint 元数据。

验收：测试推理函数只接受 EEG 与冻结 train-only prototype；代码层面无法读取测试上下文向量。

## 阶段 7：固定词表评估与消融

- 汇总全部 out-of-fold 预测后报告 Core 主结果，Main/Extended 次级结果及覆盖率。
- 指标：逐词 macro-AUPRC、多正例 Recall@1/5/10、按受试者 view 和 occurrence 组聚合结果。
- 置信区间按规范化短句组聚类 bootstrap。
- 时长审计：duration-only、frequency-only、duration+subject+run、时长匹配 AUC。
- 消融：MacBERT/BGE-M3、无 OT、无 token loss、无 prototype loss、mean pooling、GroupNorm/BatchNorm；partial/unbalanced OT 留到第一版完成后。

验收：所有指标都使用固定候选集合；资格不足的词显式保留并标注，不静默删除；结论区分主要、次要和探索性分析。
