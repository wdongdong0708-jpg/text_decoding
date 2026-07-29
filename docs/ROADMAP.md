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

- Dataset 按 `text_embedding_idx` 联结 EEG、fold role、词 occurrence 和上下文词缓存。
- Collate 输出 `eeg/eeg_mask/context_words/word_mask/keyword_ids`。
- 一个短句的全部受试者 EEG view 只共享文本缓存，不重复进入 prototype 统计。
- 空序列在数据构建时被排除，不在训练循环中临时跳过。

验收：不同 EEG 长度和词数可同批训练；padding 改变不影响有效区域；fold role 无交叉。

## 阶段 4：EEG sequence encoder

- 从旧 `SimpleConvTimeAgg` 的卷积干线提炼可复用结构，但输出 `[B, T', 256]` 而非整句向量。
- 默认时间 stride 4，控制 16 GB 显存占用。
- 用 GroupNorm 或 LayerNorm 替代会受 padding 比例影响的 BatchNorm。
- 保留 duration-only、mean-pooling 与旧整句向量基线的独立入口。

验收：mask 下采样正确；padding 不改变有效输出；无 NaN；正反向 smoke test 通过。

## 阶段 5：Sinkhorn OT 与三项损失

- masked、balanced、log-domain、内部 FP32 的 Sinkhorn。
- 均匀有效时间边缘与均匀有效词边缘，cosine cost。
- 实现 `OT-context + λ_token context-token + λ_lex prototype`。
- prototype 损失只覆盖达到 inner-train `group_df` 门槛的词；同词 occurrence 不互作负例。
- 第一版不加入关键词 BCE 主损失、词序代价或顺序损失。

验收：无效位置 transport mass 为零；行列边缘误差达标；极端相似度无 NaN；梯度可回传。

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
