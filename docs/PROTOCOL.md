# Little Prince Context-OT Protocol v1

## 1. 研究问题

主结论针对：

> 在未见过的短句上下文中，模型能否从 EEG 检测训练词表中的单词信息？

这是固定标签空间下的 occurrence 泛化，不是开放词表零样本解码，也不是精确的词时间定位。

## 2. 固定项与划分项

固定项：

- Core / Main / Extended / Master 词表；
- 人物、地点、动作等版本化关键词视图；
- 文本模型名称、revision 与 tokenizer；
- 指标和候选集合；
- 外层 fold 与内层验证规则。

必须划分的样本：

- 短句 occurrence；
- 与该 occurrence 对应的全部受试者 EEG；
- 规范化后完全相同的重复短句。

词类型允许跨 fold；完整短句及其 EEG view 不允许跨 fold。

## 3. 外层 5 折

外层测试采用 5 折分组、多标签分层划分。

- 一个 `text_embedding_idx` 的全部 EEG view 属于同一 group。
- 规范化文本相同的多个 `text_embedding_idx` 合并为同一 group。
- group 不得跨外层 fold。
- 分层目标优先平衡 Core 和 Main 的短句级存在标签。
- Extended 和 story-local 覆盖率单独审计，不通过移动测试样本来追求结果好看。
- 每个有效短句恰好产生一次 out-of-fold 测试预测。

外层 fold 只决定最终测试样本。任何训练选择不得读取当前 fold 的测试 EEG、测试损失或测试指标。

## 4. 内层验证

每个外层 fold 的其余 4/5 数据再按相同 group 约束划出内层验证集。

- 目标验证规模：外层训练数据的 12.5%，约等于全数据的 10%。
- 内层验证用于早停、checkpoint、loss 权重、OT 参数、score temperature 和全局阈值。
- Prototype 只使用 inner-train occurrence。
- inner-validation 上下文表示不得进入 prototype。
- 外层测试不得参与内层选择。

最终每个外层模型大约使用 70% 全数据训练、10% 验证、20% 测试。

## 5. 文本使用边界

冻结语言模型可以离线缓存所有文本的上下文特征，但每个训练运行必须通过 fold mask 限制用途：

| 数据 | OT训练目标 | Prototype | checkpoint选择 | 最终指标 |
|---|---:|---:|---:|---:|
| inner-train | 是 | 是 | 否 | 否 |
| inner-validation | 否 | 否 | 是 | 否 |
| outer-test | 否 | 否 | 否 | 是 |

测试文本只允许用于生成真值和离线错误分析。推理接口只能读取 EEG 和训练集冻结 prototype。

## 6. 词表与统计资格

完整候选集保持固定，不因某个 fold 中的正例数少而缩小。

逐词主统计使用预先冻结的 eligibility：

- `*_df`：每个 `text_embedding_idx` 短句 occurrence 计一次，不按 6–8 个受试者 EEG view 重复计数；
- `*_group_df`：规范化文本完全相同的短句组只计一次，是上下文泛化的主资格口径；
- prototype 探索最低：inner-train `group_df >= 10`；
- 主分析建议：train >= 20、validation >= 5、test >= 5；
- 不满足门槛的词仍留在候选集合，但不进入逐词 macro 推断；
- 所有覆盖率必须与指标一起报告。

Core 是第一主分析词表。Main 和 Extended 在 out-of-fold 结果上作为次级分析；story-local 只支持故事内探索性结论。

分层目标同时平衡短句 occurrence DF 和规范化上下文组 DF，并以
`0.1 : 0.9` 的权重优先保护独立上下文覆盖；样本总数偏差项权重为
`0.25`。该规则只使用冻结文本标签，在任何 EEG 模型训练和结果观察前确定。

## 7. 模型与损失范围

第一阶段总损失：

\[
\mathcal L =
\mathcal L_{\mathrm{OT-context}}
+\lambda_{\mathrm{token}}\mathcal L_{\mathrm{context-token}}
+\lambda_{\mathrm{lex}}\mathcal L_{\mathrm{prototype}}
\]

明确不加入：

- 关键词多标签 BCE 主损失；
- 词序位置代价；
- 顺序损失；
- 使用测试上下文构建 prototype。

第一版 Sinkhorn 使用 masked balanced OT、均匀有效边缘分布、cosine cost 和 log-domain FP32 实现。Partial / unbalanced OT 属于后续消融。

## 8. 主指标

- 每词 macro-AUPRC；
- 固定 Recall@1 / 5 / 10；
- EEG 时长和短句长度匹配的逐词 AUC；
- 单受试者 view 指标；
- 同一 occurrence 跨受试者聚合指标；
- 按 occurrence group 聚类 bootstrap 的置信区间；
- frequency-only、duration-only、duration + subject + run 基线。

同一句的 6–8 个受试者 view 不能被当作独立文本 occurrence 计算 DF、prototype 权重或统计置信区间。

## 9. 独立辅助协议

LOSO 可以用于研究跨受试者迁移，但它不替代主协议。若其他受试者提供过同一句刺激，LOSO 不能证明未见短句上下文泛化。
