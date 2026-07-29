# 《小王子》通用高频词候选评估集 v1

## 产物

- `littleprince_hf_lexicon_v1.csv`：DF≥10的主词表，以及Core/Main/Extended嵌套标记。
- `littleprince_sentence_keyword_labels_v1.csv`：2,837个正式正文短句的分词结果与多标签。
- Excel审阅版：`outputs/littleprince_hf_v1/littleprince_hf_candidate_evaluation_v1.xlsx`。

## 数据范围

- 源工作簿：`D:/dataset/littleprince.xlsx`
- 源工作簿SHA256：`f3e0c6cafe96abd85618a5e0ca0e881f00c440ae7a8b4754b15818db2117d349`
- 正文映射：`data/manifests/littleprince_text_embedding_map.csv`
- 映射SHA256：`0ee44131a126c4ee064d3706837cc108785d814e2e25ae799dc44ce5002c0216`
- 只保留 `is_formal_littleprince=True`。
- 共有2,837个独立 `text_embedding_idx`，范围16..2852。

## 分词

未使用语言模型。采用：

1. Unicode NFKC规范化；
2. 自定义多字词按长度优先保护；
3. `Intl.Segmenter("zh-CN", {granularity: "word"})`；
4. 人称代词所有格的确定性拆分，例如“我的”→“我|的”；
5. 章节编号不进入词表。

完整规则见Excel工作簿的 `SegmentationRules` 工作表。

## 计数与层级

- TF：正文中的总出现次数。
- DF：包含该词的独立短句（text_embedding_idx）数量。
- story-local：出现章节少于3章、覆盖全文区间少于3个，或单章占比超过70%。
- Core：DF≥50且非story-local，共33词。
- Main：DF≥30且非story-local，共64词。
- Extended：DF≥20且非story-local，共100词。
- Master：DF≥10，共247词；包含story-local词，供主题关键词词表使用。

Core ⊆ Main ⊆ Extended ⊆ Master。

## 评估约束

- 词频按独立故事occurrence统计，不按受试者EEG重复数统计。
- 同一 `text_embedding_idx` 的全部受试者记录必须进入同一数据划分。
- 每个词的负样本需匹配短句字符数和EEG时长。
- 主要指标使用每词宏平均AUPRC、时长匹配AUC和固定Recall@K。
- story-local词不能单独用于支持通用词汇解码结论。
