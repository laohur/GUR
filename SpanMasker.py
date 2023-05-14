import collections
import math
import os
import random
import time

import numpy as np
from logzero import logger
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from transformers.models.bert.tokenization_bert import BasicTokenizer


def merge_spans(span_lens):
    if len(span_lens) >= 3:
        span_lens.sort()
        for i in range(1, len(span_lens) - 1):
            a = span_lens[i - 1]
            b = span_lens[i]
            if a[0] == b[0] == 1:
                span_lens[i] = (a[0] + b[0], a[1] + b[1])
                span_lens[i - 1] = None
    return [x for x in span_lens if x]


class SpanMasker:
    def __init__(self, mask_ratio=0.15, sentinels=[]):
        """https://github.com/facebookresearch/SpanBERT/blob/main/pretraining/fairseq/data/masking.py"""
        super().__init__()
        self.mask_ratio = mask_ratio
        self.sentinels = sentinels
        self.lower = 1
        self.upper = 10
        self.lens = list(range(self.lower, self.upper + 1))
        self.p = 0.2
        # self.len_distrib = (
        #     [
        #         self.p * (1 - self.p) ** (i - self.lower)
        #         for i in range(self.lower, self.upper + 1)
        #     ]
        #     if self.p >= 0
        #     else None
        # )  # 3.797097503983286

        self.p = 0.66
        self.len_distrib = (
            [self.p ** abs(i - 3) for i in range(self.lower, self.upper + 1)]
            if self.p >= 0
            else None
        )  # 3.7950959958414012
        self.len_distrib = [x / (sum(self.len_distrib))
                            for x in self.len_distrib]
        self.expected_len = sum(
            [self.lens[i] * self.len_distrib[i] for i in range(len(self.lens))]
        )

    def get_mask_spans(self, sent_length):
        """ mask """

        if not sent_length:
            return []

        sum_cover = 0
        span_pairs = []
        while sum_cover != sent_length:
            span_len = np.random.choice(self.lens, p=self.len_distrib)
            cover = round(span_len / self.mask_ratio)
            if sum_cover + cover >= sent_length:
                cover = sent_length - sum_cover
                span_len = round(cover * self.mask_ratio)
            if span_len >= 1:
                span_pairs.append([span_len, cover])
                sum_cover += cover
            else:
                if span_pairs:
                    span_pairs[-1][1] += cover
                    sum_cover += cover
                else:
                    return []
        assert sum(x[1] for x in span_pairs) == sent_length

        span_pairs = merge_spans(span_pairs)

        random.shuffle(span_pairs)
        # logger.info(span_lens)

        mask_pairs = []
        left = 0
        for span_len, cover in span_pairs:
            end = left + cover
            began = random.randint(left, end - span_len - 1)
            mask_pairs.append((began, span_len))
            left += cover
        assert left == sent_length
        return mask_pairs

    def paint(self, tokens0, mask_pairs, style="t5"):
        """ paint """
        if not mask_pairs:
            return tokens0, tokens0
        tokens = tokens0[:]
        if style == "bert":
            tgt = tokens0
            for began, span_len in mask_pairs:
                for i in range(began, began + span_len):
                    tokens[i] = self.sentinels[i]
        if style == "t5":
            tgt = []
            for j, x in enumerate(mask_pairs[:100]):
                began, span_len = x
                tgt += [self.sentinels[j]] + tokens0[began: began + span_len]
                for i in range(began, began + span_len):
                    tokens[i] = ""
                tokens[began] = self.sentinels[j]

        src = [x for x in tokens if x]
        return src, tgt

    def mask_tokens(self, tokens, style="t5"):
        spans = self.get_mask_spans(len(tokens))
        src, tgt = self.paint(tokens, spans, style)
        return src, tgt

    # https://github.com/joeljang/Pretraining_T5_custom_dataset/blob/7dfbee9963197f2cda37c8a14085b78ed7c0bd54/pretrain.py#L472


if __name__ == "__main__":
    sentinels = [f"<extra_id_{i}>" for i in range(100)]

    basicTokenizer = BasicTokenizer()
    line = "NBA2k20 二〇二二年▌♥never_split chinese 汉字前两字或者后两字成词不切， [SEP]    非ascii逐字切 苟利国家生死以，岂因福祸避趋之！"
    line = "青道路强化剂 路面翻新 道桥保养 保护防护新型环保材料"
    line = ' EVA的聚合方法：1、高压本体聚合 （塑料制品）2、溶液聚合 （PVC加工助剂）3、乳液聚合 （粘合剂制品）4、悬浮聚合乙酸乙烯（VA）含量高于30%的采用乳液聚合；\t 美国杜邦(Elvax)EVA系列学名性能特点应用领域Elvax 150EVA共聚物;抗氧化性;食品接触的合规性;热稳定性工业应用;混合;混料;粘合剂;密封剂;电线护套Elvax 150WEVA共聚物;抗氧化性;食品接触的合规性;热稳定性工业应用;混合;混料;粘合剂;密封剂;电线护套Elvax 210WEVA共聚物;抗氧化性;食品接触的合规性;热稳定性工业应用;混合;混料;粘合剂;密封剂;电线护套Elvax 220WEVA共聚物;抗氧化性;热稳定性;食品接触的合规性密封剂;工业应用;混合;混料;电线护套;粘合剂Elvax 240AEVA共聚物;抗氧化性;热稳定性;食品接触的合规性密封剂;工业应用;混合;混料;电线护套;粘合剂Elvax 240WEVA共聚物;抗氧化性;热稳定性;食品接触的合规性密封剂;工业应用;混合;混料;电线护套;粘合剂Elvax 250EVA共聚物;抗氧化性;食品接触的合规性;热稳定性工业应用;混合;混料;粘合剂;密封剂;电线护套Elvax 250AEVA共聚物;抗氧化性;食品接触的合规性工业应用Elvax 260EVA共聚物;抗氧化性;食品接触的合规性;热稳定性工业应用;混合;混料;粘合剂;密封剂;电线护套Elvax 260AEVA共聚物;抗氧化性;食品接触的合规性工业应用Elvax 265EVA共聚物;抗氧化性;食品接触的合规性;热稳定性工业应用;混合;混料;粘合剂;密封剂;电线护套Elvax 265AEVA共聚物;抗氧化性;食品接触的合规性工业应用Elvax 3120EVA低温热封性;光滑性;共聚物;抗氧化性;抗结块性;热稳定性;良好的柔韧性;良好的热封性;食品接触的合规性包装;薄膜Elvax 3124EVA低温热封性;共聚物;抗氧化性;热稳定性;良好的柔韧性;良好的热封性;食品接触的合规性包装;流延薄膜;涂层应用;薄膜Elvax 3128-1EVA低温热封性;共聚物;抗氧化性;热稳定性;良好的柔韧性;良好的热封性;食品接触的合规性包装;薄膜Elvax 3129-1EVA低温热封性;共聚物;抗氧化性;食品接触的合规性Blown FilmElvax 3130EVA低温热封性;共聚物;抗氧化性;热稳定性;良好的柔韧性;良好的热封性;食品接触的合规性包装;薄膜Elvax 3134SBZEVA低温热封性;光滑性;共聚物;抗氧化性;抗结块性;热稳定性;良好的柔韧性;良好的热封性;食品接触的合规性包装;流延薄膜;涂层应用;薄膜Elvax 3135SBEVA低温热封性;光滑性;共聚物;抗氧化性;抗结块性;热稳定性;良好的柔韧性;良好的热封性;食品接触的合规性包装;薄膜Elvax 3135XEVA低温热封性;共聚物;抗氧化性;热稳定性;良好的柔韧性;良好的热封性;食品接触的合规性包装;薄膜Elvax 3135XZEVA低温热封性;共聚物;抗氧化性;热稳定性;良好的柔韧性;良好的热封性;食品接触的合规性包装;薄膜Elvax 3150EVA低温热封性;共聚物;抗氧化性;热稳定性;良好的柔韧性;良好的热封性;食品接触的合规性包装;薄膜Elvax 3150AEVA低温热封性;抗氧化性;食品接触的合规性Blown Film;包装;薄膜Elvax 3155EVA低温热封性;共聚物;抗氧化性;热稳定性;良好的柔韧性;良好的热封性;食品接触的合规性包装;流延薄膜;涂层应用;薄膜Elvax 3165EVA低温热封性;共聚物;抗氧化性;热稳定性;良好的柔韧性;良好的热封性;食品接触的合规性包装;薄膜Elvax 3165AEVA低温热封性;抗氧化性;食品接触的合规性Blown Film;包装;薄膜Elvax 3165LGEVA低温热封性;低速凝固晶点;共聚物;抗氧化性;热稳定性;良好的柔韧性;良好的热封性;食品接触的合规性包装;薄膜Elvax 3165SBEVA低温热封性;光滑性;共聚物;抗氧化性;抗结块性;热稳定性;良好的柔韧性;良好的热封性;食品接触的合规性包装;薄膜Elvax 3165VLGAEVA低温热封性;低速凝固晶点;共聚物;抗氧化性;良好的柔韧性;良好的热封性;食品接触的合规性包装;薄膜Elvax 3169ZEVA低温热封性;光滑性;共聚物;抗氧化性;抗结块性;热稳定性;良好的柔韧性;良好的热封性;食品接触的合规性包装;薄膜Elvax 3170EVA低温热封性;共聚物;抗氧化性;热稳定性;良好的柔韧性;良好的热封性;食品接触的合规性包装;薄膜Elvax 3170AEVA低温热封性;抗氧化性;食品接触的合规性Blown Film;包装;薄膜Elvax 3170SHBEVA低温热封性;光滑性;共聚物;抗氧化性;抗结块性;热稳定性;良好的柔韧性;良好的热封性;食品接触的合规性包装;薄膜Elvax 3172ZEVA低温热封性;光滑性;共聚物;抗氧化性;抗结块性;热稳定性;良好的柔韧性;良好的热封性;食品接触的合规性Blown Film;包装;薄膜Elvax 3174EVA低温热封性;共聚物;抗氧化性;热稳定性;良好的柔韧性;良好的热封性;食品接触的合规性包装;流延薄膜;涂层应用;薄膜Elvax 3174SHBEVA低温热封性;光滑性;共聚物;抗氧化性;抗结块性;热稳定性;良好的柔韧性;良好的热封性;食品接触的合规性包装;医疗/护理用品;外壳;层压板;护罩;流延薄膜;涂层应用;粘合剂;粘结树脂;药...Elvax 3175EVA共聚物;抗氧化性Elvax 3175LGAEVA低温热封性;低速凝固晶点;抗氧化性包装;流延薄膜;薄膜Elvax 3176EVA低温热封性;光滑性;共聚物;抗氧化性;热稳定性;良好的柔韧性;良好的热封性;食品接触的合规性包装;流延薄膜;涂层应用;薄膜Elvax 3176BFZEVA共聚物;抗氧化性;食品接触的合规性Elvax 3176CW-3EVA共聚物;光滑性;抗氧化性;食品接触的合规性;中等抗结块性Elvax 3176SBEVA低温热封性;共聚物;抗氧化性;良好的热封性;良好的柔韧性;热稳定性;食品接触的合规性包装;薄膜;流延薄膜;涂层应用Elvax 3178ZEVA低温热封性;共聚物;抗氧化性;良好的热封性;良好的柔韧性;热稳定性;食品接触的合规性包装;薄膜;流延薄膜Elvax 3180EVA低温热封性;共聚物;抗氧化性;良好的热封性;良好的柔韧性;热稳定性;食品接触的合规性包装;薄膜;流延薄膜Elvax 3180ZEVA低温热封性;共聚物;抗氧化性;良好的热封性;良好的柔韧性;热稳定性;食品接触的合规性包装;薄膜;流延薄膜Elvax 3182EVA低温热封性;共聚物;抗氧化性;良好的热封性;良好的柔韧性;热稳定性;食品接触的合规性包装;薄膜;流延薄膜Elvax 3182-2EVA低温热封性;共聚物;抗氧化性;良好的热封性;良好的柔韧性;热稳定性;食品接触的合规性包装;薄膜;流延薄膜Elvax 3182AEVA低温热封性;抗氧化性;食品接触的合规性Blown Film;包装;薄膜;流延薄膜Elvax 3185EVA低温热封性;共聚物;抗氧化性;良好的热封性;良好的柔韧性;热稳定性;食品接触的合规性包装;薄膜;流延薄膜Elvax 3185LGAEVA低温热封性;共聚物;抗氧化性;良好的热封性;良好的柔韧性;热稳定性;食品接触的合规性包装;薄膜;流延薄膜Elvax 3190EVA低温热封性;共聚物;抗氧化性;良好的热封性;良好的柔韧性;热稳定性;食品接触的合规性包装;薄膜;流延薄膜Elvax 3190AEVA低温热封性;抗氧化性包装;薄膜;流延薄膜Elvax 3200-2EVA低温热封性;共聚物;抗氧化性;良好的热封性;良好的柔韧性;热稳定性;食品接触的合规性包装;薄膜;涂层应用Elvax 350EVA共聚物;抗氧化性;热稳定性;食品接触的合规性电线护套;工业应用;混合;混料;密封剂;粘合剂Elvax 360EVA共聚物;抗氧化性;热稳定性;食品接触的合规性电线护套;工业应用;混合;混料;密封剂;粘合剂Elvax 360AEVA共聚物;抗氧化性工业应用Elvax 40L-03EVA半结晶;低速凝固晶点;高分子量;共聚物;抗氧化性;热稳定性;食品接触的合规性;优良外观...电线电缆应用;电线护套;工业应用;混合;混料;密封剂;汽车领域的应用;粘合剂Elvax 40-WEVA共聚物;抗氧化性;食品接触的合规性;热稳定性工业应用;混合;混料;粘合剂;密封剂;电线护套Elvax 410EVA共聚物;抗氧化性;热稳定性;食品接触的合规性电线护套;工业应用;混合;混料;密封剂;粘合剂Elvax 420EVA共聚物;抗氧化性;热稳定性;食品接触的合规性电线护套;工业应用;混合;混料;密封剂;粘合剂Elvax 420AEVA共聚物;抗氧化性;热稳定性;食品接触的合规性工业应用;混合;混料;密封剂;粘合剂Elvax 4260EVA抗氧化性;良好粘结性;耐油脂性能;热稳定性，良好;热粘性强度;三元共聚物;食品接触的合规性电线护套;工业应用;混料;密封剂;粘合剂Elvax 4310EVA低粘度;抗氧化性;良好粘结性;耐油脂性能;热稳定性，良好;三元共聚物;食品接触的合规性电线护套;工业应用;混料;密封剂;粘合剂Elvax 4320EVA抗氧化性;热稳定性，良好;三元共聚物;食品接触的合规性;中等粘性电线护套;工业应用;混料;密封剂;粘合剂Elvax 4355EVA高分子量;抗氧化性;良好的柔韧性;热稳定性，良好;热粘性强度;韧性良好;三元共聚物;食品接触的合规性电线护套;工业应用;混料;密封剂;粘合剂Elvax 440EVA共聚物;抗氧化性;热稳定性;食品接触的合规性电线护套;工业应用;混合;混料;密封剂;粘合剂Elvax 450EVA共聚物;抗氧化性;食品接触的合规性;热稳定性工业应用;混合;混料;粘合剂;密封剂;电线护套Elvax 450AEVA共聚物;抗氧化性;食品接触的合规性工业应用Elvax 460EVA共聚物;抗氧化性;食品接触的合规性;热稳定性工业应用;混合;混料;粘合剂;密封剂;电线护套Elvax 460AEVA共聚物;抗氧化性工业应用Elvax 470EVA共聚物;抗氧化性;热稳定性;食品接触的合规性电线护套;工业应用;混合;混料;密封剂;粘合剂Elvax 470AEVA共聚物;抗氧化性;食品接触的合规性工业应用Elvax 550EVA共聚物;抗氧化性;热稳定性;食品接触的合规性电线护套;工业应用;混合;混料;密封剂;粘合剂Elvax 550AEVA共聚物;抗氧化性;食品接触的合规性工业应用;密封剂;粘合剂Elvax 560EVA共聚物;抗氧化性;热稳定性;食品接触的合规性电线护套;工业应用;混合;混料;密封剂;粘合剂Elvax 560AEVA共聚物;抗氧化性;食品接触的合规性工业应用Elvax 650QEVA共聚物;抗氧化性;热稳定性;食品接触的合规性电线护套;工业应用;混合;混料;密封剂;粘合剂Elvax 660EVA共聚物;抗氧化性;热稳定性;食品接触的合规性电线护套;工业应用;混合;混料;密封剂;粘合剂Elvax 660AEVA共聚物;抗氧化性;食品接触的合规性工业应用Elvax 670EVA共聚物;抗氧化性;热稳定性;食品接触的合规性电线护套;工业应用;混合;混料;密封剂;粘合剂Elvax 750EVA共聚物;抗氧化性;食品接触的合规性;热稳定性工业应用;混合;混料;粘合剂;密封剂;电线护套Elvax 760EVA共聚物;抗氧化性;热稳定性;食品接触的合规性密封剂;工业应用;混合;混料;电线护套;粘合剂Elvax 760AEVA共聚物;抗氧化性;食品接触的合规性工业应用Elvax 760QEVA共聚物;抗氧化性;热稳定性;食品接触的合规性密封剂;工业应用;混合;混料;电线护套;粘合剂Elvax 770EVA共聚物;抗氧化性;热稳定性;食品接触的合规性密封剂;工业应用;混合;混料;电线护套;粘合剂Elvax CE9619-1EVA光滑性;共聚物;抗结块性;食品接触的合规性母料;混料\n'

    masker = SpanMasker(
        mask_ratio=0.15, sentinels=[f"<extra_id_{i}>" for i in range(100)]
    )
    tokens = basicTokenizer.tokenize(line)
    # print(tokens)
    # src,tgt=masker.mask_tokens(tokens,style='bert')
    # print( ' '.join(src),'--->',' '.join(tgt))
    src, tgt = masker.mask_tokens(tokens, style="t5")
    print(" ".join(src), "--->", " ".join(tgt))

    pretrained_model = "Langboat/mengzi-t5-base"
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model,)  # not Fast
    trin_file = "cat /home/entropy/data/gur/sents2pair-repeat2/sku_pure_pairs-lcs1-title1-repeat2.tsv.uniq"

    for line in os.popen(trin_file):
        tokens = basicTokenizer.tokenize(line)
        try:
            src, tgt = masker.mask_tokens(tokens, style="t5")
        except Exception as e:
            logger.error((line, e))
            exit()
        if ord(line[0]) % 1000 == 0:
            print(" ".join(src), "--->", " ".join(tgt))

"""
nba2k20 二 〇 二 二 年 ▌♥never _ split chinese 汉 字 前 两 字 <extra_id_0> 成 词 不 切 ， [SEP] 非 ascii 逐 字 切 苟 利 国 家 生 死 以 ， <extra_id_1> 福 祸 避 趋 之 ！ ---> <extra_id_0> 或 者 后 两 字 <extra_id_1> 岂 因

"""
