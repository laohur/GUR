import json
import os
import random
import re
import subprocess
import unicodedata
from multiprocessing import Pool

import pysbd
from logzero import logger
from SuffixAutomaton import SuffixAutomaton

# text = "My name is Jonas E. Smith. Please turn to p. 55."
seger = pysbd.Segmenter(language="zh", clean=False)
# print(seger.segment(text))

# https://github.com/fxsjy/jieba/issues/575
resentencesp = re.compile('([﹒﹔﹖﹗．；。！？]["’”」』]{0,2}|：(?=["‘“「『]{1,2}|$))')


def splitsentence(sentence):
    s = sentence
    slist = []
    for i in resentencesp.split(s):
        if resentencesp.match(i) and slist:
            slist[-1] += i
        elif i:
            slist.append(i)
    return slist


def splitsentence1(sentence):
    slist = seger.segment(sentence)
    return slist


def clean(line):
    """ clean 
    https://stackoverflow.com/questions/32131901/best-way-to-clean-up-html-text
    """
    l = re.sub("<[^<]+?>", " ", line)
    l = " ".join(x.strip() for x in l.split() if x.strip())
    return l


def weight(l):
    score = 0
    for c in l:
        h = 0.1
        catg = unicodedata.category(c)[0]
        if catg == "L":
            h = 0.3
            if ord(c) > 10000:
                h = 1
        score += h
    return score


# def match(a, b):
#     re = lcs1(a, b)
#     if not re:
#         return False
#     (t, start, cand_start) = re[0]
#     wc=weight(t)
#     if wc >= 3:
#         return True
#     if len(a)<=len(b) and wc/weight(a)>0.7:
#         return True
#     if len(b)<len(a) and wc/weight(b)>0.7:
#         return True
#     return False


def sam_match(sam, b):
    re = sam.lcs1(b, min_len=1)
    if not re:
        return False
    (t, start, cand_start) = re[0]
    wc = max(weight(x[0]) for x in re)
    if wc >= 3:
        return True
    if wc >= 2:
        if weight(b) <= 15 or weight(sam.sequence) <= 15:
            return True
    return False


def normalize(l):
    return l[:128].lower()


def sents2pairs(doc):
    doc = [x for x in doc if 2 <= len(x) <= 128]
    if len(doc) < 2:
        return [], []

    N = len(doc)
    pairs0 = [(i, j) for i in range(0, N - 1) for j in range(i + 1, min(i + 1000, N))]
    pairs0 = random.choices(pairs0, k=min(100000, len(pairs0)))
    random.shuffle(pairs0)
    # pairs0 = pairs0[:100000]

    sams = [None] * len(doc)
    used = [0] * len(doc)
    used[0] -= args.title

    pairs = []
    doc1 = [normalize(x) for x in doc]
    for i, j in pairs0:
        if len(pairs) >= len(doc) * 2:
            break
        if used[i] >= args.repeat:
            sams[i] = None
            continue
        if used[j] >= args.repeat:
            sams[j] = None
            continue
        # if doc[i] == doc[j]:
        #     continue
        matched = False
        if not args.lcs:
            matched = True
        elif j - i < 100:
            if not sams[i]:
                sams[i] = SuffixAutomaton(doc1[i])
            if sam_match(sams[i], doc1[j]):
                matched = True
        if matched == True:
            used[i] += 1
            used[j] += 1
            pairs.append((i, j))
    return doc, pairs


def line2pair(li):
    try:
        doc0 = row_fn(li)
        doc = [clean(x) for x in doc0]
        doc = [x for x in doc0 if len(x) >= 2]
        if len(doc) >= 2:
            r = sents2pairs(doc)
            return r
    except Exception as e:
        logger.error((li[:100], len(li), e))


def doc2pair_batch(arg):
    logger.info(arg)
    src, tgt = arg
    reader = subprocess.Popen(
        src, shell=True, stdout=subprocess.PIPE, errors="ignore"
    ).stdout
    with Pool() as pool, open(tgt, "w") as w32:
        doc0 = []
        end = False
        n_pair = 0
        n_char = 0
        for i in range(10 ** 10):
            if end:
                break
            try:
                l = next(reader)
                n_char += len(l)
                doc0.append(l)
            except:
                end = True
                pass
            if end or n_char > 1 * 10 ** 7:
                docs = pool.imap_unordered(line2pair, doc0)
                for x in docs:
                    if not x:
                        continue
                    doc, pairs = x
                    for pair in pairs:
                        a = doc[pair[0]]
                        b = doc[pair[1]]
                        line = f"{a}\t{b}"
                        w32.write(line + "\n")
                        n_pair += 1
                logger.info(f" {tgt} i:{i}  batch:{len(doc0)}   n_pair:{n_pair}  ")
                doc0 = []
                n_char = 0

        logger.info(f"{src} i:{i}  batch:{len(doc0)}   -> n_pair:{n_pair}  ")


def sku2row(l):
    t = l.strip().split("\t")
    sents = t[:1] + splitsentence(t[1])
    return sents


def cls2row(l):
    doc = l.strip().split("\t")
    if len(doc) < 4:
        return ""
    title, abstract, keywords, category = doc[:4]
    sents = [title] + splitsentence(abstract)
    return sents


def wiki2row(l):
    try:
        j = json.loads(l)
    except:
        try:
            j = eval(l)
        except:
            return ""

    doc = j["text"].splitlines()
    sents = []
    for l in doc:
        sents = splitsentence(l.strip())
    sents = [x for x in sents if x]
    if len(sents) < 2:
        return []
    return sents[:-1][:10000]


# bigsort -u 1 -T "./" -i /data/gur/paragraph2pair-raw/csl_pairs-lcs1-repeat5-title0.tsv | bigsort -T "./" -s R   > /data/dataset/doc2pair/csl_pairs-lcs1-repeat5-title0.tsv
def uniq(tgt):
    cmd = f'bigsort -u 1 -T "./" -i {tgt}  | bigsort -T "./" -s R   > {tgt}.uniq'
    logger.info(cmd)
    os.system(cmd)
    logger.info("sorted")

    cmd = f"rm {tgt} "
    logger.info(cmd)
    os.system(cmd)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="")
    parser.add_argument(
        "--tgt_dir", type=str, default="/data/gur/sents2pair-raw"
    )
    parser.add_argument("--lcs", type=int, default=1)   
    parser.add_argument("--repeat", type=int, default=1)  
    parser.add_argument("--title", type=int, default=1)   
    args = parser.parse_args()
    import logzero

    logzero.logfile(
        __file__ + f"-lcs{args.lcs}-repeat{args.repeat}-title{args.title}.log", mode="w"
    )
    logger.info(args)
    src = args.src
    tgt_dir = args.tgt_dir
    repeat = args.repeat

    os.system(f"mkdir {tgt_dir}")
    row_fn = cls2row
    tgt = f"{tgt_dir}/csl_pairs-lcs{args.lcs}-title{args.title}-repeat{args.repeat}.tsv"
    arg = ("cat /data/corpus/CSL/csl_camera_readly.tsv", tgt)
    doc2pair_batch(arg)
    uniq(tgt)

    row_fn = wiki2row
    tgt = f"{tgt_dir}/wikisource_pairs-lcs{args.lcs}-title{args.title}-repeat{args.repeat}.tsv"
    arg = ("cat /data/wikipedia/zhwikisource-cirus.json", tgt)
    doc2pair_batch(arg)
    uniq(tgt)

    row_fn = wiki2row
    tgt = (
        f"{tgt_dir}/wiki_pairs-lcs{args.lcs}-title{args.title}-repeat{args.repeat}.tsv"
    )
    arg = ("cat /data/wikipedia/zhwiki-cirus.json", tgt)
    doc2pair_batch(arg)
    uniq(tgt)

    row_fn = sku2row
    line2pair(
        "体育场框架围栏网护栏网 篮球场运动场围栏网围网厂家安装\t关于球场围网详情如下;球场围网主要应用：各种体育场，运动场，学校操场，以及别墅小区健身场地。球场围网表面处理：围网选用PE/PVC 包塑，立柱横柱采用镀锌管加喷塑或者浸塑。球场围网颜色：草绿色墨绿色居多，黄色，白色，灰色都可以。球场围网优点：，网面平整，张力张紧，不受外力撞击变形，现场安装简便。球场围网常见规格型号：勾花网丝径2.2/3.8mm, 网孔5*5cm, 立柱60*2mm,横柱48*2mm，钢筋条8mm，此种规格常用于网体高3米，宽3米。安装便捷。球场围网安装方式：打混凝土预埋，法兰底盘打膨胀。球场围网生产周期：一般1000平米内7天内交货。球场围网颜色：草绿色，墨绿色，黄色，白色等等。"
    )

    tgt = f"{tgt_dir}/sku_pure_pairs-lcs{args.lcs}-title{args.title}-repeat{args.repeat}.tsv"
    arg = ("cat /data/sku/sku_desp_uniq-pure.tsv", tgt)  
    doc2pair_batch(arg)
    cmd = f'bigsort -u 1 -T "./" -i {tgt}  | bigsort -T "./" -s R   > {tgt}.uniq'
    uniq(tgt)
