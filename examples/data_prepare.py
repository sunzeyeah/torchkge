import os
import argparse
import logging
import json
import random

logging.basicConfig(
    format="%(asctime)s %(levelname)-4s [%(filename)s:%(lineno)s]  %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

NUM_RELATIONS_PER_CATE = 20
MIN_COUNT_RELATION = 10 # 如果某个品类中，该relation出现次数不足，则剔除该relation
RELATION_CATE_NAME = "cate_name"
RELATION_CATE_NAME_ID = 0
RELATION_INDUSTRY_NAME = "industry_name"
RELATION_INDUSTRY_NAME_ID = 1
# UNKNOW_ITEM = "/item/unknown"
# UNKNOW_VALUE = "/value/unknown"
# UNKNOW_ITEM_ID = 0
# UNKNOW_VALUE_ID = 1


def get_parser():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", required=True, type=str, help="模型训练数据地址")
    parser.add_argument("--output_dir", required=True, type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--dtypes", default="train,valid", type=str, help="data types used",)
    parser.add_argument("--filter_method", default="freq", type=str, help="过滤relation的方法，2种取值：(1) freq: 基于最小频率，"
                                                                          "(2) topn：基于频率前n位")
    parser.add_argument("--min_freq", default=10, type=int, help="过滤relation的最小频率")
    parser.add_argument("--max_rank", default=20, type=int, help="过滤relation的最大排名")

    return parser.parse_args()


def relation_filter(args):
    # 统计relation数量
    relation_count = dict()
    ct = 0
    for dtype in args.dtypes.split(","):
        file_info = os.path.join(args.data_dir, f"item_{dtype}_info.jsonl")
        with open(file_info, "r", encoding="utf-8") as r:
            while True:
                line = r.readline()
                if not line:
                    break
                d = json.loads(line.strip())
                cate_name = d['cate_name']

                # item pvs
                if 'item_pvs' in d:
                    if cate_name not in relation_count:
                        relation_count[cate_name] = dict()
                    for item_pv in d['item_pvs'].replace("#", "").split(";"):
                        try:
                            relation_key, v = item_pv.split(":", maxsplit=1)
                        except Exception:
                            logger.warning(f"[Item Pv Split Error] {item_pv}")
                            continue
                        if relation_key not in relation_count[cate_name]:
                            relation_count[cate_name][relation_key] = 0
                        relation_count[cate_name][relation_key] += 1
                        ct += 1
                #         if ct > 1000:
                #             break
                # if ct > 1000:
                #     break
    logger.info(f"# cates: {len(relation_count)}")

    # relation筛选
    relation_include = set()
    for cate, val_dict in relation_count.items():
        # 方法1：根据最小出现次数筛选
        if args.filter_method == "freq":
            for relation, ct in val_dict.items():
                if ct >= args.min_freq:
                    relation_include.add(relation)
        # 方法2：根据出现次数top-n筛选
        elif args.filter_method == "topn":
            sorted_val_dict = {k: v for k, v in sorted(val_dict.items(), key=lambda item: item[1], reverse=True)}
            for i, (cate, ct) in enumerate(sorted_val_dict.items()):
                if i >= args.max_rank:
                    break
                relation_include.add(cate)

    relation_include.add(RELATION_CATE_NAME)
    relation_include.add(RELATION_INDUSTRY_NAME)
    logger.info(f"# relations included: {len(relation_include)}")

    return relation_include


def post_processing(args):
    lef = {}
    rig = {}
    rellef = {}
    relrig = {}

    triple = open(os.path.join(args.output_dir, "train2id.txt"), "r")
    valid = open(os.path.join(args.output_dir, "valid2id.txt"), "r")
    test = open(os.path.join(args.output_dir, "test2id.txt"), "r")

    tot = (int)(triple.readline())
    for i in range(tot):
        content = triple.readline()
        h,t,r = content.strip().split()
        if not (h,r) in lef:
            lef[(h,r)] = []
        if not (r,t) in rig:
            rig[(r,t)] = []
        lef[(h,r)].append(t)
        rig[(r,t)].append(h)
        if not r in rellef:
            rellef[r] = {}
        if not r in relrig:
            relrig[r] = {}
        rellef[r][h] = 1
        relrig[r][t] = 1

    tot = (int)(valid.readline())
    for i in range(tot):
        content = valid.readline()
        h,t,r = content.strip().split()
        if not (h,r) in lef:
            lef[(h,r)] = []
        if not (r,t) in rig:
            rig[(r,t)] = []
        lef[(h,r)].append(t)
        rig[(r,t)].append(h)
        if not r in rellef:
            rellef[r] = {}
        if not r in relrig:
            relrig[r] = {}
        rellef[r][h] = 1
        relrig[r][t] = 1

    tot = (int)(test.readline())
    for i in range(tot):
        content = test.readline()
        h,t,r = content.strip().split()
        if not (h,r) in lef:
            lef[(h,r)] = []
        if not (r,t) in rig:
            rig[(r,t)] = []
        lef[(h,r)].append(t)
        rig[(r,t)].append(h)
        if not r in rellef:
            rellef[r] = {}
        if not r in relrig:
            relrig[r] = {}
        rellef[r][h] = 1
        relrig[r][t] = 1

    test.close()
    valid.close()
    triple.close()

    f = open(os.path.join(args.output_dir, "type_constrain.txt"), "w")
    f.write("%d\n"%(len(rellef)))
    for i in rellef:
        f.write("%s\t%d"%(i,len(rellef[i])))
        for j in rellef[i]:
            f.write("\t%s"%(j))
        f.write("\n")
        f.write("%s\t%d"%(i,len(relrig[i])))
        for j in relrig[i]:
            f.write("\t%s"%(j))
        f.write("\n")
    f.close()

    rellef = {}
    totlef = {}
    relrig = {}
    totrig = {}
    # lef: (h, r)
    # rig: (r, t)
    for i in lef:
        if not i[1] in rellef:
            rellef[i[1]] = 0
            totlef[i[1]] = 0
        rellef[i[1]] += len(lef[i])
        totlef[i[1]] += 1.0

    for i in rig:
        if not i[0] in relrig:
            relrig[i[0]] = 0
            totrig[i[0]] = 0
        relrig[i[0]] += len(rig[i])
        totrig[i[0]] += 1.0

    s11=0
    s1n=0
    sn1=0
    snn=0
    f = open(os.path.join(args.output_dir, "test2id.txt"), "r")
    tot = (int)(f.readline())
    for i in range(tot):
        content = f.readline()
        h,t,r = content.strip().split()
        rign = rellef[r] / totlef[r]
        lefn = relrig[r] / totrig[r]
        if (rign < 1.5 and lefn < 1.5):
            s11+=1
        if (rign >= 1.5 and lefn < 1.5):
            s1n+=1
        if (rign < 1.5 and lefn >= 1.5):
            sn1+=1
        if (rign >= 1.5 and lefn >= 1.5):
            snn+=1
    f.close()

    f = open(os.path.join(args.output_dir, "test2id.txt"), "r")
    f11 = open(os.path.join(args.output_dir, "1-1.txt"), "w")
    f1n = open(os.path.join(args.output_dir, "1-n.txt"), "w")
    fn1 = open(os.path.join(args.output_dir, "n-1.txt"), "w")
    fnn = open(os.path.join(args.output_dir, "n-n.txt"), "w")
    fall = open(os.path.join(args.output_dir, "test2id_all.txt"), "w")
    tot = (int)(f.readline())
    fall.write("%d\n"%(tot))
    f11.write("%d\n"%(s11))
    f1n.write("%d\n"%(s1n))
    fn1.write("%d\n"%(sn1))
    fnn.write("%d\n"%(snn))
    for i in range(tot):
        content = f.readline()
        h,t,r = content.strip().split()
        rign = rellef[r] / totlef[r]
        lefn = relrig[r] / totrig[r]
        if (rign < 1.5 and lefn < 1.5):
            f11.write(content)
            fall.write("0"+"\t"+content)
        if (rign >= 1.5 and lefn < 1.5):
            f1n.write(content)
            fall.write("1"+"\t"+content)
        if (rign < 1.5 and lefn >= 1.5):
            fn1.write(content)
            fall.write("2"+"\t"+content)
        if (rign >= 1.5 and lefn >= 1.5):
            fnn.write(content)
            fall.write("3"+"\t"+content)
    fall.close()
    f.close()
    f11.close()
    f1n.close()
    fn1.close()
    fnn.close()


def main():
    args = get_parser()

    # step 1: relation filter
    relation_include = relation_filter(args)

    # step 2: triplet, relation and entity id mapping
    # triplet
    triplets = set()
    # entity dict
    entity_dict = dict()
    # entity_dict[UNKNOW_ITEM] = UNKNOW_ITEM_ID
    # entity_dict[UNKNOW_VALUE] = UNKNOW_VALUE_ID
    # relation dict
    relation_dict = dict()
    relation_dict[RELATION_CATE_NAME] = RELATION_CATE_NAME_ID
    relation_dict[RELATION_INDUSTRY_NAME] = RELATION_INDUSTRY_NAME_ID

    # entity_id = UNKNOW_VALUE_ID
    entity_id = -1
    relation_id = RELATION_INDUSTRY_NAME_ID
    ct = 0
    for dtype in args.dtypes.split(","):
        file_info = os.path.join(args.data_dir, f"item_{dtype}_info.jsonl")
        with open(file_info, "r", encoding="utf-8") as r:
            while True:
                line = r.readline()
                if not line:
                    break
                d = json.loads(line.strip())
                item_id = d['item_id']
                head_entity_key = f"/item/{item_id}"
                if head_entity_key not in entity_dict:
                    entity_id += 1
                    entity_dict[head_entity_key] = entity_id

                # triplet - cate_name
                cate_name = d['cate_name']
                cate_id = d['cate_id']
                tail_entity_key = f"/value/{cate_name}-{cate_id}"
                if tail_entity_key not in entity_dict:
                    entity_id += 1
                    entity_dict[tail_entity_key] = entity_id
                # triplet = tuple((entity_dict[head_entity_key], RELATION_CATE_NAME_ID, entity_dict[tail_entity_key]))
                triplet = tuple((head_entity_key, RELATION_CATE_NAME, tail_entity_key))
                triplets.add(triplet)

                # triplet - industry_name
                industry_name = d['industry_name']
                tail_entity_key = f"/value/{industry_name}"
                if tail_entity_key not in entity_dict:
                    entity_id += 1
                    entity_dict[tail_entity_key] = entity_id
                # triplet = tuple((entity_dict[head_entity_key], RELATION_INDUSTRY_NAME_ID, entity_dict[tail_entity_key]))
                triplet = tuple((head_entity_key, RELATION_INDUSTRY_NAME, tail_entity_key))
                triplets.add(triplet)

                # item pvs
                if 'item_pvs' in d:
                    for item_pv in d['item_pvs'].replace("#", "").split(";"):
                        try:
                            relation_key, v = item_pv.split(":", maxsplit=1)
                        except Exception:
                            logger.warning(f"[Item Pv Split Error] {item_pv}")
                            continue
                        if relation_key not in relation_include:
                            continue
                        tail_entity_key = f"/value/{v}"
                        if tail_entity_key not in entity_dict:
                            entity_id += 1
                            entity_dict[tail_entity_key] = entity_id
                        if relation_key not in relation_dict:
                            relation_id += 1
                            relation_dict[relation_key] = relation_id
                        # triplet = tuple((entity_dict[head_entity_key], relation_dict[relation_key], entity_dict[tail_entity_key]))
                        triplet = tuple((head_entity_key, relation_key, tail_entity_key))
                        triplets.add(triplet)
                        ct += 1
                #         if ct > 1000:
                #             break
                # if ct > 1000:
                #     break
    logger.info(f"# triplets: {len(triplets)}, # relations: {len(relation_dict)}, # entities: {len(entity_dict)}")

    # saving
    file_entity2id = os.path.join(args.output_dir, "entity2id.txt")
    file_relation2id = os.path.join(args.output_dir, "relation2id.txt")
    with open(file_entity2id, "w", encoding="utf-8") as w:
        # w.write(str(len(entity_dict))+"\n")
        for entity_name, entity_id in entity_dict.items():
            w.write("\t".join((entity_name, str(entity_id)))+"\n")
    with open(file_relation2id, "w", encoding="utf-8") as w:
        # w.write(str(len(relation_dict))+"\n")
        for relation_name, relation_id in relation_dict.items():
            w.write("\t".join((relation_name, str(relation_id)))+"\n")

    # train, valid & test split
    triplets = list(triplets)
    random.shuffle(triplets)
    test_proportion = 0.1
    valid_proportion = 0.1
    test_split_index = int(len(triplets) * test_proportion)
    valid_split_index = test_split_index + int(len(triplets) * valid_proportion)
    triplets_test = triplets[:test_split_index]
    triplets_valid = triplets[test_split_index:valid_split_index]
    triplets_train = triplets[valid_split_index:]
    logger.info(f"# train: {len(triplets_train)}, # valid: {len(triplets_valid)}, # test: {len(triplets_test)}")

    file_train2id = os.path.join(args.output_dir, "train2id.txt")
    with open(file_train2id, "w", encoding="utf-8") as w:
        # w.write(str(len(triplets_train))+"\n")
        for hid, rid, tid in triplets_train:
            w.write("\t".join((str(hid), str(rid), str(tid)))+"\n")
    file_valid2id = os.path.join(args.output_dir, "valid2id.txt")
    with open(file_valid2id, "w", encoding="utf-8") as w:
        # w.write(str(len(triplets_valid))+"\n")
        for hid, rid, tid in triplets_valid:
            w.write("\t".join((str(hid), str(rid), str(tid)))+"\n")
    file_test2id = os.path.join(args.output_dir, "test2id.txt")
    with open(file_test2id, "w", encoding="utf-8") as w:
        # w.write(str(len(triplets_test))+"\n")
        for hid, rid, tid in triplets_test:
            w.write("\t".join((str(hid), str(rid), str(tid)))+"\n")

    # step 3: post processing
    # post_processing(args)


if __name__ == "__main__":
    main()
