#@title
# UTILS

import json
import os


class Alphabet:
    def __init__(self, name, padflag=True, unkflag=True, keep_growing=True):
        self.name = name
        self.PAD = "</pad>"
        self.UNKNOWN = "</unk>"
        self.padflag = padflag
        self.unkflag = unkflag
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing
        self.next_index = 0
        self.index_num = {}
        if self.padflag:
            self.add(self.PAD)
        if self.unkflag:
            self.add(self.UNKNOWN)

    def clear(self, keep_growing=True):
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing
        self.next_index = 0

    def add(self, instance):
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = self.next_index
            self.index_num[self.next_index] = 1
            self.next_index += 1


    def get_index(self, instance):
        try:
            index = self.instance2index[instance]
            self.index_num[index] = self.index_num[index] + 1
            return index
        except KeyError:
            if self.keep_growing:
                index = self.next_index
                self.add(instance)
                return index
            else:
                if self.UNKNOWN in self.instance2index:
                    return self.instance2index[self.UNKNOWN]
                else:
                    print(self.name + " get_index raise wrong, return 0. Please check it")
                    return 0

    def get_instance(self, index):
        if index == 0:
            if self.padflag:
                print(self.name +" get_instance of </pad>, wrong?")
            if not self.padflag and self.unkflag:
                print(self.name +" get_instance of </unk>, wrong?")
            return self.instances[index]
        try:
            return self.instances[index]
        except IndexError:
            # print('WARNING: '+ self.name + ' Alphabet get_instance, unknown instance, return the </unk> label.')
            return

    def size(self):
        return len(self.instances)

    def iteritems(self):
        return self.instance2index.items()

    def enumerate_items(self, start=1):
        if start < 1 or start >= self.size():
            raise IndexError("Enumerate is allowed between [1 : size of the alphabet)")
        return zip(range(start, len(self.instances) + 1), self.instances[start - 1:])

    def close(self):
        self.keep_growing = False

    def open(self):
        self.keep_growing = True

    def get_content(self):
        return {'instance2index': self.instance2index, 'instances': self.instances}

    def from_json(self, data):
        self.instances = data["instances"]
        self.instance2index = data["instance2index"]

    def save(self, output_directory, name=None):
        """
        Save both alhpabet records to the given directory.
        :param output_directory: Directory to save model and weights.
        :param name: The alphabet saving name, optional.
        :return:
        """
        saving_name = name if name else self.__name
        try:
            json.dump(self.get_content(), open(os.path.join(output_directory, saving_name + ".json"), 'w'))
        except Exception as e:
            print("Exception: Alphabet is not saved: " % repr(e))

    def load(self, input_directory, name=None):
        """
        Load model architecture and weights from the give directory. This allow we use old models even the structure
        changes.
        :param input_directory: Directory to save model and weights
        :return:
        """
        loading_name = name if name else self.__name
        self.from_json(json.load(open(os.path.join(input_directory, loading_name + ".json"))))


class AverageMeter(object):
    """
    Computes and stores the average and current value of metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """
        String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


import os, pickle, copy, sys, copy

from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoConfig


class Data:
    def __init__(self):
        self.relational_alphabet = Alphabet("Relation", unkflag=False, padflag=False)
        self.train_loader = []
        self.valid_loader = []
        self.test_loader = []
        self.orig_tokens = []
        self.weight = {}


    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Relation Alphabet Size: %s" % self.relational_alphabet.size())
        print("     Train  Instance Number: %s" % (len(self.train_loader)))
        print("     Valid  Instance Number: %s" % (len(self.valid_loader)))
        print("     Test   Instance Number: %s" % (len(self.test_loader)))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def generate_instance(self, args, data_process):
        tokenizer = BertTokenizer.from_pretrained(args.bert_directory, do_lower_case=False)

        if "train_file" in args:
            self.train_loader, _ = data_process(args.train_file, self.relational_alphabet, tokenizer)
            self.weight = copy.deepcopy(self.relational_alphabet.index_num)
        if "valid_file" in args:
            self.valid_loader, _ = data_process(args.valid_file, self.relational_alphabet, tokenizer)
        if "test_file" in args:
            self.test_loader, tokens = data_process(args.test_file, self.relational_alphabet, tokenizer)
            self.orig_tokens = tokens
        self.relational_alphabet.close()

def build_data(args):

    file = args.generated_data_directory + args.dataset_name + "_" + args.model_name + "_data.pickle"
    if os.path.exists(file) and not args.refresh:
        data = load_data_setting(args)
    else:
        data = Data()
        data.generate_instance(args, data_process)
        save_data_setting(data, args)
    return data


def save_data_setting(data, args):
    new_data = copy.deepcopy(data)
    data.show_data_summary()
    if not os.path.exists(args.generated_data_directory):
        os.makedirs(args.generated_data_directory)
    saved_path = args.generated_data_directory + args.dataset_name + "_" + args.model_name + "_data.pickle"
    with open(saved_path, 'wb') as fp:
        pickle.dump(new_data, fp)
    print("Data setting is saved to file: ", saved_path)


def load_data_setting(args):

    saved_path = args.generated_data_directory + args.dataset_name + "_" + args.model_name + "_data.pickle"
    with open(saved_path, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting is loaded from file: ", saved_path)
    data.show_data_summary()
    return data


import torch, collections


def list_index(list1: list, list2: list) -> list:
    try:
      start = [i for i, x in enumerate(list2) if x == list1[0]]
      end = [i for i, x in enumerate(list2) if x == list1[-1]]
      if len(start) == 1 and len(end) == 1:
          return start[0], end[0]
      else:
          index = 0, 1
          for i in start:
              for j in end:
                  if i <= j:
                      if list2[i:j+1] == list1:
                          index = (i, j)
                          break
          return index[0], index[1]
    except:
      print(list1, list2)





# def list_index(list1: list, list2: list) -> list:
#     start = [i for i, x in enumerate(list2) if x == list1[0]]
#     end = [i for i, x in enumerate(list2) if x == list1[-1]]
#     if len(start) == 1 and len(end) == 1:
#         return start[0], end[0]
#     else:
#         for i in start:
#             for j in end:
#                 if i <= j:
#                     if list2[i:j+1] == list1:
#                         break
#         return i, j
#

def remove_accents(text: str) -> str:
    accents_translation_table = str.maketrans(
    "áéíóúýàèìòùỳâêîôûŷäëïöüÿñÁÉÍÓÚÝÀÈÌÒÙỲÂÊÎÔÛŶÄËÏÖÜŸ",
    "aeiouyaeiouyaeiouyaeiouynAEIOUYAEIOUYAEIOUYAEIOUY"
    )
    return text.translate(accents_translation_table)


def data_process(input_doc, relational_alphabet, tokenizer):
    samples = []
    tokenized_texts = []
    with open(input_doc) as f:
        lines = f.readlines()
        lines = [eval(ele) for ele in lines]
    for i in range(len(lines)):
        token_sent = [tokenizer.cls_token] + tokenizer.tokenize(remove_accents(lines[i]["sentText"])) + [tokenizer.sep_token]
        triples = lines[i]["relationMentions"]
        target = {"relation": [], "head_start_index": [], "head_end_index": [], "tail_start_index": [], "tail_end_index": []}
        for triple in triples:
            head_entity = remove_accents(triple["em1Text"])
            tail_entity = remove_accents(triple["em2Text"])
            head_token = tokenizer.tokenize(head_entity)
            tail_token = tokenizer.tokenize(tail_entity)
            relation_id = relational_alphabet.get_index(triple["label"])
            head_start_index, head_end_index = list_index(head_token, token_sent)
            assert head_end_index >= head_start_index
            tail_start_index, tail_end_index = list_index(tail_token, token_sent)
            assert tail_end_index >= tail_start_index
            target["relation"].append(relation_id)
            target["head_start_index"].append(head_start_index)
            target["head_end_index"].append(head_end_index+1)
            target["tail_start_index"].append(tail_start_index)
            target["tail_end_index"].append(tail_end_index+1)
        sent_id = tokenizer.convert_tokens_to_ids(token_sent)
        samples.append([i, sent_id, target])
        tokenized_texts.append(token_sent)
    return samples, tokenized_texts


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def generate_span(start_logits, end_logits, info, args):
    seq_lens = info["seq_len"] # including [CLS] and [SEP]
    sent_idxes = info["sent_idx"]
    _Prediction = collections.namedtuple(
        "Prediction", ["start_index", "end_index", "start_prob", "end_prob"]
    )
    output = {}
    start_probs = start_logits.softmax(-1)
    end_probs = end_logits.softmax(-1)
    start_probs = start_probs.cpu().tolist()
    end_probs = end_probs.cpu().tolist()
    for (start_prob, end_prob, seq_len, sent_idx) in zip(start_probs, end_probs, seq_lens, sent_idxes):
        output[sent_idx] = {}
        for triple_id in range(args.num_generated_triples):
            predictions = []
            start_indexes = _get_best_indexes(start_prob[triple_id], args.n_best_size)
            end_indexes = _get_best_indexes(end_prob[triple_id], args.n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the sentence. We throw out all
                    # invalid predictions.
                    if start_index >= (seq_len-1): # [SEP]
                        continue
                    if end_index >= (seq_len-1):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > args.max_span_length:
                        continue
                    predictions.append(
                        _Prediction(
                            start_index=start_index,
                            end_index=end_index,
                            start_prob=start_prob[triple_id][start_index],
                            end_prob=end_prob[triple_id][end_index],
                        )
                    )
            output[sent_idx][triple_id] = predictions
    return output


def generate_relation(pred_rel_logits, info, args):
    rel_probs, pred_rels = torch.max(pred_rel_logits.softmax(-1), dim=2)
    rel_probs = rel_probs.cpu().tolist()
    pred_rels = pred_rels.cpu().tolist()
    sent_idxes = info["sent_idx"]
    output = {}
    _Prediction = collections.namedtuple(
        "Prediction", ["pred_rel", "rel_prob"]
    )
    for (rel_prob, pred_rel, sent_idx) in zip(rel_probs, pred_rels, sent_idxes):
        output[sent_idx] = {}
        for triple_id in range(args.num_generated_triples):
            output[sent_idx][triple_id] = _Prediction(
                            pred_rel=pred_rel[triple_id],
                            rel_prob=rel_prob[triple_id])
    return output


def generate_triple(output, info, args, num_classes):
    _Pred_Triple = collections.namedtuple(
        "Pred_Triple", ["pred_rel", "rel_prob", "head_start_index", "head_end_index", "head_start_prob", "head_end_prob", "tail_start_index", "tail_end_index", "tail_start_prob", "tail_end_prob"]
    )
    pred_head_ent_dict = generate_span(output["head_start_logits"], output["head_end_logits"], info, args)
    pred_tail_ent_dict = generate_span(output["tail_start_logits"], output["tail_end_logits"], info, args)
    pred_rel_dict = generate_relation(output['pred_rel_logits'], info, args)
    triples = {}
    for sent_idx in pred_rel_dict:
        triples[sent_idx] = []
        for triple_id in range(args.num_generated_triples):
            pred_rel = pred_rel_dict[sent_idx][triple_id]
            pred_head = pred_head_ent_dict[sent_idx][triple_id]
            pred_tail = pred_tail_ent_dict[sent_idx][triple_id]
            triple = generate_strategy(pred_rel, pred_head, pred_tail, num_classes, _Pred_Triple)
            if triple:
                triples[sent_idx].append(triple)
    # print(triples)
    return triples


def generate_strategy(pred_rel, pred_head, pred_tail, num_classes, _Pred_Triple):
    if pred_rel.pred_rel != num_classes:
        if pred_head and pred_tail:
            for ele in pred_head:
                if ele.start_index != 0:
                    break
            head = ele
            for ele in pred_tail:
                if ele.start_index != 0:
                    break
            tail = ele
            return _Pred_Triple(pred_rel=pred_rel.pred_rel, rel_prob=pred_rel.rel_prob, head_start_index=head.start_index, head_end_index=head.end_index, head_start_prob=head.start_prob, head_end_prob=head.end_prob, tail_start_index=tail.start_index, tail_end_index=tail.end_index, tail_start_prob=tail.start_prob, tail_end_prob=tail.end_prob)
        else:
            return
    else:
        return


# def strict_strategy(pred_rel, pred_head, pred_tail, num_classes, _Pred_Triple):
#     if pred_rel.pred_rel != num_classes:
#         if pred_head and pred_tail:
#             if pred_head[0].start_index != 0 and pred_tail[0].start_index != 0:
#                 return _Pred_Triple(pred_rel=pred_rel.pred_rel, rel_prob=pred_rel.rel_prob, head_start_index=pred_head[0].start_index, head_end_index=pred_head[0].end_index, head_start_prob=pred_head[0].start_prob, head_end_prob=pred_head[0].end_prob, tail_start_index=pred_tail[0].start_index, tail_end_index=pred_tail[0].end_index, tail_start_prob=pred_tail[0].start_prob, tail_end_prob=pred_tail[0].end_prob)
#             else:
#                 return
#         else:
#             return
#     else:
#         return


def formulate_gold(target, info):
    sent_idxes = info["sent_idx"]
    gold = {}
    for i in range(len(sent_idxes)):
        gold[sent_idxes[i]] = []
        for j in range(len(target[i]["relation"])):
            gold[sent_idxes[i]].append(
                (target[i]["relation"][j].item(), target[i]["head_start_index"][j].item(), target[i]["head_end_index"][j].item(), target[i]["tail_start_index"][j].item(), target[i]["tail_end_index"][j].item())
            )
    return gold


def metric(pred, gold):
    assert pred.keys() == gold.keys()
    gold_num = 0
    rel_num = 0
    ent_num = 0
    right_num = 0
    pred_num = 0
    for sent_idx in pred:
        gold_num += len(gold[sent_idx])
        pred_correct_num = 0
        prediction = list(set([(ele.pred_rel, ele.head_start_index, ele.head_end_index, ele.tail_start_index, ele.tail_end_index) for ele in pred[sent_idx]]))
        pred_num += len(prediction)
        for ele in prediction:
            if ele in gold[sent_idx]:
                right_num += 1
                pred_correct_num += 1
            if ele[0] in [e[0] for e in gold[sent_idx]]:
                rel_num += 1
            if ele[1:] in [e[1:] for e in gold[sent_idx]]:
                ent_num += 1
        # if pred_correct_num != len(gold[sent_idx]):
        #     print("Gold: ", gold[sent_idx])
        #     print("Pred: ", prediction)
        #     print(pred[sent_idx])
    if pred_num == 0:
        precision = -1
        r_p = -1
        e_p = -1
    else:
        precision = (right_num + 0.0) / pred_num
        e_p = (ent_num + 0.0) / pred_num
        r_p = (rel_num + 0.0) / pred_num

    if gold_num == 0:
        recall = -1
        r_r = -1
        e_r = -1
    else:
        recall = (right_num + 0.0) / gold_num
        e_r = ent_num / gold_num
        r_r = rel_num / gold_num

    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)

    if (e_p == -1) or (e_r == -1) or (e_p + e_r) <= 0.:
        e_f = -1
    else:
        e_f = 2 * e_r * e_p / (e_p + e_r)

    if (r_p == -1) or (r_r == -1) or (r_p + r_r) <= 0.:
        r_f = -1
    else:
        r_f = 2 * r_p * r_r / (r_r + r_p)
    print("gold_num = ", gold_num, " pred_num = ", pred_num, " right_num = ", right_num, " relation_right_num = ", rel_num, " entity_right_num = ", ent_num)
    print("precision = ", precision, " recall = ", recall, " f1_value = ", f_measure)
    print("rel_precision = ", r_p, " rel_recall = ", r_r, " rel_f1_value = ", r_f)
    print("ent_precision = ", e_p, " ent_recall = ", e_r, " ent_f1_value = ", e_f)
    return {"precision": precision, "recall": recall, "f1": f_measure}


def num_metric(pred, gold):
    test_1, test_2, test_3, test_4, test_other = [], [], [], [], []
    for sent_idx in gold:
        if len(gold[sent_idx]) == 1:
            test_1.append(sent_idx)
        elif len(gold[sent_idx]) == 2:
            test_2.append(sent_idx)
        elif len(gold[sent_idx]) == 3:
            test_3.append(sent_idx)
        elif len(gold[sent_idx]) == 4:
            test_4.append(sent_idx)
        else:
            test_other.append(sent_idx)

    pred_1 = get_key_val(pred, test_1)
    gold_1 = get_key_val(gold, test_1)
    pred_2 = get_key_val(pred, test_2)
    gold_2 = get_key_val(gold, test_2)
    pred_3 = get_key_val(pred, test_3)
    gold_3 = get_key_val(gold, test_3)
    pred_4 = get_key_val(pred, test_4)
    gold_4 = get_key_val(gold, test_4)
    pred_other = get_key_val(pred, test_other)
    gold_other = get_key_val(gold, test_other)
    # pred_other = dict((key, vals) for key, vals in pred.items() if key in test_other)
    # gold_other = dict((key, vals) for key, vals in gold.items() if key in test_other)
    print("--*--*--Num of Gold Triplet is 1--*--*--")
    _ = metric(pred_1, gold_1)
    """print("--*--*--Num of Gold Triplet is 2--*--*--")
    _ = metric(pred_2, gold_2)
    print("--*--*--Num of Gold Triplet is 3--*--*--")
    _ = metric(pred_3, gold_3)
    print("--*--*--Num of Gold Triplet is 4--*--*--")
    _ = metric(pred_4, gold_4)
    print("--*--*--Num of Gold Triplet is greater than or equal to 5--*--*--")
    _ = metric(pred_other, gold_other)"""


def overlap_metric(pred, gold):
    normal_idx, multi_label_idx, overlap_idx = [], [], []
    for sent_idx in gold:
        triplets = gold[sent_idx]
        if is_normal_triplet(triplets):
            normal_idx.append(sent_idx)
        if is_multi_label(triplets):
            multi_label_idx.append(sent_idx)
        if is_overlapping(triplets):
            overlap_idx.append(sent_idx)
    pred_normal = get_key_val(pred, normal_idx)
    gold_normal = get_key_val(gold, normal_idx)
    pred_multilabel = get_key_val(pred, multi_label_idx)
    gold_multilabel = get_key_val(gold, multi_label_idx)
    pred_overlap = get_key_val(pred, overlap_idx)
    gold_overlap = get_key_val(gold, overlap_idx)
    print("--*--*--Normal Triplets--*--*--")
    _ = metric(pred_normal, gold_normal)
    print("--*--*--Multiply label Triplets--*--*--")
    _ = metric(pred_multilabel, gold_multilabel)
    print("--*--*--Overlapping Triplets--*--*--")
    _ = metric(pred_overlap, gold_overlap)



def is_normal_triplet(triplets):
    entities = set()
    for triplet in triplets:
        head_entity = (triplet[1], triplet[2])
        tail_entity = (triplet[3], triplet[4])
        entities.add(head_entity)
        entities.add(tail_entity)
    return len(entities) == 2 * len(triplets)


def is_multi_label(triplets):
    if is_normal_triplet(triplets):
        return False
    entity_pair = [(triplet[1], triplet[2], triplet[3], triplet[4]) for triplet in triplets]
    return len(entity_pair) != len(set(entity_pair))


def is_overlapping(triplets):
    if is_normal_triplet(triplets):
        return False
    entity_pair = [(triplet[1], triplet[2], triplet[3], triplet[4]) for triplet in triplets]
    entity_pair = set(entity_pair)
    entities = []
    for pair in entity_pair:
        entities.append((pair[0], pair[1]))
        entities.append((pair[2], pair[3]))
    entities = set(entities)
    return len(entities) != 2 * len(entity_pair)


def get_key_val(dict_1, list_1):
    dict_2 = dict()
    for ele in list_1:
        dict_2.update({ele: dict_1[ele]})
    return dict_2

def join_sent(sent):
  sent = " ".join(sent)
  sent = sent.replace(" ##", "")
  sent = sent.replace("[UNK]", "")
  return sent

def read_predictions(instance, predictions, read_instance):
  read = []

  for idx, prediction in enumerate(predictions.items()):
    relations = []
    _, prediction = prediction
    sent = join_sent(instance[idx])
    sent_arr = sent.split(" ")

    for triple in prediction:
      
      pred_rel, rel_prob, head_start_index, head_end_index, head_start_prob, head_end_prob, tail_start_index, tail_end_index, tail_start_prob, tail_end_prob = triple
      relations.append([join_sent(instance[idx][head_start_index:head_end_index+1]), join_sent(instance[idx][tail_start_index:tail_end_index+1]), read_instance(pred_rel)])
    
    read.append({ 'sentence': sent, 'relations': relations })

  return read