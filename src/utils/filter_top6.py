import numpy as np
import random
import re

RELATIONS = ["HAS_CAUSE", "HAS_RESULT", "HAS_FORM", "HAS_LOCATION", "HAS_ATTRIBUTE", "DEFINED_AS"]

def filter(input_file, output_file, input_file2=None, output_file2=None):

    f_in = open(input_file, "r", encoding="utf-8")
    f_out = open(output_file, "w", encoding="utf-8")

    while True:
        sentence = f_in.readline().replace("\n", "")
        if not sentence: break
        id, sentence = sentence.split("\t")
        relation = f_in.readline()
        relation_stripped = re.sub(r'\(.*\)\n', '', relation)
        if not sentence or not relation: break
        relation = relation.replace("\n", "")

        f_in.readline()
        f_in.readline()

        if relation_stripped not in RELATIONS: continue
        
        f_out.write("%s\t%s\n%s\nComment:\n\n" % (str(id), sentence, relation))

    f_in.close()
    f_out.close()

    if input_file2 is None and output_file2 is None: return

    f_in2 = open(input_file2, "r", encoding="utf-8")
    f_out2 = open(output_file2, "w", encoding="utf-8")

    while True:
        sentence = f_in2.readline().replace("\n", "")
        if not sentence: break
        id, sentence = sentence.split("\t")
        sentence = sentence.replace("<e1>", "[E1]")
        sentence = sentence.replace("</e1>", "[/E1]")
        sentence = sentence.replace("<e2>", "[E2]")
        sentence = sentence.replace("</e2>", "[/E2]")
        relations = f_in2.readline()
        if not sentence or not relations: break
        relations = relations.replace("\n", "")

        relations_relevant = set()

        for relation in relations.split(" "):
            relation_stripped = re.sub(r'\(.*\)\n*', '', relation)
            if relation_stripped in RELATIONS: relations_relevant.add(relation)

        f_in2.readline()
        f_in2.readline()
        
        if len(relations_relevant) == 0: continue

        f_out2.write("%s\t%s\n%s\nComment:\n\n" % (str(id), sentence, ' '.join(relations_relevant)))


input_file = "AdditionalAnnotatedDefinitions_SL_all_split.txt"
input_file2 = "AdditionalAnnotatedDefinitions_SL_all.txt"
output_file = "termframe_test_slo.txt"
output_file2 = "termframe_test_slo_v2.txt"
filter(input_file, output_file, input_file2, output_file2)