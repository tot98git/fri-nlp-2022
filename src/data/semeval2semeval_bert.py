import re

def convert(mapping, input_file, output_file):
    """
    Maps standard SemEval form to SemEval form needed for R-BERT
    """
    f_in = open(input_file, "r", encoding="utf-8")
    f_out = open(output_file, "w", encoding="utf-8")

    count = 0

    while True:

        sentence = f_in.readline().replace("\n", "").lower()
        sentence = sentence.replace("<e1>", "[E11] ")
        sentence = sentence.replace("</e1>", " [E12]")
        sentence = sentence.replace("<e2>", "[E21] ")
        sentence = sentence.replace("</e2>", " [E22]")
        relations = f_in.readline().split(" ")
        if len(relations) != 1:
            f_in.readline()
            f_in.readline()
            print("Skipped")
            continue
        if not sentence or not relations: break
        relation = relations[0].replace("\n", "")
        relation_mapped = mapping.get(relation)

        f_in.readline()
        f_in.readline()

        if not relation_mapped: continue
        
        f_out.write("%s\t%s\t%s\t%s\n" % (sentence, *relation_mapped))

        count += 1


    f_in.close()
    f_out.close()

    print("%d sentences processed." % count)

mapping = {
    "Cause-Effect(e1,e2)": [9, "cause\teffect", 1],
    "Cause-Effect(e2,e1)": [10, "effect\tcause", 2],
    "Content-Container(e1,e2)": [17, "content\tcontainer", 1],
    "Content-Container(e2,e1)": [18, "container\tcontent", 2],
    "Entity-Origin(e1,e2)": [13, "entity\torigin", 1],
    "Entity-Origin(e2,e1)": [14, "origin\tentity", 2],
    "Component-Whole(e1,e2)": [11, "component\twhole", 1],
    "Component-Whole(e2,e1)": [12, "whole\tcomponent", 2],
    "Member-Collection(e1,e2)": [15, "member\tcollection", 1],
    "Member-Collection(e2,e1)": [16, "collection\tmember", 2]
}

annotated_file = "termframe_SE_train_sl.txt"
output_file = "termframe_SE_train_sl.tsv"

convert(mapping, annotated_file, output_file)