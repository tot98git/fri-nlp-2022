import re

def convert(mapping, annotated_file, output_file):
    """
    Maps TermFrame relations to SemEval relations
    """
    f_ann = open(annotated_file, "r", encoding="utf-8")
    f_out = open(output_file, "w", encoding="utf-8")

    count = 0

    while True:

        sentence = f_ann.readline()
        relations = f_ann.readline().split(" ")
        if not sentence or not relations: break
        relations_mapped = []
        for relation in relations:
            relation_stripped = re.sub(r"\(.*\)\n*", "", relation)
            relation_mapped = mapping.get(relation_stripped)
            if relation_mapped is None: continue
            for rel_mapped in relation_mapped:
                relations_mapped.append(relation.replace(relation_stripped, rel_mapped).replace("\n", ""))

        f_ann.readline()
        f_ann.readline()

        #if len(relations_mapped) == 0: continue # Skip entities with 0 mapped relations
        if len(relations_mapped) != 1: continue # Skip entities with 0 mapped relations or more than 1 relation

        f_out.write(sentence)
        f_out.write(" ".join(relations_mapped))
        f_out.write("\nComment: \n\n")

        count += 1


    f_ann.close()
    f_out.close()

    print("%d sentences processed." % count)


mapping = {
        "HAS_CAUSE": ["Cause-Effect"],
        "HAS_RESULT": ["Cause-Effect"],
        "HAS_COMPOSITION": ["Content-Container", "Entity-Origin"],
        "CONTAINS": ["Content-Container", "Component-Whole", "Member-Collection"],
        "HAS_LOCATION": ["Entity-Origin"]
}

annotated_file = "AnnotatedDefinitions_EN_all.txt"
output_file = "AnnotatedDefinitions_EN_mapped_single.txt"

convert(mapping, annotated_file, output_file)