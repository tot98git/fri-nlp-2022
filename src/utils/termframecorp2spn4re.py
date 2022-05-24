import json
import re
def read(path, out):

    file1 = open(path, 'r')
    Lines = file1.readlines()

    
    f = open(out, "w", encoding="utf-8")

    # Strips the newline character
    for line in Lines:
        if line.strip():
            line = re.sub("[\(\[].*?[\)\]]", "is a", line)
            item = { "sentText": line.lower(), "relationMentions": [] }
            json.dump(item, f)
            f.write("\n")

annotated_definitions_path = "../../data/EN/new_train/test.txt"
output_file = "../../data/EN/new_train/new_test.json"

read(annotated_definitions_path, output_file)
