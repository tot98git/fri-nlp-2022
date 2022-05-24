import pandas as pd

def replace_entity_tags(input_file, output_file):
    f_in = open(input_file, "r", encoding="utf-8")
    f_out = open(output_file, "w", encoding="utf-8")

    count = 0

    while True:
        sentence = f_in.readline()
        if not sentence: break

        sentence = sentence.replace("<e1>", "[E1]")
        sentence = sentence.replace("</e1>", "[/E1]")
        sentence = sentence.replace("<e2>", "[E2]")
        sentence = sentence.replace("</e2>", "[/E2]")

        f_out.write(sentence)

input_file = "AnnotatedDefinitions_EN_single_test.txt"
output_file = "AnnotatedDefinitions_EN_single_test_v2.txt"

replace_entity_tags(input_file, output_file)