import re

def process_unannotated(input_file, output_file):
    f_in = open(input_file, "r", encoding="utf-8")
    f_out = open(output_file, "w", encoding="utf-8")

    count = 1

    while True:
        sentence = f_in.readline().strip("\n ")
        if not sentence: break
        f_in.readline()

        sentence_split = re.split('\[.*\]', sentence)
        sentence_split[0] = sentence_split[0].strip(". ")
        sentence_split[1] = sentence_split[1].strip()
        sentence_split[1] = re.sub("\. \(.*\)", ".", sentence_split[1])
        sentence_split[1] = re.sub("^A ", "a ", sentence_split[1])
        sentence_split[1] = re.sub("^An ", "an ", sentence_split[1])

        f_out.write("%s\t%s\n" % (str(count), " is ".join(sentence_split)))

        count += 1

input_file = "../../data/Model input files/en/unannotated_raw.txt"
output_file = "unannotated.txt"

process_unannotated(input_file, output_file)