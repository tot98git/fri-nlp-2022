import numpy as np
import random
import re

RELATIONS = ["HAS_CAUSE", "HAS_RESULT", "HAS_FORM", "HAS_LOCATION", "HAS_ATTRIBUTE", "DEFINED_AS"]

def split_train_test(input_file, output_file, test_size=0.3, process_all=False, process_all_file=None):
    #samples_all = list(np.arange(0, 650, 1))

    # Top 6 relations all samples
    samples_all = [1, 2, 3, 5, 9, 10, 11, 13, 14, 15, 16, 18, 19, 21, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91, 94, 97, 98, 99, 100, 101, 102, 103, 
                    104, 105, 106, 107, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 149, 150, 152, 153, 154, 156, 157, 158, 159, 160, 161, 162, 163, 167, 168, 169, 170, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 184, 185, 186, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 201, 202, 203, 204, 206, 207, 208, 209, 211, 212, 214, 215, 216, 217, 218, 219, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 241, 243, 244, 245, 246, 247, 248, 249, 251, 252, 253, 255, 256, 259, 260, 261, 263, 264, 265, 266, 267, 268, 269, 270, 271, 273, 274, 276, 277, 278, 280, 281, 282, 284, 285, 286, 287, 288, 290, 292, 293, 294, 295, 298, 299, 300, 301, 302, 303, 304, 305, 306, 308, 309, 310, 311, 312, 313, 314, 316, 317, 318, 319, 320, 321, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 361, 363, 364, 365, 366, 367, 368, 369, 370, 371, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 391, 392, 394, 395, 396, 397, 398, 399, 400, 402, 403, 404, 405, 406, 407, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 478, 479, 480, 481, 482, 483, 
                    484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 541, 542, 543, 544, 545, 546, 547, 548, 549, 552, 553, 554, 557, 558, 559, 563, 564, 565, 566, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 634, 635, 636, 637, 638, 640, 641, 642, 643, 644, 645, 646, 647, 648]
    #samples_all = [0, 1, 2, 3, 6, 8, 9, 10, 11, 12, 13, 14, 16, 21, 22, 23, 24, 26, 27, 28, 31, 32, 34, 36, 37, 39, 40, 41, 43, 45, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 61, 63, 64, 65, 67, 68, 69, 71, 72, 74, 75, 76, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]

    random.seed(42)
    #samples_test = random.sample(samples_all, int(552*test_size))
    samples_test = []
    print(samples_test)

    f_in = open(input_file, "r", encoding="utf-8")
    f_out_train = open(output_file + "_train.txt", "w", encoding="utf-8")
    f_out_test = open(output_file + "_test.txt", "w", encoding="utf-8")

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

        if int(id) not in samples_all or relation_stripped not in RELATIONS: continue
        
        if int(id) in samples_test: 
            f_out_test.write("%s\t%s\n%s\nComment:\n\n" % (str(id), sentence, relation))
        else: 
            f_out_train.write("%s\t%s\n%s\nComment:\n\n" % (str(id), sentence, relation))

    f_in.close()
    f_out_train.close()
    f_out_test.close()

    if not process_all_file: return

    f_test_all = open(process_all_file, "r", encoding="utf-8")
    f_out_all = open(output_file + "_test_v2.txt", "w", encoding="utf-8")

    while True:
        sentence = f_test_all.readline().replace("\n", "")
        if not sentence: break
        id, sentence = sentence.split("\t")
        sentence = sentence.replace("<e1>", "[E1]")
        sentence = sentence.replace("</e1>", "[/E1]")
        sentence = sentence.replace("<e2>", "[E2]")
        sentence = sentence.replace("</e2>", "[/E2]")
        relations = f_test_all.readline()
        if not sentence or not relations: break
        relations = relations.replace("\n", "")

        relations_relevant = set()

        for relation in relations.split(" "):
            relation_stripped = re.sub(r'\(.*\)\n*', '', relation)
            if relation_stripped in RELATIONS: relations_relevant.add(relation)

        f_test_all.readline()
        f_test_all.readline()
        
        if len(relations_relevant) == 0 or int(id) not in samples_all: continue

        if len(relations_relevant) > 0 and int(id) in samples_test:
            f_out_all.write("%s\t%s\n%s\nComment:\n\n" % (str(id), sentence, ' '.join(relations_relevant)))


input_file = "AnnotatedDefinitions_EN_all_split.txt"
input_all_file = "AdditionalAnnotatedDefinitions_EN_all.txt"
output_file = "termframe_original"
split_train_test(input_file, output_file, test_size=1)