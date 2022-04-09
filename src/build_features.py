from bs4 import BeautifulSoup as bs

def build_wcl_features(data):
    """Takes a set of tagged sentences and returns a set of tags only paired with the hyponym-hypernym for each sentence. To help define the relation between elements, one might need the numbered argument.

    Input:  [NP_NP_American  NP_NP_Standard  NP_NP_Code PP_IN_for PP_NP_Information       PP_NP_Interchange (_( NP_NP_TARGET )_) <VERB>  VP_VBZ_is <GENUS> NP_DT_a NP_NN_character <HYPER> VP_VVG_encoding </HYPER> <REST>  VP_VVN_based PP_IN_on PP_DT_the PP_JJ_English   PP_NN_alphabet  SENT_.]

    Returns:  [(NP_NP  NP_NP NP_NP PP_IN PP_NP PP_NP .... VP_VVN PP_IN PP_DT PP_JJ PP_NN, HYPONYM_TAG, SEQUENCE OF HYPERNYM TAGS)]
    """

    
    def _build_wcl_feature(sent, numbered = True):
        sequence = []
        hyponym, hypernym = "", []
        isHypernymSeq = False
        index = 0

        for token in sent.split():
            isValid = False

            if token == "<GENUS>":
                isHypernymSeq = True

            elif token == "</HYPER>":
                isHypernymSeq = False
            else:
                *tags, word = token.split("_")

                if len(tags) != 0:
                    for c in word:
                        if c.isalpha():
                            isValid = True
                            break

                    tags = "_".join(tags)

                    if isValid:
                        if numbered:
                            tags += f"_{index}"

                        if isHypernymSeq:
                            hypernym.append(tags)
                        elif "TARGET" in token:
                            hyponym = tags
                        else:
                            sequence.append(tags)

                        index += 1

        return sequence, hyponym, hypernym
        
    return [_build_wcl_feature(item) for item in data]
