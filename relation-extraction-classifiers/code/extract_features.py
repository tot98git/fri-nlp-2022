from config import args
import spacy
import pandas as pd
import numpy as np
from spacy import displacy
import networkx as nx
from nltk.corpus import stopwords
import nltk
import en_core_web_sm
import os

nlp = en_core_web_sm.load()
stopwords = stopwords.words('english')

def extract_sentences_and_relations(file):
    '''
    read the input file line by line and extract sentences and relations from it. 
    '''
    sentences=[]
    relations=[]

    with open(file) as f:
        i=1
        for line in f:
            if i==1:
                sen=line.split('"')[1]
                sentences.append(sen)

            elif i==2:
                relation= line.strip()
                relations.append(relation)
                
            elif i%4==0:
                i=0

            i+=1
    return sentences, relations


def get_entity_index(sen):
    sen_list= sen.split()

    for i, word in enumerate(sen_list):
        if word=='<e1>':
            start1=i

        elif word=='</e1>':
            end1=i

        if word=='<e2>':
            start2=i

        elif word=='</e2>':
            end2=i
    
    # get e1 and e2
    e1= " ".join(sen_list[start1+1 : end1])
    e2= " ".join(sen_list[start2+1 : end2])
    
    return e1, e2, [start1, end1, start2, end2]


def get_sen_without_entity(sen):
    sen_list= sen.split()
    sen_without_entity= " ".join([token for token in sen_list if token not in {'<e1>','</e1>', '<e2>', '</e2>'}]) 
    return sen_without_entity


def get_pos(sen_without_entity):
    sen_pos = [token.pos_ for token in nlp(sen_without_entity)]
    return sen_pos


def get_root(entity):
    # create a span object that has property .root
    doc = nlp(entity)
    sen= list(doc.sents)[0]
    return str(sen.root)


def get_enr(entity):
    for ent in nlp(entity).ents:
        return str(ent.label_)


def shortest_dep_path(sen, root_e1, root_e2):
    doc = nlp(sen)
    
    #print dependency tree 
    #displacy.render(doc,jupyter=True)

    # Load spacy's dependency tree into a networkx graph
    edges = []
    for token in doc:
        for child in token.children:
            edges.append(('{0}'.format(token.lower_),
                          '{0}'.format(child.lower_)))
            
    graph = nx.Graph(edges)
    entity1 = root_e1.lower()
    entity2 = root_e2.lower()
    
    try:
        out = str(" ".join(nx.shortest_path(graph, source=entity1, target=entity2)[1:-1]))
        
    except (nx.NetworkXNoPath,  nx.NodeNotFound) as e:
        out= None
    
    return out


def extract_nlp_features(df):
    df['sen_without_entity']= df.Sentences.apply(get_sen_without_entity)
    df['sen_pos']= df.sen_without_entity.apply(get_pos)
    df['pos_e1']= df.apply(lambda row: str(row.sen_pos[row.position[0]]), axis=1)
    df['pos_e2']= df.apply(lambda row: str(row.sen_pos[row.position[2]-2]), axis=1)
    df['enr_e1']= df.e1.apply(get_enr)
    df['enr_e2']= df.e2.apply(get_enr)
    df['root_e1']= df.e1.apply(get_root)
    df['root_e2']= df.e2.apply(get_root)
    df['shortest_dep_path'] = df.apply(lambda row: shortest_dep_path(row.sen_without_entity, row.root_e1, row.root_e2), axis=1)
    
    return df


def get_words_in_between(sen):
    '''
    get the words in between entities which are not stop words
    '''
    words= sen.sen_without_entity.split()

    if isinstance(sen.position, list):
        position = sen.position
    else:
        position = sen.position[1:-2].split(', ')
    
    words_in_between= words[int(position[1]) - 1 : int(position[2]) - 2]

    return " ".join([word for word in words_in_between if word not in stopwords])    


def get_sen_without_entity(sen):
    '''
    remove entity tags from the sentence and get its lemma form, return string
    '''
    sen_list= sen.split()
    sen_without_entity= " ".join([token for token in sen_list if token not in {'<e1>','</e1>', '<e2>', '</e2>'}]) 
    words=[str(token.lemma_) for token in nlp(sen_without_entity)]

    return " ".join(words)


def main(args):
    train_file = args.train_path
    test_file = args.test_path

    print("Starting feature extraction")

    # Get sentences and relations dataframe
    print("Extracting sentances and relations")
    train_sent, train_rel= extract_sentences_and_relations(train_file)
    test_sent, test_rel= extract_sentences_and_relations(test_file)

    train_dict = {'Sentences': train_sent, 'Relations': train_rel}
    test_dict = {'Sentences': test_sent, 'Relations': test_rel}
    train_df= pd.DataFrame(train_dict, columns=['Sentences', 'Relations'])
    test_df= pd.DataFrame(test_dict, columns=['Sentences', 'Relations'])

    # Extract e1, e2 and its position
    print("Extracting e1, e2 and its position")
    train_df[['e1', 'e2', 'position']]= train_df.Sentences.apply(lambda sen: get_entity_index(sen)).apply(pd.Series)
    test_df[['e1', 'e2', 'position']]= test_df.Sentences.apply(lambda sen: get_entity_index(sen)).apply(pd.Series)

    # Replace empty e1 and e2 with na and then dropna 
    train_df.replace('', np.nan, inplace=True)
    train_df.dropna(inplace=True)

    test_df.replace('', np.nan, inplace=True)
    test_df.dropna(inplace=True)

    # Extract NLP features
    print("Extracting NLP features (this may take some time)")
    train_df= extract_nlp_features(train_df)
    test_df= extract_nlp_features(test_df)

    # Drop missing shortest dep path rows
    train_df.dropna(subset=['shortest_dep_path'], inplace= True)
    test_df.dropna(subset=['shortest_dep_path'], inplace= True)

    # Get words in between in lemma form and after removing stop words
    print("Getting words in between in lemma form and after removing stop words")
    train_df['words_in_between']= train_df.apply(get_words_in_between, axis=1)
    test_df['words_in_between']= test_df.apply(get_words_in_between, axis=1)

    train_df['sen_without_entity']= train_df.Sentences.apply(get_sen_without_entity)
    test_df['sen_without_entity']= test_df.Sentences.apply(get_sen_without_entity)

    # Saving extracted features to defined folder
    if not os.path.exists(os.path.join(args.output_path, args.task_name)):
        print(f"Creating folder {os.path.join(args.output_path, args.task_name)}")
        os.makedirs(os.path.join(args.output_path, args.task_name))

    print(f"Saving extracted features to {args.output_path}/{args.task_name}/train_{args.task_name}.csv")
    train_df.to_csv(f'{args.output_path}/{args.task_name}/train_{args.task_name}.csv', index=False)
    test_df.to_csv(f'{args.output_path}/{args.task_name}/test_{args.task_name}.csv', index=False)


if __name__ == "__main__":
    print(args)
    main(args)