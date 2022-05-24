from config import args
import pandas as pd
import category_encoders as ce
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
import pickle


def encode(sen, model):
    '''
    each word in 100 dimension, sum up these vectors for each word in sentence, to get one final 100 size vector. 
    '''
    encoding=[0]*100
    words= str(sen).split()
    for word in words:
        if word != 'nan':
            if word in model.wv.key_to_index: 
                encoding += model.wv[word]
    return encoding


def encode_entity(df, model):
    enc_e1= df.e1.apply(encode, model=model).apply(pd.Series)
    enc_e2= df.e2.apply(encode, model=model).apply(pd.Series)
    
    return pd.concat([enc_e1, enc_e2], axis=1)


def encode_root(df, model):
    df_enc_root_e1= df.root_e1.apply(encode, model=model).apply(pd.Series)
    df_enc_root_e2= df.root_e2.apply(encode, model=model).apply(pd.Series)
    enc_root= pd.concat([df_enc_root_e1, df_enc_root_e2], axis=1)
    
    return enc_root


def main(args):
    print("Loading extracted features")
    train_df = pd.read_csv(f'{args.output_path}/{args.task_name}/train_{args.task_name}.csv')
    test_df = pd.read_csv(f'{args.output_path}/{args.task_name}/test_{args.task_name}.csv')

    print("Loading Word2Vec model")
    model = Word2Vec.load(f'{args.model_path}/word_model_{args.task_name}.bin')

    # Encoding pos_e1, pos_e2, enr_e1, enr_e2
    print("Encoding pos_e1, pos_e2, enr_e1, enr_e2")
    train_features= train_df[['pos_e1', 'pos_e2', 'enr_e1', 'enr_e2']]
    test_features= test_df[['pos_e1', 'pos_e2', 'enr_e1', 'enr_e2']]

    enc= ce.BinaryEncoder()
    enc= enc.fit(train_features)
    train_enc= enc.transform(train_features)
    test_enc= enc.transform(test_features)

    file = open(f"{args.output_path}/{args.task_name}/ce.obj","wb")
    pickle.dump(enc, file)
    file.close()

    train_enc.drop(['pos_e1_0', 'pos_e2_0', 'enr_e1_0', 'enr_e2_0'], axis=1, inplace=True)
    test_enc.drop(['pos_e1_0', 'pos_e2_0', 'enr_e1_0', 'enr_e2_0'], axis=1, inplace=True)

    train_enc.to_csv(f'{args.output_path}/{args.task_name}/train_{args.task_name}_encodings_pos_enr.csv', index=False)
    test_enc.to_csv(f'{args.output_path}/{args.task_name}/test_{args.task_name}_encodings_pos_enr.csv', index=False)

    # Encoding e1, e2 using Word2Vec
    train_encodings_e1_e2= encode_entity(train_df, model)
    test_encodings_e1_e2= encode_entity(test_df, model)

    train_encodings_e1_e2.to_csv(f'{args.output_path}/{args.task_name}/train_{args.task_name}_encodings_e1_e2.csv', index=False)
    test_encodings_e1_e2.to_csv(f'{args.output_path}/{args.task_name}/test_{args.task_name}_encodings_e1_e2.csv', index=False)

    # Encoding shortest dependency path
    train_encodings_SDP= train_df.shortest_dep_path.apply(encode, model=model).apply(pd.Series)
    test_encodings_SDP= test_df.shortest_dep_path.apply(encode, model=model).apply(pd.Series)

    train_encodings_SDP.to_csv(f'{args.output_path}/{args.task_name}/train_{args.task_name}_encodings_SDP.csv', index=False)
    test_encodings_SDP.to_csv(f'{args.output_path}/{args.task_name}/test_{args.task_name}_encodings_SDP.csv', index=False)

    # Encode labels
    enc= LabelEncoder()
    enc_label= enc.fit(train_df.Relations)
    train_enc_label= enc.transform(train_df.Relations)
    test_enc_label= enc.transform(test_df.Relations)

    print(f"Saving encodins to {args.output_path}/{args.task_name}/")
    pd.Series(train_enc_label).to_csv(f'{args.output_path}/{args.task_name}/train_{args.task_name}_label_enc.csv', index=False)
    pd.Series(test_enc_label).to_csv(f'{args.output_path}/{args.task_name}/test_{args.task_name}_label_enc.csv', index=False)

    #Encode words in between
    train_enc_words_in_between= train_df.words_in_between.apply(encode, model=model).apply(pd.Series)
    test_enc_words_in_between= test_df.words_in_between.apply(encode, model=model).apply(pd.Series)

    train_enc_words_in_between.to_csv(f'{args.output_path}/{args.task_name}/train_{args.task_name}_enc_words_in_between.csv', index=False)
    test_enc_words_in_between.to_csv(f'{args.output_path}/{args.task_name}/test_{args.task_name}_enc_words_in_between.csv', index=False)

    # Encode root words
    train_enc_root= encode_root(train_df, model)
    test_enc_root= encode_root(test_df, model)

    train_enc_root.to_csv(f'{args.output_path}/{args.task_name}/train_{args.task_name}_enc_root.csv', index=False)
    test_enc_root.to_csv(f'{args.output_path}/{args.task_name}/test_{args.task_name}_enc_root.csv', index=False)

    # Encode all features
    train_pos_enr_e1e2_root_between= pd.concat([train_enc, train_encodings_e1_e2, train_encodings_SDP, train_enc_words_in_between, train_enc_root], axis=1)
    test_pos_enr_e1e2_root_between= pd.concat([test_enc, test_encodings_e1_e2, test_encodings_SDP, test_enc_words_in_between, test_enc_root], axis=1)

    train_pos_enr_e1e2_root_between.to_csv(f'{args.output_path}/{args.task_name}/train_{args.task_name}_pos_enr_e1e2_root_between.csv', index=False)
    test_pos_enr_e1e2_root_between.to_csv(f'{args.output_path}/{args.task_name}/test_{args.task_name}_pos_enr_e1e2_root_between.csv', index=False)


if __name__ == "__main__":
    print(args)
    main(args)