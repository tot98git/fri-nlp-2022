from config import args
import pandas as pd
from gensim.models import Word2Vec


def get_words(df):
    return [[words for words in sen.split()] for sen in df.Sentences]


def main(args):
    print("Loading extracted features")
    train_df = pd.read_csv(f'{args.output_path}/{args.task_name}/train_{args.task_name}.csv')
    test_df = pd.read_csv(f'{args.output_path}/{args.task_name}/test_{args.task_name}.csv')

    # Train word embeddings
    print("Training Word2Vec model")
    sentences= get_words(train_df) + get_words(test_df)
    model = Word2Vec(sentences, min_count=0)

    print(f"Saving Word2Vec model to {args.model_path}/word_model_{args.task_name}.bin")
    model.save(f'{args.model_path}/word_model_{args.task_name}.bin')


if __name__ == "__main__":
    print(args)
    main(args)