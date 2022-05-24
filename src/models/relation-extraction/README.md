# Preparation

1. Run `pip install -r requirements.txt`
2. Run `python code/loader.py` to download necessary dependencies
3. Run `python -m spacy download en_core_web_sm`

# Relation Extraction between Entities

All available cmdline arguments can be found in `code/config.py`

1. Extract features with `python extract_features.py [args]`
2. Next run `python word_embedding.py [args]`, to save Word2Vec model
3. Encode all extracted features with `python encoddings.py [args]`
4. To classify semantic relations run `python classifiers.py [args]` (relations are matched irrespective of direction)
