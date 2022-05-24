from config import args
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from xgboost import XGBClassifier


def test_metrics(model, test_x, test_y):
    '''
    Only relation matches irrespective of direction. 
    We find accuracy, precision, recall and f1 macro scores
    '''
    test_predict= pd.Series(model.predict(test_x))
    
    test_y_relation = test_y.apply(lambda y: y.split('(')[0])
    test_predict_relation= test_predict.apply(lambda y: y.split('(')[0])

    accuracy_relation= accuracy_score(test_y_relation, test_predict_relation)
    precision_relation= precision_score(test_y_relation, test_predict_relation, average='macro')
    recall_relation= recall_score(test_y_relation, test_predict_relation, average='macro')
    f1_relation= f1_score(test_y_relation, test_predict_relation, average='macro')
    
    return accuracy_relation, precision_relation, recall_relation, f1_relation


def run_model(train_x, train_y, test_x, test_y, features):
    models = [DecisionTreeClassifier(), SVC(), XGBClassifier()]
    names = ["Decision_Tree", "SVM", "XGBoost"]
    results, model_names= [], []

    for name, model in zip(names, models):
        model.fit(train_x, train_y)
        result= test_metrics(model, test_x, test_y)
        results.append(result)
        model_names.append(name + '_' + features)
        
        print('Metrics for {} model is {}'.format(name, result))
        
    model_df= pd.DataFrame(model_names, columns=['model_name'])
    result_df= pd.DataFrame(results, columns=['accuracy_relation', 'precision_relation','recall_relation','f1_relation'])
    df= pd.concat([model_df, result_df], axis=1)

    return df


def main(args):
    print("Loading encoded features")
    train_y= pd.read_csv(f'{args.output_path}/{args.task_name}/train_{args.task_name}.csv').Relations
    test_y= pd.read_csv(f'{args.output_path}/{args.task_name}/test_{args.task_name}.csv').Relations
    # pos and enr
    train_x_pos_enr= pd.read_csv(f'{args.output_path}/{args.task_name}/train_{args.task_name}_encodings_pos_enr.csv')
    test_x_pos_enr= pd.read_csv(f'{args.output_path}/{args.task_name}/test_{args.task_name}_encodings_pos_enr.csv')
    # e1 and e2
    train_x_e1_e2= pd.read_csv(f'{args.output_path}/{args.task_name}/train_{args.task_name}_encodings_e1_e2.csv')
    test_x_e1_e2= pd.read_csv(f'{args.output_path}/{args.task_name}/test_{args.task_name}_encodings_e1_e2.csv')
    # SDP
    train_x_SDP= pd.read_csv(f'{args.output_path}/{args.task_name}/train_{args.task_name}_encodings_SDP.csv')
    test_x_SDP= pd.read_csv(f'{args.output_path}/{args.task_name}/test_{args.task_name}_encodings_SDP.csv')
    # words in between 
    train_x_words_in_between= pd.read_csv(f'{args.output_path}/{args.task_name}/train_{args.task_name}_enc_words_in_between.csv')
    test_x_words_in_between= pd.read_csv(f'{args.output_path}/{args.task_name}/test_{args.task_name}_enc_words_in_between.csv')
    # root words
    train_x_root= pd.read_csv(f'{args.output_path}/{args.task_name}/train_{args.task_name}_enc_root.csv')
    test_x_root= pd.read_csv(f'{args.output_path}/{args.task_name}/test_{args.task_name}_enc_root.csv')
    # pos_enr_e1_e2
    train_x_pos_enr_e1_e2= pd.concat([train_x_pos_enr, train_x_e1_e2], axis=1)
    test_x_pos_enr_e1_e2= pd.concat([test_x_pos_enr, test_x_e1_e2], axis=1)

    # pos, enr, e1, e2, root_e1_e2, words in between, SDP
    train_pos_enr_e1e2_root_between= pd.read_csv(f'{args.output_path}/{args.task_name}/train_{args.task_name}_pos_enr_e1e2_root_between.csv')
    test_pos_enr_e1e2_root_between= pd.read_csv(f'{args.output_path}/{args.task_name}/test_{args.task_name}_pos_enr_e1e2_root_between.csv')

    print("Classifing relations based on pos1, pos2, enr1, enr2")
    results= run_model(train_x_pos_enr, train_y, test_x_pos_enr, test_y, 'pos_enr')

    print("Classifing relations based on e1 and e2")
    results1= run_model(train_x_e1_e2, train_y, test_x_e1_e2, test_y, 'e1_e2')

    print("Classifing relations based on pos, enr, e1 and e2")
    results2= run_model(train_x_pos_enr_e1_e2, train_y, test_x_pos_enr_e1_e2, test_y, 'pos_enr_e1_e2')

    print("Classifing relations based on shortest dependency path")
    results3= run_model(train_x_SDP, train_y, test_x_SDP, test_y, 'SDP')

    print("Classifing relations based on words in between")
    results4 = run_model(train_x_words_in_between, train_y, test_x_words_in_between, test_y, 'words_in_between')

    print("Classifing relations based on root words")
    results5 = run_model(train_x_root, train_y, test_x_root, test_y, 'root_words')

    print("Classifing relations based on pos, enr, e1, e2, root words, words in between")
    results6= run_model(train_pos_enr_e1e2_root_between, train_y, test_pos_enr_e1e2_root_between, test_y, 'pos_enr_e1e2_root_between')

    print(f"Saving results to {args.output_path}/{args.task_name}/")
    result = pd.concat([results, results1, results2, results3, results4, results5, results6])
    result.to_csv(f'{args.output_path}/{args.task_name}/results_{args.task_name}.csv', index=False)

if __name__ == "__main__":
    print(args)
    main(args)