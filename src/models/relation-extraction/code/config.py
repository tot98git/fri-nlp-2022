from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--train_path', 
    default='../data/Semeval_2010/train.txt', 
    type=str, 
    help='Path to the dataset'
)

parser.add_argument('--test_path', 
    default='../data/Semeval_2010/test.txt', 
    type=str, 
    help='Path to the dataset'
)
parser.add_argument('--output_path', 
    default='../output/', 
    type=str, 
    help='Path to the output folder'
)

parser.add_argument('--task_name', 
    default='semeval', 
    type=str, 
    help='Name of the task'
)

parser.add_argument('--model_path', 
    default='../model', 
    type=str, 
    help='Path to the model'
)

args = parser.parse_args()