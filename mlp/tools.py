import pandas as pd
from model import NCFModel
from data import Goodbooks
import torch
import pickle

def main():
    # 参数设置
    hidden_dim = 16
    train_dir = '../datasets/train.csv'
    device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
    print("using {} device".format(device))
    df = pd.read_csv(train_dir)
    traindataset = Goodbooks(df, 'training')
    model_path = '../checkpoints/mlp_13.pth'
    print(traindataset.user_nums)
    print(traindataset.book_nums)
    # 导出traindataset.user_book_map这个list
    with open('../datasets/user_book_map.pkl', 'wb') as f:
        pickle.dump(traindataset.user_book_map, f)

if __name__ == "__main__":
    main()