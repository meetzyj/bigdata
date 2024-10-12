import pandas as pd
from model import NCFModel
from data import Goodbooks
import torch
import pickle
from tqdm import tqdm

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i: i+n]

def main():
    # 参数设置
    hidden_dim = 16
    batch_size = 512
    test_dir = '../datasets/test.csv'
    # train_dir = '../datasets/train.csv'
    device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
    print("using {} device".format(device))
    # df = pd.read_csv(train_dir)
    # traindataset = Goodbooks(df, 'training')
    # validdataset = Goodbooks(df, 'validation')
    model_path = '../checkpoints/mlp_13.pth'
    user_nums = 53424
    book_nums = 10000
    with open('../datasets/user_book_map.pkl', 'rb') as f:
        user_book_map = pickle.load(f)

    model = NCFModel(hidden_dim, user_nums, book_nums).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    df = pd.read_csv(test_dir)

    user_for_test = df['user_id'].tolist()

    predict_item_id = []
    f = open('../results/mlp_submission.csv', 'w', encoding='utf-8')

    for user in tqdm(user_for_test, desc="Processing users"):
        #将用户已经交互过的物品排除
        user_visited_items = user_book_map[user]
        # print(user_visited_items)
        items_for_predict = list(set(range(book_nums)) - set(user_visited_items))
        # print(items_for_predict)
        
        results = []
        user = torch.LongTensor([user]).to(device)

        for batch in chunks(items_for_predict, batch_size):
            batch = torch.LongTensor(batch).unsqueeze(0).to(device)

            result = model(user.repeat(1, batch.size(1)), batch).view(-1).detach().cpu()
            results.append(result)
        
        results = torch.cat(results, dim=-1)
        # 取分数最高的十本书给用户进行推荐
        predict_item_id = (-results).argsort()[:10]
        list(map(lambda x: f.write('{},{}\n'.format(user.cpu().item(), x)), predict_item_id))

    f.flush()
    f.close()

if __name__ == "__main__":
    main()