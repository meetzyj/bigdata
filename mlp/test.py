import pandas as pd
from model import NCFModel
from data import Goodbooks
import torch


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i, i+n]

def main():
    # 参数设置
    hidden_dim = 16
    test_dir = '../datasets/test.csv'
    device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
    print("using {} device".format(device))
    model_path = '../checkpoints/mlp_20e.pth'
    model = NCFModel(hidden_dim, traindataset.user_nums, traindataset.book_nums).to(device)
    model.load_state_dict(torch.load(model_path))

    df = pd.read_csv(test_dir)
    traindataset = Goodbooks(df, 'training')
    validdataset = Goodbooks(df, 'validation')

    user_for_test = df['user_id'].tolist()

    predict_item_id = []
    f = open('../results/mlp_submission.csv', 'w', encoding='utf-8')

    for user in user_for_test:
        #将用户已经交互过的物品排除
        user_visited_items = traindataset.user_book_map[user]
        items_for_predict = list(set(range(traindataset.book_nums)) - set(user_visited_items))
        
        results = []
        user = torch.Tensor([user]).to(device)

        for batch in chunks(items_for_predict, 64):
            batch = torch.Tensor(batch).unsqueeze(0).to(device)

            result = model(user, batch).view(-1).detach().cpu()
            results.append(result)
        
        results = torch.cat(results, dim=-1)
        predict_item_id = (-results).argsort()[:10]
        list(map(lambda x: f.write('{},{}\n'.format(user.cpu().item(), x)), predict_item_id))

    f.flush()
    f.close()
