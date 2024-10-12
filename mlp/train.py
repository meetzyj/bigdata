from model import NCFModel
from data import Goodbooks
import torch
import random 
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import logging

def main():
    # 参数设置
    train_dir = "../datasets/train.csv"
    save_dir = "../checkpoints/mlp_"
    log_file = "../log/mlp_20e.log"
    seed = 114514
    np.random.seed(seed)
    random.seed(seed)
    BATCH_SIZE = 512
    hidden_dim = 16
    epochs = 20
    device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
    print("using {} device".format(device))

     # 设置日志
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("using {} device".format(device))

    #建立训练和验证dataloader
    df = pd.read_csv(train_dir)
    traindataset = Goodbooks(df, 'training')
    validdataset = Goodbooks(df, 'validation')
    trainloader = DataLoader(traindataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=32)
    validloader = DataLoader(validdataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=32)

    model = NCFModel(hidden_dim, traindataset.user_nums, traindataset.book_nums).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.BCELoss()

    loss_for_plot = []
    hits_for_plot = []

    # 训练
    for epoch in range(epochs):

        losses = []
        for index, data in enumerate(trainloader):
            user, item, label = data
            user, item, label = user.to(device), item.to(device), label.to(device).float()
            y_ = model(user, item).squeeze()

            loss = crit(y_, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().item()) 

        hits = []
        for index, data in enumerate(validloader):
            user, pos, neg = data
            pos = pos.unsqueeze(1)
            all_data = torch.cat([pos, neg], dim=-1)
            output = model.predict(user.to(device), all_data.to(device)).detach().cpu()
            
            for batch in output:
                if 0 not in (-batch).argsort()[:10]:
                    hits.append(0)
                else:
                    hits.append(1)
        logging.info('Epoch {} finished, average loss {}, hits@20 {}'.format(epoch, sum(losses)/len(losses), sum(hits)/len(hits)))
        print('Epoch {} finished, average loss {}, hits@20 {}'.format(epoch, sum(losses)/len(losses), sum(hits)/len(hits)))
        loss_for_plot.append(sum(losses)/len(losses))
        hits_for_plot.append(sum(hits)/len(hits))
        
        torch.save(model.state_dict(), save_dir+str(epoch)+'.pth')

if __name__ == "__main__":
    main()