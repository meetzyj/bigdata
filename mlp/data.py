from torch.utils.data import Dataset
import torch
import tqdm 
import random 

class Goodbooks(Dataset):
    def __init__(self, df, mode='training', negs = 99):
        super().__init__()
        #固定随机种子，确保每次数据构建的都是一样的，不然train和test之间存在gap
        random.seed(2024)
        torch.manual_seed(2024)

        self.df = df
        self.mode = mode

        self.book_nums = max(df['item_id'])+1
        self.user_nums = max(df['user_id'])+1

        self._init_dataset()
    
    def _init_dataset(self):
        self.Xs = []

        self.user_book_map = {}
        for i in range(self.user_nums):
            self.user_book_map[i] = []

        for index, row in self.df.iterrows():
            user_id, book_id = row
            self.user_book_map[user_id].append(book_id)
        
        if self.mode == 'training':
            for user, items in tqdm.tqdm(self.user_book_map.items()):
                for item in items[:-1]:
                    self.Xs.append((user, item, 1))
                    for _ in range(3):
                        while True:
                            neg_sample = random.randint(0, self.book_nums-1)
                            if neg_sample not in self.user_book_map[user]:
                                self.Xs.append((user, neg_sample, 0))
                                break

        elif self.mode == 'validation':
            for user, items in tqdm.tqdm(self.user_book_map.items()):
                if len(items) == 0:
                    continue
                self.Xs.append((user, items[-1]))
    
    def __getitem__(self, index):

        if self.mode == 'training':
            user_id, book_id, label = self.Xs[index]
            return user_id, book_id, label
        elif self.mode == 'validation':
            user_id, book_id = self.Xs[index]

            negs = list(random.sample(
                list(set(range(self.book_nums)) - set(self.user_book_map[user_id])),
                k=99
            ))
            return user_id, book_id, torch.LongTensor(negs)
    
    def __len__(self):
        return len(self.Xs)