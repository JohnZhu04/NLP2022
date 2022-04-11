import torch
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmbedBranch(nn.Module):
    def __init__(self, input_dim, embed_dim, metric_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(embed_dim, metric_dim)
        )
    
    def forward(self, x):
        # print("x.shape = {}".format(x.shape))
        return self.fc(x)


class ImageSentenceEmbeddingNetwork(nn.Module):
    def __init__(self, embed_dim, metric_dim):
        super().__init__()
        text_feat_dim = 100  # dim of word_vec
        image_feat_dim = 2048

        # 資料番号
        self.name_of_item_branch = EmbedBranch(text_feat_dim, embed_dim, metric_dim)  
        # コレクション名
        self.name_of_collection_branch = EmbedBranch(text_feat_dim, embed_dim, metric_dim)
        # 備考  
        self.notes_branch = EmbedBranch(text_feat_dim, embed_dim, metric_dim)
        # concated texts
        self.text_branch = nn.Linear(3 * metric_dim, metric_dim)
        # image
        self.image_branch = EmbedBranch(image_feat_dim, embed_dim, metric_dim)

        
    def forward(self, name_of_item, name_of_collection, notes, images):
        name_of_item = self.name_of_item_branch(name_of_item)
        name_of_collection = self.name_of_collection_branch(name_of_collection)
        notes = self.notes_branch(notes)
        
        # concat_text = torch.cat((name_of_item, name_of_collection, notes), dim=1) # (64, 3*metric_dim)
        concat_text = torch.cat((name_of_item[None, :], name_of_collection[None, :], notes[None, :]), dim=1) # (64, 3*metric_dim)

        textual_representations = self.text_branch(concat_text)
        image_representations = self.image_branch(images)
        
        # normalize
        textual_representations = torch.nn.functional.normalize(textual_representations, dim=-1).reshape(-1)
        image_representations = torch.nn.functional.normalize(image_representations, dim=-1).reshape(-1)
        
        return textual_representations, image_representations


class LSTMEncoder(nn.Module):
    def __init__(self, input_size=320, embed_size=2048, hidden_size=2048, output_size=2048, num_layers=1, dropout=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        # input_seq = input_seq.view(len(input_seq), 1, -1)
        embedded = self.embedding(input_seq)
        # print("input_seq.shape = {}".format(input_seq.shape))
        # print("embedded.shape = {}".format(embedded.shape))
        output, (hidden, cell) = self.lstm(embedded)
        output = self.fc(output[:, -1, :])
        return output



class LSTMImageSentenceEmbeddingNetwork(nn.Module):
    def __init__(self, embed_dim, metric_dim):
        super().__init__()
        image_feat_dim = 2048

        # 資料番号
        self.name_of_item_branch = LSTMEncoder()  
        # コレクション名
        self.name_of_collection_branch = LSTMEncoder()  
        # 備考  
        self.notes_branch = LSTMEncoder()
        # concated texts
        self.text_branch = nn.Linear(3 * metric_dim, metric_dim)
        # image
        self.image_branch = EmbedBranch(image_feat_dim, embed_dim, metric_dim)

        
    def forward(self, name_of_item, name_of_collection, notes, images):
        name_of_item = self.name_of_item_branch(name_of_item)
        name_of_collection = self.name_of_collection_branch(name_of_collection)
        notes = self.notes_branch(notes)
        
        # concat_text = torch.cat((name_of_item, name_of_collection, notes), dim=1) # (64, 3*metric_dim)
        concat_text = torch.cat((name_of_item[None, :], name_of_collection[None, :], notes[None, :]), dim=1) # (64, 3*metric_dim)

        textual_representations = self.text_branch(concat_text)
        image_representations = self.image_branch(images)
        
        # normalize
        textual_representations = torch.nn.functional.normalize(textual_representations, dim=-1).reshape(-1)
        image_representations = torch.nn.functional.normalize(image_representations, dim=-1).reshape(-1)
        
        return textual_representations, image_representations