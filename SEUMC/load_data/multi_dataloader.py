# -*- coding: utf-8 -*-
# @FileName: multi_dataloader.py
"""
    Description:
        
"""
import numpy as np
import pickle
import torch
import torch.utils.data as data
import os
class Data(data.Dataset):
    def __init__(self, path, mode='train'):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        if os.path.basename(path) == 'iemocap.pkl':
            """
                {
                    "text": numpy_array,
                    "audio": numpy_array,
                    "vision": numpy_array,
                    "labels": numpy_array

                    audio,vision,text: (batch size,sequence length,feature dimension)
                    label:  (batch size,)
                }
            """
            if mode == 'train':
                dataset = data['train']
            elif mode == 'valid':
                dataset = data['valid']
            else:
                dataset = data['test']

            # print(dataset['labels'])

            text = dataset['text'].astype(np.float32)
            text[text == -np.inf] = 0
            self.text = torch.tensor(text)
            audio = dataset['audio'].astype(np.float32)
            audio[audio == -np.inf] = 0
            self.audio = torch.tensor(audio)
            vision = dataset['vision'].astype(np.float32)
            vision[vision == -np.inf] = 0
            self.vision = torch.tensor(vision)
            labels = torch.argmax(torch.tensor(dataset['labels']), -1)
            labels=labels.numpy()
            unique_patterns = np.unique(labels, axis=0)
            pattern_to_id = {tuple(pattern): i for i, pattern in enumerate(unique_patterns)}
            labels_single = np.array([pattern_to_id[tuple(row)] for row in labels])
            self.label = labels_single ##happy, sad, angry, neutral

            print(self.text.shape)
            # print(self.label[:, 0, 0].sum())
            # print(self.label[:, 1, 0].sum())
            # print(self.label[:, 2, 0].sum())
            # print(self.label[:, 3, 0].sum())
        else:
            """
            {
                "train": [
                    (words, visual, acoustic), label_id, segment,
                    ...
                ],
                "dev": [ ... ],
                "test": [ ... ]
            }
            """
            data_convert = convert_data_format(data)
            if mode == 'train':
                dataset = data_convert['train']
            elif mode == 'valid':
                dataset = data_convert['valid']
            else:
                dataset = data_convert['test']

            text = dataset['text'].astype(np.float32)
            text[text == -np.inf] = 0
            self.text = torch.tensor(text)
            audio = dataset['audio'].astype(np.float32)
            audio[audio == -np.inf] = 0
            self.audio = torch.tensor(audio)
            vision = dataset['vision'].astype(np.float32)
            vision[vision == -np.inf] = 0
            self.vision = torch.tensor(vision)
            self.label = dataset['labels'].astype(np.float32)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        vision = self.vision[index]
        audio = self.audio[index]
        label = torch.argmax(torch.tensor(self.label[index]), -1)
        return text, audio, vision, label


def convert_data_format(old_data):
    new_data = {"train": {}, "valid": {}, "test": {}}

    for split in ["train", "dev", "test"]:
        if split == "dev":
            new_split = "valid"
        else:
            new_split = split

        text, audio, vision, labels = [], [], [], []

        for sample in old_data[split]:
            (words, visual, acoustic), label_id, segment = sample
            text.append(words)
            audio.append(acoustic)
            vision.append(visual)
            labels.append(label_id)

        new_data[new_split]["text"] = np.array(text, dtype=np.float32)
        new_data[new_split]["audio"] = np.array(audio, dtype=np.float32)
        new_data[new_split]["vision"] = np.array(vision, dtype=np.float32)
        new_data[new_split]["labels"] = np.array(labels, dtype=np.float32)

    return new_data

if __name__ == '__main__':
    # with open("../data/iemocap.pkl", "rb") as f:
    #     old_data = pickle.load(f)

    dev_dataset = Data("../data/iemocap.pkl", 'train')

    print(dev_dataset)
