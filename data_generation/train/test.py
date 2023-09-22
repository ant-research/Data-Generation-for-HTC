from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import sys
sys.path.append("./")
import argparse
import os
from train.train_classifier import BertDataset
from train.eval import evaluate
from model.contrast import ContrastModel
import pandas as pd
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch', type=int, default=32, help='Batch size.')
parser.add_argument('--name', type=str, required=True, help='Name of checkpoint. Commonly as DATASET-NAME.')
parser.add_argument('--extra', default='_macro', choices=['_macro', '_micro'], help='An extra string in the name of checkpoint.')
parser.add_argument('--output_dir', type=str, default=None, help='The dir of output.')

args = parser.parse_args()

if __name__ == '__main__':
    checkpoint = torch.load(os.path.join('checkpoints', args.name, 'checkpoint_best{}.pt'.format(args.extra)),
                            map_location='cpu')
    batch_size = args.batch
    device = args.device
    extra = args.extra
    output_dir = args.output_dir
    args = checkpoint['args'] if checkpoint['args'] is not None else args
    data_path = os.path.join('data', args.data)

    if not hasattr(args, 'graph'):
        args.graph = False
    print(args)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
    label_dict = {i: tokenizer.decode(v, skip_special_tokens=True) for i, v in label_dict.items()}
    num_class = len(label_dict)

    dataset = BertDataset(device=device, pad_idx=tokenizer.pad_token_id, data_path=data_path)
    model = ContrastModel.from_pretrained('bert-base-uncased', num_labels=num_class,
                                          contrast_loss=args.contrast, graph=args.graph,
                                          layer=args.layer, data_path=data_path, multi_label=args.multi,
                                          lamb=args.lamb, threshold=args.thre)
    split = torch.load(os.path.join(data_path, 'split.pt'))
    test = Subset(dataset, split['test'])

    # head_label = pickle.load(open('NLU_training_dataset/WebOfScience/head_label.pkl','rb'))
    # tail_label = pickle.load(open('NLU_training_dataset/WebOfScience/tail_label.pkl','rb'))


    # head_index = []
    # tail_index = []
    # label2index = dict()
    # for i in range(len(test)):
    #     label = torch.nonzero(test[i]['label']==1)[:,0]
        

    #     tail = False

    #     for item in label:

    #         if item.item() in tail_label:
    #             tail = True
        
    #         if item.item() in label2index:
    #             label2index[item.item()].append(i)
    #         else:
    #             label2index[item.item()] = [i]

    #     if tail:
    #         tail_index.append(i)
    #     else:
    #         head_index.append(i)

    testsets = []
    testsets.append(test)
    # testsets.append(Subset(test,head_index))
    # testsets.append(Subset(test,tail_index))
    # for i in range(num_class):
    #     testsets.append(Subset(test,label2index[i]))


    # test = DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    test_dataloaders = []
    for testset in testsets:
        test_dataloaders.append(DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn))



    model.load_state_dict(checkpoint['param'])

    model.to(device)
    model.eval()

    result={
        'model':[output_dir] 
    }

    idx2name={
        0:'all',
        1:'head',
        2:'tail',
    }

    for item in label_dict.items(): 
        idx2name[item[0]+3] = item[1]

    # for testset_idx in range(len(test_dataloaders)):
    for testset_idx in range(1):

        test = test_dataloaders[testset_idx]

        truth = []
        pred = []
        index = []
        slot_truth = []
        slot_pred = []


        pbar = tqdm(test)
        with torch.no_grad():
            for data, label, idx in pbar:
                padding_mask = data != tokenizer.pad_token_id
                output = model(data, padding_mask, return_dict=True, )
                for l in label:
                    t = []
                    for i in range(l.size(0)):
                        if l[i].item() == 1:
                            t.append(i)
                    truth.append(t)
                for l in output['logits']:
                    pred.append(torch.sigmoid(l).tolist())

        pbar.close()
        scores = evaluate(pred, truth, label_dict)
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']
        print('macro', macro_f1, 'micro', micro_f1)
        result[idx2name[testset_idx]+'_macro_f1']=[macro_f1]
        result[idx2name[testset_idx]+'_micro_f1']=[micro_f1]

    results=pd.DataFrame(result)
    result_path = os.path.join(output_dir, 'result.csv')
    results.to_csv(result_path, mode='a', index=False, header=None)

