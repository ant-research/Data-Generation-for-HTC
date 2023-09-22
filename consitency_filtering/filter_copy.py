from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import argparse
import os
from train_aug import BertDataset,AugDataset,WOSAugDataset
from eval import evaluate
from model.newcontrast import ContrastModel2
from model.contrast import ContrastModel
import numpy as np






parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch', type=int, default=32, help='Batch size.')
parser.add_argument('--name', type=str, required=True, help='Name of checkpoint. Commonly as DATASET-NAME.')
parser.add_argument('--extra', default='_macro', choices=['_macro', '_micro'], help='An extra string in the name of checkpoint.')
parser.add_argument('--input_dir', type=str, default=None, help='the data need to be filtered')
parser.add_argument('--output_dir', type=str, default=None, help='the filtered data')

args = parser.parse_args()

if __name__ == '__main__':
    checkpoint = torch.load(os.path.join('checkpoints', args.name, 'checkpoint_best{}.pt'.format(args.extra)),
                            map_location='cpu')
    batch_size = args.batch
    device = args.device
    extra = args.extra
    input_dir = args.input_dir
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


    if args.data == 'WebOfScience':
        aug_dataset = WOSAugDataset(device=device, pad_idx=tokenizer.pad_token_id, data_path=input_dir, tokenizer=tokenizer)    
    else:
        aug_dataset = AugDataset(device=device, pad_idx=tokenizer.pad_token_id, data_path=input_dir, tokenizer=tokenizer,task=args.data,num_class=num_class)

    model = ContrastModel.from_pretrained('bert-base-uncased', num_labels=num_class,
                                          contrast_loss=args.contrast, graph=args.graph,
                                          layer=args.layer, data_path=data_path, multi_label=args.multi,
                                          lamb=args.lamb, threshold=args.thre)
    split = torch.load(os.path.join(data_path, 'split.pt'))

    test = DataLoader(aug_dataset, batch_size=batch_size, shuffle=False, collate_fn=aug_dataset.collate_fn)
    model.load_state_dict(checkpoint['param'])

    model.to(device)

    truth = []
    pred = []
    index = []
    slot_truth = []
    slot_pred = []

    model.eval()
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
    # print('pred',pred)
    # print('truth',truth)
    # print('label_dict',label_dict)
    # scores = evaluate(pred, truth, label_dict)
    # macro_f1 = scores['macro_f1']
    # micro_f1 = scores['micro_f1']
    # print('macro', macro_f1, 'micro', micro_f1)


    pred_idx = []
    for sample_predict in pred:
        np_sample_predict = np.array(sample_predict, dtype=np.float32)
        sample_predict_descent_idx = np.argsort(-np_sample_predict)
        sample_predict_id_list = []

        for j in range(len(sample_predict)):
            if np_sample_predict[sample_predict_descent_idx[j]] > 0.5:
                sample_predict_id_list.append(sample_predict_descent_idx[j])    
        
        pred_idx.append(sample_predict_id_list)


    # keyword_checkpoint = torch.load('checkpoint_best_keyword.pt',
    #                         map_location='cpu')
    # # utils.seed_torch(3)


    # args = keyword_checkpoint['args'] if keyword_checkpoint['args'] is not None else args

    # # if args.data == "WebOfScience":
    # #     args.data = "wos"
    # data_path = os.path.join('data', args.data)

    # if not hasattr(args, 'graph'):
    #     args.graph = False




      

    # model = ContrastModel2.from_pretrained('bert-base-uncased', num_labels=num_class,
    #                                       contrast_loss=args.contrast, graph=True,
    #                                       layer=args.layer, data_path=data_path, multi_label=args.multi,
    #                                       lamb=args.lamb, threshold=args.thre, args=args)                                          
    # split = torch.load(os.path.join(data_path, 'split.pt'))

    # test = DataLoader(aug_dataset, batch_size=batch_size, shuffle=False, collate_fn=aug_dataset.collate_fn)
    # model.load_state_dict(keyword_checkpoint['param'])

    # model.to(device)

    # truth = []
    # pred = []
    # index = []
    # slot_truth = []
    # slot_pred = []

    # model.eval()
    # pbar = tqdm(test)
    # with torch.no_grad():
    #     for data, label, idx in pbar:
    #         padding_mask = data != tokenizer.pad_token_id
    #         output = model(data, padding_mask,labels=label, return_dict=True, )
    #         for l in label:
    #             t = []
    #             for i in range(l.size(0)):
    #                 if l[i].item() == 1:
    #                     t.append(i)
    #             truth.append(t)
    #         for l in output['logits']:
    #             pred.append(torch.sigmoid(l).tolist())

    # pbar.close()



    # keyword_pred_idx = []
    # for sample_predict in pred:
    #     np_sample_predict = np.array(sample_predict, dtype=np.float32)
    #     sample_predict_descent_idx = np.argsort(-np_sample_predict)
    #     sample_predict_id_list = []

    #     for j in range(len(sample_predict)):
    #         if np_sample_predict[sample_predict_descent_idx[j]] > 0.5:
    #             sample_predict_id_list.append(sample_predict_descent_idx[j])    
        
    #     keyword_pred_idx.append(sample_predict_id_list)






    lines = open(input_dir, 'r').readlines()

    filtered_data = []
    for i in range(len(lines)):
        # if set(pred_idx[i]) == set(truth[i]) or set(keyword_pred_idx[i]) == set(truth[i]):
        if set(pred_idx[i]) == set(truth[i]):            
            filtered_data.append(lines[i])

    f = open(output_dir, 'w')
    f.writelines(filtered_data)
    f.close()