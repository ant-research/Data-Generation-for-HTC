from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import argparse
import os
from train import BertDataset
from eval import evaluate
# from model.contrast import ContrastModel
import pandas as pd
import pickle
from model.roberta import RobertaContrastModel


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
    data_path = os.path.join('roberta_data', args.data)

    if not hasattr(args, 'graph'):
        args.graph = False
    print(args)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    label_dict = torch.load(os.path.join(data_path, 'roberta_value_dict.pt'))
    label_dict = {i: tokenizer.decode(v, skip_special_tokens=True) for i, v in label_dict.items()}
    num_class = len(label_dict)

    dataset = BertDataset(device=device, pad_idx=tokenizer.pad_token_id, data_path=data_path)
    model = RobertaContrastModel.from_pretrained('roberta-base', num_labels=num_class,
                                          contrast_loss=args.contrast, graph=args.graph,
                                          layer=args.layer, data_path=data_path, multi_label=args.multi,
                                          lamb=args.lamb, threshold=args.thre)
    split = torch.load(os.path.join(data_path, 'split.pt'))
    test = Subset(dataset, split['test'])

    test = DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    model.load_state_dict(checkpoint['param'])

    model.to(device)
    model.eval()

    result={
        'model':[output_dir] 
    }

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
    result['macro_f1']=[macro_f1]
    result['micro_f1']=[micro_f1]

    results=pd.DataFrame(result)
    result_path = os.path.join(output_dir, 'result.csv')
    results.to_csv(result_path, mode='a', index=False, header=None)

    