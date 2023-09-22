import argparse
import sys
sys.path.append(".")
print(sys.path)
from http.client import ImproperConnectionState
import torch
import torch.nn as nn
from model import t5_model
import numpy as np
from config import Config
import os, sys, math
from utils.checkpointing import CheckpointManager
import random
from pre_train_dataset import *
from utils import common_utils
from tqdm import tqdm
import sacrebleu
from fast_bleu import SelfBLEU
from seqeval.metrics import f1_score
import wandb
import pdb

def get_label_dict(label_path):

    label_dict = {}
    with open(label_path) as out:
        for l in out.readlines():
            l = l.strip()
            label_dict[l] = len(label_dict)
    return label_dict

def get_bio_seq(config, index2label, gen, input_seq):
    gen_words = gen.split()
    label = []
    word = []
    label_seq = []
    current_label = None
    has_error = False
    for index, w in enumerate(gen_words):
        if w.startswith("B-"):
            entity_label = w[2:]
            if current_label is not None:
                has_error = True
                break
            if len(w) == 2: 
                has_error = True
                break
            if entity_label not in index2label:
                has_error = True
                break
            entity_label = index2label[entity_label][2:]
            current_label = "B-" + entity_label
        elif w == "&&":
            if current_label is None:
                has_error = True
                break
            current_label = None
        else:
            word.append(w)
            if current_label is None:
                label.append('O')
            else:
                label.append(current_label)
                if current_label.startswith('B-'):
                    label_seq.append(current_label)
                    current_label = 'I-' + current_label[2:]

    if current_label is not None:
        has_error = True

    if not (len(word) == len(label) and len(word) > 0):
        has_error = True
    # input_seq = ' '.join(input_seq.split()[1:])
    # if not ' && '.join(label_seq) == input_seq:
    # 	has_error = True
    return word, label, has_error

def get_sentence_classification_output(label_list, gen):

    # import pdb
    # pdb.set_trace()

    text = gen.split('<extra_id_0>')
    if len(text) ==2:
        word = tokenizer.decode(tokenizer(text[1].strip())['input_ids'], skip_special_tokens=True)
        label = tokenizer.decode(tokenizer(text[0].strip())['input_ids'], skip_special_tokens=True).split(' and ')

        has_error = False

        for label_name in label:
            if label_name not in label_list or len(label) != 2:
                has_error=True
                break

        return word, label, has_error
    else:
        return None, None, True

def get_pair_sentence_classification_output(label_list, gen):
    has_error = True
    for l in label_list:
        if gen.startswith(l):
            has_error = False
            break

    words = gen.split()
    if len(words) > 3:
        if words[0] == 'not':
            sentence_pair = words[2:]
            label = ' '.join(words[:2])
        else:
            sentence_pair = words[1:]
            label = words[0]

        has_error = '[SEP]' not in sentence_pair or label not in label_list

        word = ' '.join(sentence_pair) 
        if '[SEP]' in sentence_pair:
            word = '\t'.join(word.split('[SEP]'))

        return word, label, has_error
    else:
        return None, None, True


def evaluation(config, eval_data, model, label_list, label_index, device, show_detail=False, output_path=None):
    model.eval()
    preds = None
    input_sentences = []
    gen_ner = []
    gen_ner_sp = []    
    gt_tags = []
    loss_list = []
    index2label = {v:k for (k,v) in label_index.items()}
    cnt = 0
    with torch.no_grad():
        for batch in tqdm(eval_data):
            cnt += 1

            max_len = int(batch['encoder_mask'].sum(1).max())
            batch['encoder_input_ids'] = batch['encoder_input_ids'][:,:max_len]
            batch['encoder_mask'] = batch['encoder_mask'][:,:max_len]

            # if cnt > 2:
            #     break
            for n in batch:
                if n not in eval_data.dataset.SKIP_ATTRIBUTES and batch[n] is not None:
                    batch[n] = batch[n].to(device)
            # import pdb
            # pdb.set_trace()
            if config.select_model_by_ppl:
                outputs = model(
                    input_ids=batch['encoder_input_ids'], 
                    task_ids=batch['task_index'],
                    attention_mask=batch['encoder_mask'], 
                    labels=batch['decoder_input_ids'],
                )
                loss = outputs.loss
                loss_list.append(loss.item())
            else:
                outputs = model.generate(
                    input_ids=batch['encoder_input_ids'], 
                    task_ids=batch['task_index'],
                    attention_mask=batch['encoder_mask'], 
                    max_length=config.max_length,
                    min_length=config.min_length,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=config.sample_num, 
                    do_sample=True,
                    top_p=0.9,
                    early_stopping=True
                )				

                outputs = outputs.view(len(batch['gt_x']), config.sample_num, -1)
                for i in range(len(batch['gt_x'])):
                    for j in range(config.sample_num):
                        sen = tokenizer.decode(outputs[i][j], skip_special_tokens=True)
                        gen_ner.append(sen)
                        # gen_ner_sp.append(tokenizer.decode(outputs[i][j], skip_special_tokens=False))
                        input_sentences.append(batch['gt_x'][i].split(';')[0].split(': ')[1])
                        # try:
                        #     input_sentences.append(batch['gt_x'][i].split(';')[1].split(': ')[1])                        
                        # except:
                        #     continue
                        gt_tags.append(batch['gt_y'][i])

    bio_labels = []
    bio_words = []
    new_F = 0
    if config.select_model_by_ppl:
        new_F = sum(loss_list) / len(loss_list)
        print("PPL %.2f" % new_F)
        new_F = -1 * new_F
    else:
        total_count = 0
        correct_count = 0


        for gen, input_sen in zip(gen_ner, input_sentences):
            total_count += 1
            if config.enable_sentence_classification:
                label = input_sen
                word = gen
                has_error = False
            elif config.enable_pair_sentence_classification:
                word, label, has_error = get_pair_sentence_classification_output(label_list, gen)
            else:
                word, label, has_error = get_bio_seq(config, index2label, gen, input_sen)
            if (not has_error) or (not config.enable_filtering_error):
                bio_labels.append(label)
                bio_words.append(word)
                
            correct_count += 1 if not has_error else 0
        print("Sentence Sample Successful Ratio %.2f, remaining %d instances" % (100 * correct_count / total_count, len(bio_words)))
        
        # sen_score = sacrebleu.corpus_bleu(gen_ner, [gt_tags]).score
        # print("BLEU %.2f" % sen_score)
        # new_F = sen_score

        # words_to_tags = {}
        # for word, tag in zip(bio_words, bio_labels):
        #     if config.enable_sentence_classification:
        #         word_key = word
        #         tag_value = tag
        #     else:
        #         word_key = ' '.join(word)
        #         tag_value = ' '.join(tag)
        #     if word_key not in words_to_tags:
        #         words_to_tags[word_key] = []

        #     for single_tag in tag_value:
        #         if single_tag not in words_to_tags[word_key]:
        #             words_to_tags[word_key].append(single_tag)

        # new_bio_labels, new_bio_words = [], []
        # for key in words_to_tags:
        #     if len(words_to_tags[key]) == 2:
        #         if config.enable_sentence_classification:
        #             new_bio_words.append(key)
        #             new_bio_labels.append(words_to_tags[key])
        #         else:
        #             new_bio_words.append(key.split())
        #             new_bio_labels.append(words_to_tags[key][0].split())
        # print("unique count %d" % len(new_bio_words))

    if output_path is not None:
        with open(output_path, 'w') as out:
            for gen, labels in zip(bio_words, bio_labels):
                if config.enable_sentence_classification or config.enable_pair_sentence_classification:
                    out.write("%s\t%s\n" % (gen, labels))
                else:
                    for g, l in zip(gen, labels):
                        out.write("%s %s\n" % (g, l))
                    out.write("\n")

    return new_F



parser = argparse.ArgumentParser("Train a MT5 for Machine Translation")
parser.add_argument(
    "--config", required=True, help="Path to a config file with all configuration parameters."
)
parser.add_argument(
    "--config-override",
    default=[],
    nargs="*",
    help="A sequence of key-value pairs specifying certain config arguments (with dict-like "
    "nesting) using a dot operator. The actual config will be updated and recorded in "
    "the serialization directory.",
)
parser.add_argument(
    "--serialization-dir",
    default=None,
    help="Path to a (non-existent) directory for serializing checkpoints and tensorboard logs.",
)
parser.add_argument(
    "--start-from-checkpoint",
    default=None,
    help="Path to load checkpoint and continue training [only supported for module_training].",
)
parser.add_argument(
    "--output-path",
    default=None,
    help="Path to save output captions",
)
parser.add_argument(
    "--pre-compute",
    action='store_true',
    help="Pre Compute",
)
parser.add_argument("--pre_trained_model",type=str,default="t5-large")
parser.add_argument("--wandb",default=False, action='store_true')
group = parser.add_mutually_exclusive_group()
group.add_argument('--train', action='store_true')
group.add_argument('--validation', action='store_true')
group.add_argument('--test', action='store_true')

if __name__ == "__main__":
    # import nltk
    # nltk.download("stopwords")
    # nltk.download("punkt")
    _A = parser.parse_args()
    _C = Config(_A.config, _A.config_override)
    _C.pre_trained_model = _A.pre_trained_model
    np.random.seed(_C.random_seed)
    random.seed(_C.random_seed)
    torch.manual_seed(_C.random_seed)
    torch.cuda.manual_seed_all(_C.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    label_dict = get_label_dict(_C.label_path)
    label_list = list(label_dict.keys())
    label_index = {}
    for key in label_dict:
        if key.startswith("B-"):
            label_index[key] = key[2:]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _C.device = device

    old_prefix_set_number = _C.prefix_set_number
    if old_prefix_set_number > 1 and _C.load_from_pretrained:
        _C.prefix_set_number = 1
    if _C.enable_full_finetune or _C.enable_adam_opt:
        tokenizer, model = t5_model.get_full_finetune_t5_model(_C)
    else:
        tokenizer, model = t5_model.get_t5_model(_C)

    if _A.start_from_checkpoint is not None:
        model.load_state_dict(torch.load(os.path.join(_A.start_from_checkpoint, 'model-best.pth'), map_location=torch.device('cpu'))['model'], strict=False)

    if old_prefix_set_number > 1 and _C.load_from_pretrained:
        model.update_prefix_embedding(old_prefix_set_number)
        _C.prefix_set_number = old_prefix_set_number

    total_parameter_count = 0
    trainable_parameter_count = 0
    for p in model.parameters():
        total_parameter_count += p.numel()
        if p.requires_grad:
            trainable_parameter_count += p.numel()
    print('Total Parameter Count %d' % total_parameter_count)
    print('Trainable Parameter Count %d' % trainable_parameter_count)
    
    print(_C)
    for arg in vars(_A):
        print("{:<20}: {}".format(arg, getattr(_A, arg)))

    if _C.enable_sentence_classification or _C.enable_pair_sentence_classification:
        dev_data = NLGMixSenClsDataset(_C, _C.dev_path, tokenizer, label_index)
    else:
        dev_data = NLGMixDataset(_C, _C.dev_path, tokenizer, label_index)
    dev_loader = nlg_get_data_loader(_C, dev_data, _C.batch_size, shuffle=False)

    if _C.enable_sentence_classification or _C.enable_pair_sentence_classification:
        test_data = NLGMixSenClsDataset(_C, _C.test_path, tokenizer, label_index)
    else:
        test_data = NLGMixDataset(_C, _C.test_path, tokenizer, label_index)
    test_loader = nlg_get_data_loader(_C, test_data, _C.batch_size, shuffle=False)


    model.parallelize()

    # pdb.set_trace()
    # model.to(device)
    # pdb.set_trace()

    
    if _A.validation or _A.test:
        assert _A.start_from_checkpoint is not None, "start-from-checkpoint cannot be None in validation or test mode"
        selected_data = dev_loader if _A.validation else test_loader
        evaluation(_C, selected_data, model, label_list, label_index, device, show_detail=True, output_path=_A.output_path)

    if _A.train:
        if _C.enable_sentence_classification or _C.enable_pair_sentence_classification:
            train_data = NLGMixSenClsDataset(_C, _C.train_path, tokenizer, label_index, is_training=True)
        else:
            train_data = NLGMixDataset(_C, _C.train_path, tokenizer, label_index, is_training=True)

        train_loader = nlg_get_data_loader(_C, train_data, _C.batch_size, shuffle=True)

        train_iter = iter(train_loader)

        if _C.num_training_steps == 0:
            _C.num_training_steps = int(len(train_iter) * _C.max_epoch / _C.gradient_accumulation_steps)
        epoch_num = math.ceil(_C.num_training_steps / _C.checkpoint_every_step)

        if _C.enable_adam_opt:
            optimizer = common_utils.build_optimizer(_C, model)
        elif _C.enable_full_finetune:
            optimizer = common_utils.build_t5_finetune_optimizer(_C, model)
        else:
            optimizer = common_utils.build_t5_optimizer(_C, model)
            
        os.makedirs(_A.serialization_dir, exist_ok=True)
        _C.dump(os.path.join(_A.serialization_dir, "config.yml"))

        checkpoint_manager = CheckpointManager(model, _A.serialization_dir, mode="max")

        eval_every = _C.checkpoint_every_step * _C.gradient_accumulation_steps
        total_step = 0
        best_test_performance = 0
        if _A.wandb:
            wandb.init(project="pda", entity="jordddan",name="balance_sample",job_type='training',reinit=True)
        for epoch in range(epoch_num):
            print('EPOCH %d / %d' % (epoch + 1, epoch_num))
            run_step = eval_every if total_step + eval_every < _C.num_training_steps * _C.gradient_accumulation_steps else  _C.num_training_steps * _C.gradient_accumulation_steps - total_step
            model.train()
            if _C.sampling_type == "class_balance":
                train_loader = nlg_get_data_loader(_C, train_data, _C.batch_size, shuffle=True)
                train_iter = iter(train_loader)

            with tqdm(total=math.ceil(run_step / _C.gradient_accumulation_steps), file=sys.stdout) as pbar:
                for step in range(run_step):
                    try:

                        batch = next(train_iter)
                    except:
                        train_iter = iter(train_loader)
                        batch = next(train_iter)

                    for n in batch:
                        if n not in train_loader.dataset.SKIP_ATTRIBUTES and batch[n] is not None:
                            batch[n] = batch[n].to(device)
                    total_step += 1



                    outputs = model(
                        input_ids=batch['encoder_input_ids'], 
                        task_ids=batch['task_index'],
                        attention_mask=batch['encoder_mask'], 
                        labels=batch['decoder_input_ids'],
                    )
                    loss = outputs.loss
                    loss = loss / _C.gradient_accumulation_steps
                    loss.backward()
                    if _A.wandb:
                        wandb.log({"loss":loss})
                    if (step + 1) % _C.gradient_accumulation_steps == 0:
                        optimizer.step()
                        if torch.cuda.is_initialized():
                            torch.cuda.synchronize()
                        pbar.set_description("loss %.2f" % (loss.item() * _C.gradient_accumulation_steps))
                        pbar.update(1)
                        optimizer.zero_grad()


            _score = evaluation(_C, dev_loader, model, label_list, label_index, device, output_path=_A.output_path)
            if _A.wandb:
                wandb.log({"PPL":_score})
            checkpoint_manager.step(_score)

    model.deparallelize()	