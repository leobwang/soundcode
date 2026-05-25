import argparse
import pprint
from models import build_or_load_gen_model
from data_utils import get_data, get_pos_data, CustomDataset, CLDataset,DatasetCA
import os
import time
import math
from tqdm import tqdm
import json
from itertools import cycle

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig, RobertaTokenizer, set_seed
from torch import cuda

#import matplotlib.pyplot as plt

from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState



def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)


def get_rank():
    return int(os.environ.get("RANK", "0"))


def get_world_size():
    return os.environ.get("CUDA_VISIBLE_DEVICES", "0").count(',') + 1


def validation(args, model, loader):
    model.eval()
    device = args.accelerator.device
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for data in tqdm(loader, total=len(loader), desc="Eval: "):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            if 'codebert' in args.load:
                outputs, loss = model(ids, mask)
                fin_targets.extend(targets.cpu().detach().tolist())
                outputs = outputs.view(-1, 1)
                tmp_outputs = torch.sigmoid(outputs).cpu().detach().tolist()
                tmp_outputs = np.array(tmp_outputs) >= 0.5
                fin_outputs.extend([1 if p[0] == True else 0 for p in tmp_outputs])
            else:
                outputs, loss = model(text1=ids, mask1=mask)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                # no sigmoid
                fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


def validation_multigpu(args, model, loader, tokenizer=None):
    model.eval()
    loader = args.accelerator.prepare(loader)
    args.accelerator.print(
        f"The size of valid dataloader: {len(loader)}, {len(loader.dataset)}. (Batch size: {args.batch_size}, Word size: {torch.cuda.device_count()})")
    device = args.accelerator.device
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        index = 0
        for data in tqdm(loader, total=len(loader), desc="Eval: "):
            ids = data['ids'].to(device, dtype=torch.long)
            # mask = data['mask'].to(device, dtype = torch.long)
            mask = ids.ne(tokenizer.pad_token_id)
            targets = data['targets'].to(device, dtype=torch.float)

            if 'codet5ca' in args.load or 'deepseekca' in args.load or 'codebert' in args.load:
                targets = data['targets'].to(device, dtype=torch.long)
                ids2 = data['ids_2'].to(device, dtype=torch.long)
                # mask2 = data['mask_2'].to(device, dtype = torch.long)
                mask2 = ids2.ne(tokenizer.pad_token_id)
                outputs, loss = model(text1=ids, mask1=mask, label1=targets, text2=ids2, mask2=mask2)
                all_targets.extend(args.accelerator.gather_for_metrics(targets.cpu().detach().numpy().tolist()))
                all_predictions.extend(args.accelerator.gather_for_metrics(outputs.cpu().detach().numpy().tolist()))
            else:
                outputs, loss = model(text1=ids, mask1=mask)
                all_targets.extend(args.accelerator.gather_for_metrics(targets.cpu().detach().numpy().tolist()))
                all_predictions.extend(args.accelerator.gather_for_metrics(outputs.cpu().detach().numpy().tolist()))

    all_predictions = all_predictions[:len(loader.dataset)]
    all_targets = all_targets[:len(loader.dataset)]

    return all_predictions, all_targets


def run(args, model, tokenizer, optimizer, data_list):
    t0 = time.time()
    if args.accelerator.is_main_process:
        f_summary = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    training_loader, validation_loader, testing_loader = data_list
    device = args.accelerator.device
    args.do_train = True
    training_classifier = args.train_classifier
    print(f'Training config: is traing {args.do_train}, is validating {args.do_eval}, \
            is testing {args.do_test}, is training classifier {training_classifier}, load model {args.load}\
                length of training_loader {len(training_loader)}')
    if args.do_train:
        best_accf1 = 0
        # 2026-05-24 patch (claude-code): respect args.start_epoch so resumed
        # runs use the correct epoch index for output paths and don't clobber
        # the prior run's checkpoint files.
        for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
            model.train()

            bar = tqdm(training_loader, total=len(training_loader), desc="Training")
            for idx, data in enumerate(bar):
                ids = data['ids'].to(device, dtype=torch.long)
                #print("ids2=========================", ids)
                # mask = data['mask'].to(device, dtype = torch.long)
                mask = ids.ne(tokenizer.pad_token_id)
                #print("mask2==========================", mask)
                if 'codet5ca' in args.load or 'deepseekca' in args.load or 'codebert' in args.load:
                    targets = data['targets'].to(device, dtype=torch.long)
                    if 'contrastive' in args.output_dir.lower():
                        ids2 = data['ids_code1'].to(device, dtype=torch.long)
                        ids3 = data['ids_code2'].to(device, dtype=torch.long)
                        mask2 = ids2.ne(tokenizer.pad_token_id)
                        mask3 = ids3.ne(tokenizer.pad_token_id)
                        outputs, loss = model(text1=ids, mask1=mask, label1=targets, text2=ids2, mask2=mask2,
                                              text3=ids3, mask3=mask3, training_classifier=training_classifier)
                    else:
                        ids2 = data['ids_2'].to(device, dtype=torch.long)

                        mask2 = ids2.ne(tokenizer.pad_token_id)
                        outputs, loss = model(text1=ids, mask1=mask, label1=targets, text2=ids2, mask2=mask2,
                                              training_classifier=training_classifier)
                else:
                    print('training normal')
                    targets = data['targets'].to(device, dtype=torch.long)
                    outputs, loss = model(text1=ids, mask1=mask, label1=targets,
                                          training_classifier=training_classifier)

                if args.grad_acc_steps > 1:
                    loss = loss / args.grad_acc_steps
                if (idx + 1) % args.grad_acc_steps == 0:
                    optimizer.zero_grad()
                    # loss.sum().backward()
                    args.accelerator.backward(loss)
                    optimizer.step()

                    if args.accelerator.is_main_process:
                        bar.set_description(
                            "[{}] Step: {}/{} Train loss {}".format(epoch + 1, idx + 1, len(training_loader),
                                                                    loss * args.grad_acc_steps))
                else:
                    with args.accelerator.no_sync(model):
                        args.accelerator.backward(loss)
            save_path = os.path.join(args.output_dir, f'checkpoinss/{epoch}')
            os.makedirs(save_path, exist_ok=True)
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), os.path.join(save_path, f'model_{epoch}.bin'))

            # args.do_eval = True

            if args.do_eval:
                # valid_outs, valid_tgts = validation(args, model, validation_loader)
                valid_outs, valid_tgts = validation_multigpu(args, model, validation_loader, tokenizer)
                print("Length of valid_outs: ", len(valid_outs))
                print("Length of valid_tgts: ", len(valid_tgts))
                if args.accelerator.is_main_process:
                    accuracy = metrics.accuracy_score(valid_tgts, valid_outs)
                    f1_score = metrics.f1_score(valid_tgts, valid_outs, average='weighted')
                    prediction_folder = os.path.join(args.output_dir, 'prediction')
                    os.makedirs(prediction_folder, exist_ok=True)
                    with open(os.path.join(prediction_folder, f'valid_res_{epoch}.json'), 'w') as f:
                        for p, g in zip(valid_outs, valid_tgts):
                            f.write(json.dumps({"pred": p, "gold": g}) + '\n')
                    accf1 = accuracy + f1_score
                    logger.info(f'Accuracy:  {accuracy}, F1:  {f1_score}, Best Acc+F1:  {max(accf1, best_accf1)}')
                    if accf1 > best_accf1:
                        not_inc_cnt = 0
                        f_summary.write("[%d] Best acc+f1 changed into %.2f (accuracy: %.2f, f1_score: %.2f)\n" % (
                        epoch, accuracy + f1_score, accuracy, f1_score))
                        save_path = os.path.join(args.output_dir, 'checkpoint-best')
                        os.makedirs(save_path, exist_ok=True)
                        best_accf1 = accuracy + f1_score
                        model_to_save = model.module if hasattr(model, 'module') else model
                        # model_to_save = args.accelerator.unwrap_model(model_to_save)
                        torch.save(model_to_save.state_dict(), os.path.join(save_path, 'model.bin'))
                        logger.info("Save Model!")
                    else:
                        not_inc_cnt += 1
                        f_summary.write(
                            "[%d] Best acc+f1 (%.2f) does not drop changed for %d epochs, cur acc+f1: %.2f (accuracy: %.2f, f1_score: %.2f)\n" % (
                                epoch, best_accf1, not_inc_cnt, accuracy + f1_score, accuracy, f1_score))

                        # save
                        save_path = os.path.join(args.output_dir, 'checkpoins')
                        os.makedirs(save_path, exist_ok=True)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save(model_to_save.state_dict(), os.path.join(save_path, f'model_{epoch}.bin'))

                        if not_inc_cnt >= 3:
                            stop_early_str = "[%d] Early stop as not_inc_cnt=%d\n" % (epoch, not_inc_cnt)
                            logger.info(stop_early_str)
                            args.accelerator.set_trigger()
                if args.accelerator.check_trigger():
                    break

    print("Start testing.......")
    # args.do_test = True
    if args.do_test:

        save_path = os.path.join(args.output_dir, 'checkpoint-best')
        prediction_folder = os.path.join(args.output_dir, 'prediction')
        model = model.module if hasattr(model, 'module') else model
        # model = args.accelerator.unwrap_model(model)
        model.load_state_dict(torch.load(os.path.join(save_path, 'model.bin'), map_location='cpu'))
        # outputs, targets = validation(args, model, testing_loader)
        outputs, targets = validation_multigpu(args, model, testing_loader, tokenizer)
        print("Length of outputs: ", len(outputs))
        print("Length of targets: ", len(targets))
        # outputs = np.array(outputs) >= 0.5
        if args.accelerator.is_main_process:
            accuracy = metrics.accuracy_score(targets, outputs)
            f1_score = metrics.f1_score(targets, outputs, average='weighted')
            logger.info(f"Accuracy Score = {accuracy}")
            logger.info(f"F1 Score = {f1_score}")
            with open(os.path.join(prediction_folder, 'test_res.json'), 'w') as f:
                for p, g in zip(outputs, targets):
                    # p = 1 if p[0]==True else 0
                    f.write(json.dumps({"pred": p, "gold": g}) + '\n')
            logger.info("Finish and take {}".format(get_elapse_time(t0)))
            f_summary.write("Testing results: Accuracy: {}, F1: {}".format(accuracy, f1_score))
            f_summary.write("Finish and take {}".format(get_elapse_time(t0)))
            f_summary.close()


def main(args):
    accelerator = Accelerator()
    # mixed_precision = 'bf16'
    # args.mixed_precision = mixed_precision
    # accelerator = Accelerator(mixed_precision=mixed_precision)
    AcceleratorState().deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size
    AcceleratorState().deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"] = args.grad_acc_steps
    args.accelerator = accelerator

    # Model
    if 'codebert' in args.load:
        args.model_type = 'codebert'
        args.model_name_or_path = 'microsoft/codebert-base'
    # 2026-05-24 patch (claude-code): args.model_type was set from args.load
    # (e.g. 'deepseek-coder-1.3b-base'), but MODEL_CLASSES only keys on
    # 'deepseek'. Fixed to use the bare 'deepseek' key. Also allow
    # SEMGUARD_BACKBONE env var to override the model dir so we can point at
    # a local HF download without editing this file.
    elif 'deepseek' in args.output_dir.lower():
        args.model_type = 'deepseek'
        args.model_name_or_path = os.environ.get(
            'SEMGUARD_BACKBONE', 'deepseek-coder-1.3b-base'
        )
    elif 'codet5' == args.load:
        args.model_type = 'codet5'
        args.model_name_or_path = 'Salesforce/codet5-base'


    config, model, tokenizer = build_or_load_gen_model(args)
    logger.info("Load model from {}".format(args.model_name_or_path))
    accelerator.log(model)
    model = model.to(accelerator.device)

    # Config
    MAX_LEN = args.max_source_len
    TRAIN_BATCH_SIZE = args.batch_size
    VALID_BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    model, optimizer = accelerator.prepare(model, optimizer)

    # 2026-05-24 patch (claude-code): load weights from a prior run's checkpoint
    # if --resume_from_checkpoint is set. We unwrap the model first (mirroring
    # the test-time load at the bottom of run() / run_gen()) so the keys line
    # up regardless of DeepSpeed/DDP wrapping.
    if args.resume_from_checkpoint:
        ckpt_path = args.resume_from_checkpoint
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"--resume_from_checkpoint not found: {ckpt_path}")
        logger.info("Resuming from checkpoint: %s", ckpt_path)
        state_dict = torch.load(ckpt_path, map_location='cpu')
        unwrapped = accelerator.unwrap_model(model)
        missing, unexpected = unwrapped.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys when loading checkpoint: %s", missing[:5])
        if unexpected:
            logger.warning("Unexpected keys when loading checkpoint: %s", unexpected[:5])
        logger.info("Checkpoint loaded; resuming at epoch %d.", args.start_epoch)
        del state_dict

    train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, "drop_last": True}
    valid_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': True}
    test_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': False}

    # 2026-05-24 patch (claude-code): upstream hardcoded validation_set =
    # testing_set = None in every branch, which then crashed
    # validation_multigpu(len(None)) when --do_eval=1. DatasetCA already
    # supports data_type='valid' / 'test' (it tokenizes via the same path as
    # 'train' but skips the rank-sharded caching). Construct them on demand
    # so per-epoch validation works.
    if 'codet5ca' in args.load or 'deepseekca' in args.load or 'codebert' in args.load:
        if 'contrastive' in args.output_dir.lower():
            print("Prepare contrastive learning dataset......")
            training_set = CLDataset(args, tokenizer, MAX_LEN, data_type='train_pair_cl_sampled')
            validation_set = testing_set = None
        else:
            print("Prepare single with cross attention samples dataset......")
            training_set = DatasetCA(args, tokenizer, MAX_LEN, data_type='train')
            validation_set = DatasetCA(args, tokenizer, MAX_LEN, data_type='valid') if args.do_eval else None
            testing_set = DatasetCA(args, tokenizer, MAX_LEN, data_type='test') if args.do_test else None
    else:
        print("Prepare single with cross attention samples dataset......")
        training_set = DatasetCA(args, tokenizer, MAX_LEN, data_type='train')
        validation_set = DatasetCA(args, tokenizer, MAX_LEN, data_type='valid') if args.do_eval else None
        testing_set = DatasetCA(args, tokenizer, MAX_LEN, data_type='test') if args.do_test else None

    training_loader = DataLoader(training_set, **train_params)
    validation_loader = DataLoader(validation_set, **valid_params) if validation_set is not None else None
    testing_loader = DataLoader(testing_set, **test_params) if testing_set is not None else None

    args.train_classifier = True

    model = model.to(accelerator.device)
    data_list = [training_loader, validation_loader, testing_loader]

    if 'generation' in args.load.lower():
        logger.info("Generation task......")
        run_gen(args, model, tokenizer, optimizer, data_list)
    else:
        logger.info("Classification task......")
        run(args, model, tokenizer, optimizer, data_list)


def validation_multigpu_generation(args, model, loader, tokenizer=None):
    model.eval()
    loader = args.accelerator.prepare(loader)
    args.accelerator.print(
        f"The size of valid dataloader: {len(loader)}, {len(loader.dataset)}. (Batch size: {args.batch_size}, Word size: {torch.cuda.device_count()})")
    device = args.accelerator.device
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        index = 0
        eos_token_id = tokenizer.convert_tokens_to_ids("<|EOT|>")
        for data in tqdm(loader, total=len(loader), desc="Eval: "):
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = ids.ne(tokenizer.pad_token_id)
            targets = data['target_ids'].to(device, dtype=torch.long)
            targets_mask = targets.ne(tokenizer.pad_token_id)

            outputs = model.generate(ids, mask, max_length=args.max_source_len, pad_token_id=tokenizer.pad_token_id,
                                     eos_token_id=eos_token_id)
            all_targets.extend(args.accelerator.gather_for_metrics(targets.cpu().detach().numpy().tolist()))
            all_predictions.extend(args.accelerator.gather_for_metrics(outputs.cpu().detach().numpy().tolist()))
    all_predictions = all_predictions[:len(loader.dataset)]
    all_predictions = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in
                       all_predictions]
    all_targets = all_targets[:len(loader.dataset)]
    all_targets = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in
                   all_targets]
    return all_predictions, all_targets


def run_gen(args, model, tokenizer, optimizer, data_list):
    t0 = time.time()
    if args.accelerator.is_main_process:
        f_summary = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    training_loader, validation_loader, testing_loader = data_list
    device = args.accelerator.device
    args.do_train = True
    if args.do_train:
        best_accf1 = 0
        # 2026-05-24 patch (claude-code): respect args.start_epoch (see run()).
        for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
            model.train()
            bar = tqdm(training_loader, total=len(training_loader), desc="Training")
            for idx, data in enumerate(bar):
                ids = data['input_ids'].to(device, dtype=torch.long)
                mask = ids.ne(tokenizer.pad_token_id)
                targets = data['labels'].to(device, dtype=torch.long)
                targets_mask = targets.ne(tokenizer.pad_token_id)
                outputs, loss = model(text1=ids, mask1=mask, target_ids=targets)

                if args.grad_acc_steps > 1:
                    loss = loss / args.grad_acc_steps
                if (idx + 1) % args.grad_acc_steps == 0:
                    optimizer.zero_grad()
                    args.accelerator.backward(loss)
                    optimizer.step()
                    if args.accelerator.is_main_process:
                        bar.set_description(
                            "[{}] Step: {}/{} Train loss {}".format(epoch + 1, idx + 1, len(training_loader),
                                                                    loss * args.grad_acc_steps))
                else:
                    with args.accelerator.no_sync(model):
                        args.accelerator.backward(loss)

            # args.do_eval = True
            if args.do_eval:
                valid_outs, valid_tgts = validation_multigpu_generation(args, model, validation_loader, tokenizer)
                print("Length of valid_outs: ", len(valid_outs))
                print("Length of valid_tgts: ", len(valid_tgts))
                if args.accelerator.is_main_process:
                    accuracy = sum([1 if p == g else 0 for p, g in zip(valid_outs, valid_tgts)]) / len(valid_outs)
                    prediction_folder = os.path.join(args.output_dir, 'prediction')
                    os.makedirs(prediction_folder, exist_ok=True)
                    with open(os.path.join(prediction_folder, f'valid_res_{epoch}.json'), 'w') as f:
                        for p, g in zip(valid_outs, valid_tgts):
                            f.write(json.dumps({"pred": p, "gold": g}) + '\n')
                    accf1 = accuracy
                    logger.info(f'Accuracy:  {accuracy}')
                    if accf1 >= best_accf1:
                        not_inc_cnt = 0
                        f_summary.write("[%d] Best acc changed into %.2f\n" % (epoch, accuracy))
                        save_path = os.path.join(args.output_dir, 'checkpoint-best')
                        os.makedirs(save_path, exist_ok=True)
                        best_accf1 = accuracy
                        model_to_save = model.module if hasattr(model, 'module') else model
                        # model_to_save = args.accelerator.unwrap_model(model_to_save)
                        torch.save(model_to_save.state_dict(), os.path.join(save_path, 'model.bin'))
                        logger.info("Save Model!")
                    else:
                        not_inc_cnt += 1
                        f_summary.write(
                            f"[{epoch}] Best acc+f1 ({best_accf1}) does not drop changed for {not_inc_cnt} epochs, cur acc: {accuracy}")

                        # save
                        save_path = os.path.join(args.output_dir, 'checkpoins')
                        os.makedirs(save_path, exist_ok=True)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save(model_to_save.state_dict(), os.path.join(save_path, f'model_{epoch}.bin'))

                        if not_inc_cnt >= 3:
                            stop_early_str = "[%d] Early stop as not_inc_cnt=%d\n" % (epoch, not_inc_cnt)
                            logger.info(stop_early_str)
                            args.accelerator.set_trigger()
                if args.accelerator.check_trigger():
                    break

    print("Start testing.......")
    # args.do_test = True
    if args.do_test:
        save_path = os.path.join(args.output_dir, 'checkpoint-best')
        prediction_folder = os.path.join(args.output_dir, 'prediction')
        model = model.module if hasattr(model, 'module') else model
        model.load_state_dict(torch.load(os.path.join(save_path, 'model.bin'), map_location='cpu'))
        outputs, targets = validation_multigpu_generation(args, model, testing_loader, tokenizer)
        print("Length of outputs: ", len(outputs))
        print("Length of targets: ", len(targets))
        if args.accelerator.is_main_process:
            accuracy = sum([1 if p == g else 0 for p, g in zip(valid_outs, valid_tgts)]) / len(valid_outs)
            logger.info(f"Accuracy Score = {accuracy}")
            with open(os.path.join(prediction_folder, 'test_res.json'), 'w') as f:
                for p, g in zip(outputs, targets):
                    f.write(json.dumps({"pred": p, "gold": g}) + '\n')
            logger.info("Finish and take {}".format(get_elapse_time(t0)))
            f_summary.write(f"Testing results: Accuracy: {accuracy}")
            f_summary.write("Finish and take {}".format(get_elapse_time(t0)))
            f_summary.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='Mutation', type=str)
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--cache_path', default='', type=str)
    parser.add_argument('--output_dir', default='./saved_models', type=str)

    parser.add_argument('--load', default='deepseek-coder-1.3b-base', type=str)
    parser.add_argument('--max_source_len', default=600, type=int)
    parser.add_argument('--max_target_len', default=-1, type=int)
    parser.add_argument('--batch_size', default=60, type=int)
    parser.add_argument('--grad_acc_steps', default=1, type=int)
    parser.add_argument('--lr', default=5e-6, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--do_eval', default
    =False, type=int)
    parser.add_argument('--do_test', default=False, type=float)
    # 2026-05-24 patch (claude-code): added resume-from-checkpoint support.
    # --resume_from_checkpoint is an absolute path to a model_*.bin emitted by
    # an earlier run (e.g. checkpoinss/0/model_0.bin). --start_epoch sets the
    # epoch counter so the resumed run writes checkpoinss/<start_epoch+i>/...
    # and labels the per-epoch valid predictions correctly.
    parser.add_argument('--resume_from_checkpoint', default='', type=str,
                        help='Path to a model_*.bin produced by a previous run.')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='Epoch number to begin from (0-indexed); used for '
                             'output paths so resumed runs do not clobber prior epochs.')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # 2026-05-24 patch (claude-code): upstream hardcoded args.cache_path = ''
    # which then fails os.makedirs(''). Use the parsed value; default to
    # <output_dir>/cache if empty.
    if not args.cache_path:
        args.cache_path = os.path.join(args.output_dir, 'cache')
    os.makedirs(args.cache_path, exist_ok=True)
    argsdict = vars(args)
    with open(os.path.join(args.output_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    set_seed(42)
    main(args)