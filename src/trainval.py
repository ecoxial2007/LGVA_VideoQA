import os
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data.dataloader import default_collate
from model import LGVAConfig, LGVAModel
from trainval_single_batch import downstream_task_forward
from loss import LabelSmoothingCrossEntropy

dataset_mapping = {
    'nextqa_mc': 'datasets.nextqa',
    'msvd_oe': 'datasets.msvd',
    'msrvtt_oe': 'datasets.msrvtt',
    'msrvtt_mc': 'datasets.msrvtt',
    'tgif_frameqa_oe': 'datasets.tgif',
    'activitynet_oe': 'datasets.activitynet',
}

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def log(message, logger=None):
    """
    Placeholder log function; replace with your loggers/apis of choice (e.g. wandb, etc.)
    """
    if logger is not None: raise NotImplemented("implement your own logger")
    print(message)
    args.log_path = os.path.join("./checkpoints", f"{args.dataset}")
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    with open(os.path.join(args.log_path, 'log.txt'), 'a') as lf:
        lf.write(message+'\n')

def process_batch(batch, set_to_device=None, replace_empty_with_none=False):
    if set_to_device is not None:
        if isinstance(batch, dict):
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(set_to_device)
        elif isinstance(batch, list):
            batch = [_b.to(set_to_device) if torch.is_tensor(_b) else _b for _b in batch]

    if replace_empty_with_none:
        if isinstance(batch, dict):
            for key in batch:
                if len(batch[key]) == 0:
                    batch[key] = None
        elif isinstance(batch, list):
            batch = [_b if len(_b) > 0 else None for _b in batch]

    return batch


def main(args):
    seed_everything(args.seed)

    # create LGVAConfig from model hyperparameters
    config = LGVAConfig.from_args(args)
    device = torch.device("cuda")
    model = LGVAModel(config).to(device)


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total parameters: ", total_params)
    print("Trainable parameters: ", trainable_params)

    if args.dataset in dataset_mapping:
        module_name = dataset_mapping[args.dataset]
        VideoLanguageDataset = getattr(__import__(module_name, fromlist=['VideoLanguageDataset']),
                                       'VideoLanguageDataset')
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    dset_train = VideoLanguageDataset(args, split="train")
    dset_val = VideoLanguageDataset(args, split="val")

    dldr_train = torch.utils.data.DataLoader(dset_train,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             pin_memory=False,
                                             num_workers=args.num_workers,
                                             collate_fn=default_collate)

    dldr_val   = torch.utils.data.DataLoader(dset_val,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=False,
                                             num_workers=args.num_workers,
                                             collate_fn=default_collate)

    # create optimizer
    if args.wd > 0.0:
        optim = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.wd)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = LabelSmoothingCrossEntropy()
    # simple training loop (for illustrative purposes)
    for epoch_i in range(args.epochs):
        # train epoch
        model.train()
        for i, batch in enumerate(dldr_train):
            batch = process_batch(batch, set_to_device=device, replace_empty_with_none=True)

            # refactored the "forward pass" here
            loss, accs = downstream_task_forward(model, batch, criterion, args)
            model.zero_grad(set_to_none=True)
            loss.backward()

            # do logging stuff with accs, loss, etc. For example:
            log(f"train: epoch{epoch_i}, iter{i}: loss = {loss.item()}, acc = {accs.mean().item()}")
            if args.grad_clip_val > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_val)
            optim.step()
        torch.cuda.empty_cache()

        # val epoch
        model.eval()
        all_val_accs = []
        for i, batch in enumerate(dldr_val):
            with torch.no_grad():
                batch = process_batch(batch, set_to_device=device, replace_empty_with_none=True)
                loss, accs = downstream_task_forward(model, batch, criterion, args)
                all_val_accs.append(accs)
        overall_acc = torch.cat(all_val_accs).mean().item()
        log(f"val: epoch{epoch_i}: overall_acc = {overall_acc}")

        checkpoint = {
            "epoch": epoch_i,
            "overall_acc": overall_acc,
            "state_dict": model.state_dict()
        }
        torch.save(checkpoint, os.path.join(args.log_path, f"ckpt_{overall_acc}.pth"))
    return




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parser for LGVA training script.")

    # Training hyperparameters
    parser.add_argument('--batch_size', default=256, type=int, help="batch size")
    parser.add_argument('--lr', default=1e-4, type=float, help="learning rate")
    parser.add_argument('--wd', default=1e-2, type=float, help="weight decay")
    parser.add_argument('--epochs', default=10, type=int, help="number of training epochs")
    parser.add_argument('--grad_clip_val', default=1.0, type=float, help="gradient clip, must be set > 0 to enable")
    parser.add_argument('--gpus', default=1, type=int)  # NOTE: current script is set-up for single-gpu training only.
    parser.add_argument('--num_workers', default=1, type=int, help="number of dataset workers")
    parser.add_argument('--seed', default=3407, type=int, help="random seed")

    # Efficient model hyperparameters (for more help/details, see LGVAConfig)
    parser.add_argument('--d_input', default=768, type=int, help="see LGVAConfig")
    parser.add_argument('--n_ca_heads', default=12, type=int, help="see LGVAConfig")
    parser.add_argument('--n_et_heads', default=5, type=int, help="see LGVAConfig")
    parser.add_argument('--ca_dropout', default=0.1, type=float, help="see LGVAConfig")
    parser.add_argument('--et_dropout', default=0.3, type=float, help="see LGVAConfig")
    parser.add_argument('--n_et_layers', default=2, type=int, help="see LGVAConfig")
    parser.add_argument('--n_gnn_layers', default=2, type=int, help="see LGVAConfig")


    # I/O and tools parameters
    parser.add_argument('--data_path', type=str, help='Annotation', default='./data/Annotation')
    parser.add_argument('--feature_path', type=str, help='Feature')

    parser.add_argument('--split', type=str)
    parser.add_argument('--dataset', type=str,
                        choices=['msvd_oe', 'nextqa_mc', 'msrvtt_mc', 'msrvtt_oe',
                                 'tgif_action_mc', 'tgif_transition_mc', 'tgif_frameqa_oe', 'activitynet_oe'])
    parser.add_argument('--n_frames', default=16, type=int, help="number of frames sampled for input; see tools.py")
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--visible', action='store_true') #for check each question predicts answer
    args = parser.parse_args()

    main(args)
