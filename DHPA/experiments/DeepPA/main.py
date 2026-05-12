import torch
import numpy as np
import os
import time
import argparse
import yaml
import pickle
import scipy.sparse as sp
from scipy.sparse import linalg
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch.nn as nn
import torch

from src.utils.helper import get_dataloader, check_device, get_num_nodes
from src.models.DeepPA import DeepPA
from src.trainers.deeppa_trainer import DeepPA_Trainer
from src.utils.graph_algo import load_graph_data
from src.utils.args import get_public_config, str_to_bool


def get_config():
    parser = get_public_config()

    # get private config
    parser.add_argument("--model_name", type=str, default="DeepPA", help="which model to train")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--filter_type", type=str, default="transition")
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--n_hidden", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--spatial_flag", type=str_to_bool, default=True, help="whether to use spatial transformer")
    parser.add_argument("--temporal_flag", type=str_to_bool, default=True, help="whether to use temporal transformer")
    parser.add_argument("--spatial_encoding", type=str_to_bool, default=True, help="whether to use spatial encoding")
    parser.add_argument("--temporal_encoding", type=str_to_bool, default=True, help="whether to use temporal encoding")
    parser.add_argument("--temporal_PE", type=str_to_bool, default=True, help="whether to use temporal PE")
    parser.add_argument("--GCO", type=str_to_bool, default=True, help="whether to use GCO")
    parser.add_argument("--CLUSTER", type=str_to_bool, default=False, help="whether to use CLUSTER")
    parser.add_argument("--GCO_Thre", type=float, default=1, help="The proportion of low frequency signals")
    parser.add_argument("--base_lr", type=float, default=1e-3)
    parser.add_argument("--lr_decay_ratio", type=float, default=0.5)

    # ----------------------------- #
    # // NEW: T-S-R 分解相关超参（默认关闭，完全回退）
    parser.add_argument("--ts_decompose", type=str_to_bool, default=False,
                        help="enable trend-seasonal-residual fusion head")
    parser.add_argument("--decompose_kernel", type=int, default=7,
                        help="temporal kernel (odd) for trend smoothing")
    parser.add_argument("--season_periods", type=str, default="12,24,48",
                        help="comma-separated periods (in 15-min steps), e.g. '12,24,48'")
    # ----------------------------- #

    args = parser.parse_args()

    # // NEW: 解析 season_periods 字符串为整数列表
    # 允许空白与多余逗号，保持健壮性
    sp_str = getattr(args, "season_periods", "")
    if isinstance(sp_str, str):
        arr = [s.strip() for s in sp_str.split(",") if s.strip() != ""]
        try:
            args.season_periods = tuple(int(x) for x in arr) if len(arr) > 0 else (12, 24, 48)
        except ValueError:
            args.season_periods = (12, 24, 48)
    elif isinstance(sp_str, (list, tuple)):
        args.season_periods = tuple(int(x) for x in sp_str)
    else:
        args.season_periods = (12, 24, 48)
    # // NEW end

    if getattr(args, "lr_step", 0) and args.lr_step > 0:
        # 从 args.lr_step 开始，每隔 lr_step 一个里程碑，直到 max_epochs 之前
        args.steps = list(range(args.lr_step, args.max_epochs, args.lr_step))
    else:
        args.steps = [10, 20, 30, 40]  # 兜底，保持原逻辑

    print(args)

    folder_name = "{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(
        args.n_hidden,
        args.n_blocks,
        args.n_heads,
        args.spatial_flag,
        args.temporal_flag,
        args.spatial_encoding,
        args.temporal_encoding,
        args.temporal_PE,
        args.aug,
        args.batch_size,
        args.base_lr,
        args.n_exp,
        args.GCO,
        args.temporal_encoding,
        args.GCO_Thre,
        str(args.ts_decompose),  # // NEW: 将开关写入日志目录方便对齐
    )
    args.log_dir = "./logs/{}/{}/{}/".format(args.dataset, args.model_name, folder_name)
    print(args.log_dir)
    args.num_nodes = get_num_nodes(args.dataset)

    args.datapath = os.path.join("./data", args.dataset)
    if args.seed != 0:
        torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    return args, folder_name


def main():
    args, fname = get_config()

    device = check_device()

    model = DeepPA(
        dropout=args.dropout,
        spatial_flag=args.spatial_flag,
        temporal_flag=args.temporal_flag,
        spatial_encoding=args.spatial_encoding,
        temporal_encoding=args.temporal_encoding,
        temporal_PE=args.temporal_PE,
        GCO=args.GCO,
        CLUSTER=args.CLUSTER,
        n_hidden=args.n_hidden,
        end_channels=args.n_hidden * 8,
        n_blocks=args.n_blocks,
        # 新增这一行，把命令行参数传入模型
        temporal_causal=getattr(args, "temporal_causal", True),
        spatial_op=getattr(args, "spatial_op", "gco"),
        spatial_heads=getattr(args, "spatial_heads", 2),
        afno_keep_ratio=getattr(args, "afno_keep_ratio", 0.5),
        # ----------------------------- #
        # // NEW: 传递分解相关超参（默认关闭）
        ts_decompose=args.ts_decompose,
        decompose_kernel=args.decompose_kernel,
        season_periods=args.season_periods,
        # ----------------------------- #
        name=args.model_name,
        dataset=args.dataset,
        device=device,
        num_nodes=args.num_nodes,
        seq_len=args.seq_len,
        horizon=args.horizon,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        GCO_Thre=args.GCO_Thre,
    )
    print("Model created.")

    print("Loading dataloader. This may take a while...")
    data = get_dataloader(args.datapath, args.batch_size, args.output_dim)

    trainer = DeepPA_Trainer(
        model=model,
        adj_mat=None,
        filter_type=args.filter_type,
        data=data,
        aug=args.aug,
        base_lr=args.base_lr,
        steps=args.steps,
        lr_decay_ratio=args.lr_decay_ratio,
        log_dir=args.log_dir,
        n_exp=args.n_exp,
        wandb_flag=args.wandb,
        save_iter=args.save_iter,
        clip_grad_value=args.max_grad_norm,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=device,
    )
    print("trainer..")
    if args.mode == "train":
        print("began training..")
        trainer.train()
        trainer.test(-1, "test")

    else:
        trainer.test(-1, args.mode)
        if args.save_preds:
            trainer.save_preds(-1)


if __name__ == "__main__":
    torch.set_num_threads(8)
    main()
