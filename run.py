import argparse
import random
import numpy as np
import torch


def argparser():
    ap = argparse.ArgumentParser("Train a model for sentence classification.")

    ## Required parameters
    ap.add_argument(
        "--data",
        required=True,
        help="Input TF example files (can be a glob or comma separated).",
    )

    ap.add_argument(
        "--output_dir",
        required=True,
        help="The output directory where the model checkpoints will be written.",
    )

    ## Other parameters
    ap.add_argument(
        "--init_checkpoint",
        default=None,
        help="Initial checkpoint (usually from a pre-trained BERT model).",
    )

    ap.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="The maximum total input sequence length after WordPiece tokenization. "
        "Sequences longer than this will be truncated, and sequences shorter "
        "than this will be padded. Must match data generation.",
    )

    ap.add_argument(
        "--max_predictions_per_seq",
        type=int,
        default=20,
        help="Maximum number of masked LM predictions per sequence. "
        "Must match data generation.",
    )

    ap.add_argument("--do_train", action="store_true", help="Whether to run training.")

    ap.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )

    ap.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Total batch size for training.",
    )

    ap.add_argument(
        "--eval_batch_size", type=int, default=8, help="Total batch size for eval."
    )

    ap.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="The initial learning rate for Adam.",
    )

    ap.add_argument(
        "--num_train_steps", type=int, default=100000, help="Number of training steps."
    )

    ap.add_argument(
        "--num_warmup_steps", type=int, default=10000, help="Number of warmup steps."
    )

    ap.add_argument(
        "--save_checkpoints_steps",
        type=int,
        default=1000,
        help="How often to save the model checkpoint.",
    )

    ap.add_argument(
        "--iterations_per_loop",
        type=int,
        default=1000,
        help="How many steps to make in each estimator call.",
    )

    ap.add_argument(
        "--max_eval_steps",
        type=int,
        default=100,
        help="Maximum number of eval steps.",
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=2023,
        help="Random seed.",
    )

    return ap


def set_seed(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    ap = argparser()
    args = ap.parse_args()

    set_seed(args.seed)
