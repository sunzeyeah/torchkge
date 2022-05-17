import os
import argparse
import logging
import torch

from torch.optim import Adam
from torchkge.evaluation import LinkPredictionEvaluator
from torchkge.models import TransEModel
from torchkge.utils.datasets import load_ccks, load_fb15k
from torchkge.utils import Trainer, MarginLoss

logging.basicConfig(
    format="%(asctime)s %(levelname)-4s [%(filename)s:%(lineno)s]  %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", required=True, type=str, help="模型训练数据地址")
    parser.add_argument("--output_dir", required=True, type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--model_name", default="transE", type=str, help="Bert pre-trained model selected in the list: bert-base-uncased, "
                                                                         "bert-large-uncased, roberta-base",)
    # training
    parser.add_argument("--do_eval", action="store_true", help="是否进行模型验证")
    parser.add_argument("--cuda_mode", default="all", help="cuda mode, all or batch")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    # parser.add_argument("--eval_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=1.0, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=1e-3, type=float, help="weight decay")
    # parser.add_argument("--threads", default=8, type=int, help="Number of workers in the dataloader.")
    # optimization
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit float precision instead of 32-bit")
    # Graph Embedding
    parser.add_argument("--dim", default=64, type=int, help="dimension of graph embedding")
    parser.add_argument("--margin", default=5.0, type=float, help="maring loss")
    parser.add_argument("--n_neg", default=1, type=int, help="number of negative samples")
    # parser.add_argument("--negative_entities", default=3, type=int, help="number of negative entities")
    # parser.add_argument("--negative_relations", default=3, type=int, help="number of negative relations")
    parser.add_argument("--norm", default="L2", type=str, help="vector norm: L1, L2, torus_L1, torus_L2")
    parser.add_argument("--sampling_type", default="bern", type=str, help="sampling type, Either 'unif' (uniform negative sampling) or "
                                                                               "'bern' (Bernoulli negative sampling)")

    return parser.parse_args()


def main():
    args = get_parser()

    # Load dataset
    kg_train, kg_val, kg_test = load_ccks(args.data_dir)
    # kg_train, kg_val, kg_test = load_fb15k()
    logger.info(f"finished loading data")

    # Define the model and criterion
    model = TransEModel(args.dim, kg_train.n_ent, kg_train.n_rel,
                        dissimilarity_type=args.norm)
    criterion = MarginLoss(args.margin)
    optimizer = Adam(model.parameters(), lr=args.learning_rate,
                     weight_decay=args.weight_decay, eps=args.adam_epsilon)

    # Start Training
    use_cuda = args.cuda_mode if torch.cuda.is_available() else None
    model_save_path = os.path.join(args.output_dir, args.model_name)
    trainer = Trainer(model, criterion, kg_train, args.num_train_epochs,
                      args.train_batch_size, optimizer=optimizer,
                      model_save_path=model_save_path, sampling_type=args.sampling_type,
                      n_neg=args.n_neg, use_cuda=use_cuda)
    trainer.run()

    # Evaluation
    if args.do_eval:
        evaluator = LinkPredictionEvaluator(model, kg_test)
        evaluator.evaluate(200)
        evaluator.print_results()


if __name__ == "__main__":
    main()
