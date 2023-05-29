import os
import argparse
from solver import Solver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=30)

    parser.add_argument("--ckpt_dir", type=str, default="./checkpoint")
    parser.add_argument("--ckpt_name", type=str, default="landscape")
    parser.add_argument("--print_every", type=int, default=1)

    # if you change image size, you must change all the network channels
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--data_root", type=str, default="./dataset")

    args = parser.parse_args()
    solver = Solver(args)
    solver.fit()


if __name__ == "__main__":
    main()
