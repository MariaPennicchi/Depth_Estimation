import os
import argparse
from solver import Solver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=150)
    
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoint")
    parser.add_argument("--ckpt_name", type=str, default="depth")
    parser.add_argument("--evaluate_every", type=int, default=4)
    parser.add_argument("--visualize_every", type=int, default=100)
    parser.add_argument("--data_dir", type=str, default=os.path.join("C:\\", "Users", "Utente", "DATASET", "DepthEstimationUnreal"))
    #parser.add_argument("--data_dir", type=str, default=os.path.join("C:\\", "Users", "maria", "Desktop", "Uni", "deep_learning_and_robot_perceptron", "tesina", "DATASET", "DepthEstimationUnreal"))
    #parser.add_argument("--data_dir", type=str,default=os.path.join('/content/drive/MyDrive/CHALLENGE/DATASET/DepthEstimationUnreal'))
    #parser.add_argument("--save_loss_dir", type=str,default=os.path.join('/content/drive/MyDrive/CHALLENGE/loss'))

    parser.add_argument("--is_train", type=bool, default=False)
    parser.add_argument("--ckpt_file", type=str, default="depth_88.pth")

    args = parser.parse_args()
    solver = Solver(args)
    if args.is_train:
        solver.fit()
    else:
        solver.test()

if __name__ == "__main__":
    main()
