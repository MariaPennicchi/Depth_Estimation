import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import DepthDataset
from utils import visualize_img, ssim
#from utils import plot_loss
import torchvision.transforms as transforms
from model import Net
from torch.optim.lr_scheduler import CyclicLR


class Solver():

    def __init__(self, args):
        self.args = args
        '''
        self.train_rmse = []
        self.train_ssim = []
        self.val_rmse = []
        self.val_ssim = []
        '''

        if self.args.is_train:

            self.train_data = DepthDataset(train=DepthDataset.TRAIN,
                                           data_dir=args.data_dir,
                                           transform=transforms.Resize((144, 256))
                                           )
            self.val_data = DepthDataset(train=DepthDataset.VAL,
                                         data_dir=args.data_dir,
                                         transform=transforms.Resize((144, 256))
                                         )

            self.train_loader = DataLoader(dataset=self.train_data,
                                           batch_size=args.batch_size,
                                           num_workers=4,
                                           shuffle=True,
                                           drop_last=True)

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.net = Net().to(self.device)
            self.criterion1 = torch.nn.L1Loss()
            self.criterion2 = torch.nn.MSELoss()
            self.optim = torch.optim.Adam(self.net.parameters(), lr=args.lr)
            self.scheduler = CyclicLR(self.optim, base_lr=0.000001, max_lr=0.01, step_size_up=2000,
                                      step_size_down=2000, mode="triangular2")
            self.args = args

            if not os.path.exists(args.ckpt_dir):
                os.makedirs(args.ckpt_dir)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.net = Net().to(self.device)
            '''
            self.test_set = DepthDataset(train=DepthDataset.VAL,
                                         data_dir=self.args.data_dir,
                                         transform=transforms.Resize((144, 256))
                                         )
            '''
            self.test_set = DepthDataset(train=DepthDataset.TEST,
                                         data_dir=self.args.data_dir,
                                         transform=transforms.Resize((144, 256))
                                         )
            ckpt_file = os.path.join("checkpoint", self.args.ckpt_file)
            self.net.load_state_dict(torch.load(ckpt_file, weights_only=True))
            #self.net.load_state_dict(torch.load(ckpt_file, weights_only=True, map_location=torch.device('cpu')))

    def fit(self):
        args = self.args

        for epoch in range(args.max_epochs):
            self.net.train()
            for step, inputs in enumerate(self.train_loader):
                rgb = inputs[0].to(self.device)
                depth = inputs[1].to(self.device)
                pred = self.net(rgb)
                loss = self.criterion1(pred, depth) + 2*torch.sqrt(self.criterion2(pred, depth))

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.scheduler.step()

                if (step + 1) % 100 == 0:
                    print("Epoch [{}/{}] Loss: {:.3f} ".format(epoch + 1, args.max_epochs, loss.item()))

            if (epoch + 1) % args.evaluate_every == 0:
                train_acc = self.evaluate(DepthDataset.TRAIN)
                val_acc = self.evaluate(DepthDataset.VAL)
                self.save(args.ckpt_dir, args.ckpt_name, epoch + 1)

        return

    def evaluate(self, set):

        args = self.args
        if set == DepthDataset.TRAIN:
            dataset = self.train_data
            suffix = "TRAIN"
        elif set == DepthDataset.VAL:
            dataset = self.val_data
            suffix = "VALIDATION"
        else:
            raise ValueError("Invalid set value")

        loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=4,
                            shuffle=False, drop_last=False)

        self.net.eval()
        ssim_acc = 0.0
        rmse_acc = 0.0
        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):
                output = self.net(images.to(self.device))
                ssim_acc += ssim(output, depth.to(self.device)).item()
                rmse_acc += torch.sqrt(F.mse_loss(output, depth.to(self.device))).item()

                if i % self.args.visualize_every == 0:
                    visualize_img(images[0].cpu(),
                                  depth[0].cpu(),
                                  output[0].cpu().detach(),
                                  suffix=suffix)
        print("RMSE on", suffix, ":", rmse_acc / len(loader))
        print("SSIM on", suffix, ":", ssim_acc / len(loader))
        '''
        if set == DepthDataset.TRAIN:
            self.train_rmse.append(rmse_acc / len(loader))
            self.train_ssim.append(ssim_acc / len(loader))
            plot_loss(self.train_rmse, self.train_ssim, args.save_loss_dir, train=True)
        if set == DepthDataset.VAL:
            self.val_rmse.append(rmse_acc / len(loader))
            self.val_ssim.append(ssim_acc / len(loader))
            plot_loss(self.val_rmse, self.val_ssim, args.save_loss_dir, train=False)
        '''

    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)

    def test(self):

        loader = DataLoader(self.test_set,
                            batch_size=self.args.batch_size,
                            num_workers=4,
                            shuffle=False, drop_last=False)
        self.net.eval()
        ssim_acc = 0.0
        rmse_acc = 0.0
        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):
                output = self.net(images.to(self.device))
                ssim_acc += ssim(output, depth.to(self.device)).item()
                rmse_acc += torch.sqrt(F.mse_loss(output, depth.to(self.device))).item()

                if i % self.args.visualize_every == 0:
                    visualize_img(images[0].cpu(),
                                  depth[0].cpu(),
                                  output[0].cpu().detach(),
                                  suffix="TEST")

        print("RMSE on TEST :", rmse_acc / len(loader))
        print("SSIM on TEST:", ssim_acc / len(loader))
