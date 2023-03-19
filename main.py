import torch
import random
import argparse
from torch import nn
from tqdm import tqdm
from fcmeans import FCM
from os import path, makedirs
from sklearn import preprocessing
import torchvision.datasets as tds
import torchvision.utils as vutils
from sklearn.cluster import KMeans as KM
import torchvision.transforms as transforms
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI

from utils import *
from GAN_MNIST import *
from Kappa_calculate import Kappa



parser = argparse.ArgumentParser('arguments for training')
parser.add_argument('--data_root', type = str, help = 'path to dataset directory')
parser.add_argument('--exp_dir', type = str, help = 'path to experiment directory')
parser.add_argument('--p', type = float, default = 0.25, help = 'non-IID level')
parser.add_argument('--class_num', type = int, default = 10, help = 'number of classes')
parser.add_argument('--z_dim', type = int, default = 62, help = 'dimension of the noise z')
parser.add_argument('--epochs', type = int, default = 50, help = 'number of training epochs')
parser.add_argument('--lr_G', type = float, default = 0.0006, help = 'learning rate')
parser.add_argument('--lr_D', type = float, default = 0.0002, help = 'learning rate')
args = parser.parse_args()



def train_GANs(args, images, ground_truth, device):
    
    print(f"Training on: {device}")
    
    setup_seed(20)
    Nets = locals()
    construct_clients(Nets, args.class_num, args.p, images, ground_truth)
    G_input_fixed, _ = gen_cond_label(64, args.class_num, args.z_dim)
    G_input_fixed = G_input_fixed.to(device)
    for m in range(args.class_num):
        torch_dataset = torch.utils.data.TensorDataset(Nets[f'X_{m}'], Nets[f'y_{m}'])
        
        print(f'The size of client {m}:', sum(torch.bincount(Nets[f'y_{m}'])))
        print(f'The label distribution of client {m}:', torch.bincount(Nets[f'y_{m}']))

        Nets[f'G_{m}'] = Generator(args.class_num, args.z_dim)
        Nets[f'D_{m}'] = Discriminator()
        Nets[f'G_{m}'].apply(init_weights)
        Nets[f'D_{m}'].apply(init_weights)
        Nets[f'G_{m}'].to(device)
        Nets[f'D_{m}'].to(device)

        Nets[f'G_{m}'].train()
        Nets[f'D_{m}'].train()

        with tqdm(total = args.epochs) as progress_bar:
            for epoch in range(args.epochs):
                decay = 0.99 ** epoch
                Nets[f'optimizer_G_{m}'] = torch.optim.Adam(Nets[f'G_{m}'].parameters(), 
                                                            lr = decay * 3 * args.lr_G, betas = (0.5, 0.999))
                Nets[f'optimizer_D_{m}'] = torch.optim.Adam(Nets[f'D_{m}'].parameters(), 
                                                            lr = decay * 1 * args.lr_D, betas = (0.5, 0.999))

                train_iter = torch.utils.data.DataLoader(dataset = torch_dataset, batch_size = 64,
                                                         shuffle = True, num_workers = 4)
                for i, (X, _) in enumerate(train_iter):
                    #train D first
                    Nets[f'optimizer_D_{m}'].zero_grad()
                    X = X.to(device)
                    X_size = X.shape[0]
                    #real_images_loss 
                    D_real = Nets[f'D_{m}'](X)
                    Y_real = torch.ones((X_size, 1), device = device)
                    D_real_loss = torch.nn.BCEWithLogitsLoss()(D_real, Y_real)

                    #fake_images_loss
                    G_input, _ = gen_cond_label(X_size, args.class_num, args.z_dim)
                    G_input = G_input.to(device)
                    X_fake = Nets[f'G_{m}'](G_input).detach()
                    D_fake = Nets[f'D_{m}'](X_fake)
                    Y_fake = torch.zeros((X_size, 1), device = device)
                    D_fake_loss = torch.nn.BCEWithLogitsLoss()(D_fake, Y_fake)

                    #total loss
                    D_loss = D_real_loss + D_fake_loss
                    D_loss.backward()
                    Nets[f'optimizer_D_{m}'].step()


                    #train G
                    Nets[f'optimizer_G_{m}'].zero_grad()

                    G_input, cond_label = gen_cond_label(X_size, args.class_num, args.z_dim)
                    G_input = G_input.to(device)
                    X_fake = Nets[f'G_{m}'](G_input)
                    D_fake = Nets[f'D_{m}'](X_fake)
                    Y_fake = torch.ones((X_size, 1), device = device)

                    G_loss = torch.nn.BCEWithLogitsLoss()(D_fake, Y_fake)
                    G_loss.backward()
                    Nets[f'optimizer_G_{m}'].step()

                    #save images
                    if i % 100 == 0:
                        vutils.save_image(X.cpu(), f'output/mnist/{m}/real_samples.png', normalize=True)
                        X_fake_fixed = Nets[f'G_{m}'](G_input_fixed).detach()
                        vutils.save_image(X_fake_fixed, args.exp_dir + f'/{m}/fake_samples_m_{m}_epoch_{epoch}.png', normalize = True)
                
                progress_bar.set_description(f'Client_{m}:')
                progress_bar.update(1)

    #generate 
    fake_num = 200
    iterations = 35
    images_fake = torch.zeros(0, device = device)  
    for m in range(args.class_num):
        for _ in range(iterations):       
            Nets[f'G_{m}'].eval()
            G_input, cond_label = gen_cond_label(fake_num, args.class_num, args.z_dim)
            pseudo_label = torch.argmax(cond_label, 1)
            G_input, _ = G_input.to(device), pseudo_label.to(device)
            X_fake = Nets[f'G_{m}'](G_input).detach()

            images_fake = torch.cat([images_fake, X_fake], 0)
    
    return images_fake.cpu()


def main():
    #check directories
    if not path.exists(args.data_root):
        makedirs(args.data_root)
    for m in range(args.class_num):
        output_dir = args.exp_dir + f'/{m}'
        if not path.exists(output_dir):
            makedirs(output_dir) 
        
    #load MNIST
    train_dataset = tds.MNIST(root = args.data_root, train=True, transform = transforms.ToTensor(), download=True)
    test_dataset = tds.MNIST(root = args.data_root, train=False, transform = transforms.ToTensor(), download=True)
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    
    data_loader = torch.utils.data.DataLoader(combined_dataset, batch_size = 1000, shuffle = False, num_workers = 0)
    real_images = []
    real_labels = []
    for X, y in data_loader:
        real_images.append(X)
        real_labels.append(y)
        
    real_images = torch.cat(real_images, dim = 0)
    real_labels = torch.cat(real_labels, dim = 0)
    

    print('1. Global synthetic data construction: ...\t')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_fake = train_GANs(args, real_images, real_labels, device)
    torch.save(X_fake, args.exp_dir + '/X_fake')

    
    
    print('2. Cluster assignment: ...\t')
    real_images_flatten = real_images.view(-1, 28*28)
    fake_images_flatten = X_fake.view(-1, 28*28)
    X_normed = preprocessing.normalize(real_images_flatten)
    X_fake_normed = preprocessing.normalize(fake_images_flatten)
    
    print('1)K-means: ...\t')
    setup_seed(20)
    kmeans = KM(n_clusters = args.class_num).fit(X_fake_normed)
    centers_fake_normed_km = preprocessing.normalize(kmeans.cluster_centers_)
    pred_km = torch.mm(torch.from_numpy(X_normed), torch.from_numpy(centers_fake_normed_km).T).argmax(1)
    
    print('2)F-cmeans: ...\t')
    fcm = FCM(n_clusters = args.class_num, m = 1.1, random_state = 20)
    fcm.fit(X_fake_normed)
    centers_fake_normed_fcm = preprocessing.normalize(fcm.centers)
    pred_fcm = torch.mm(torch.from_numpy(X_normed), torch.from_numpy(centers_fake_normed_fcm).T).argmax(1)
    
    torch.save(pred_km, args.exp_dir + '/pred_km')
    torch.save(pred_fcm, args.exp_dir + '/pred_fcm')
    print(f'p = {args.p}:')
    print(f'K-means: NMI = {NMI(real_labels, pred_km)}, Kappa = {Kappa(real_labels, pred_km)}')
    print(f'Fuzzy-cmeans: NMI = {NMI(real_labels, pred_fcm)}, Kappa = {Kappa(real_labels, pred_fcm)}')
    
    
if __name__ == '__main__':
    main()    
    
