from networks import VQVAE_Simp, Predictor, simple_CNN
from data import DmgData
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import matplotlib.pyplot as plt


def train_vqvae(opt='SDEG'):
    data_path = r'F:\DATASET\LVI_SM\npy\dataset2'
    param_dir = r'params\\'
    batch_size = 4
    num_workers = 0
    device = torch.device('cuda:0')

    data_loader = DmgData(data_path, opt, batch_size, num_workers, device)
    train_dl = data_loader.train_dataloader()
    valid_dl = data_loader.val_dataloader()

    embedding_dim = 16
    num_embeddings = 256
    num_epoch = 800

    model = VQVAE_Simp(opt, embedding_dim=embedding_dim, num_embeddings=num_embeddings).to(device)

    if os.path.exists(param_dir + opt + "_%d_%d_coder_last.pth"%(embedding_dim, num_embeddings)):
        model.load_state_dict(torch.load(param_dir + opt + "_%d_%d_coder_last.pth"%(embedding_dim, num_embeddings)))
        loss_rec = np.loadtxt(param_dir + opt + "_%d_%d_recorder.txt"%(embedding_dim, num_embeddings), delimiter=',').tolist()
    else:
        loss_rec = [[0, 0.0003, 100, 100, 100, 100]]
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch = (len(loss_rec) - 1) * 10

    while epoch < num_epoch:
        epoch += 1
        tmp_loss_rec = []
        tmp_loss_mse_rec = []
        for idx, [load, sql_npy] in enumerate(train_dl):
            optimizer.zero_grad()
            x_recon, loss, loss_recons, loss_vq = model.training_step(sql_npy)
            loss.backward()
            optimizer.step()
            tmp_loss_rec.append(loss.item())
            tmp_loss_mse_rec.append(loss_recons)

        if epoch % 10 == 0 or epoch == num_epoch - 1:
            train_loss_mean = np.mean(np.array(tmp_loss_rec))
            train_loss_mse_mean = np.mean(np.array(tmp_loss_mse_rec))
            tmp_loss_rec = []
            tmp_loss_mse_rec = []
            for idx, [load, sql_npy] in enumerate(valid_dl):
                x_recon, loss, loss_recons, loss_vq = model.validation_step(sql_npy)
                tmp_loss_rec.append(loss)
                tmp_loss_mse_rec.append(loss_recons)
                # if epoch%100 == 0:
                #     show(x_recon, sql_npy, load, idx=0)
            valid_loss_mean = np.mean(np.array(tmp_loss_rec))
            valid_loss_rec_mean = np.mean(np.array(tmp_loss_mse_rec))

            loss_rec.append([epoch, lr, train_loss_mean, train_loss_mse_mean, valid_loss_mean, valid_loss_rec_mean])

            if valid_loss_rec_mean < np.min(np.array(loss_rec[:-1])[:, -1]):
                torch.save(model.state_dict(), param_dir + opt + "_%d_%d_coder_best.pth"%(embedding_dim, num_embeddings))
            print("Epoch %d, Train Loss: %.5f, Valid Loss: %.5f, Rec loss: %.5f" %
                  (epoch, train_loss_mean, valid_loss_mean, valid_loss_rec_mean))
            torch.save(model.state_dict(), param_dir + opt + "_%d_%d_coder_last.pth"%(embedding_dim, num_embeddings))
            np.savetxt(param_dir + opt + "_%d_%d_recorder.txt"%(embedding_dim, num_embeddings), np.array(loss_rec), delimiter=',')
            if epoch%200 == 0:
                plt.plot(np.log(np.array(loss_rec)[1:, 2]), label='Train Loss')
                plt.plot(np.log(np.array(loss_rec)[1:, 3]), label='Train Rec Loss')
                plt.plot(np.log(np.array(loss_rec)[1:, 4]), label='Valid Loss')
                plt.plot(np.log(np.array(loss_rec)[1:, 5]), label='Valid Rec Loss')
                plt.title(opt + ': Dim %d_ Num %d'%(embedding_dim, num_embeddings))
                plt.legend()
                plt.savefig(param_dir + opt + '_Dim%d_Num%d'%(embedding_dim, num_embeddings) + '.png')
                plt.show()
                show(x_recon, sql_npy, load, idx=0)


def train_predictor(opt='SDEG'):
    data_path = r'F:\DATASET\LVI_SM\npy\dataset2'
    param_dir = r'params\\'
    batch_size = 4
    num_workers = 0
    device = torch.device('cuda:0')

    data_loader = DmgData(data_path, opt, batch_size, num_workers, device)
    train_dl = data_loader.train_dataloader()
    valid_dl = data_loader.val_dataloader()
    embedding_dim = 16
    num_embeddings = 256
    # model = VQVAE(opt, embedding_dim=64, n_codes=512, n_hiddens=384, n_res_layers=8, downsample=(4, 8, 8), device=device).to(device)
    coder_model = VQVAE_Simp(opt, embedding_dim=embedding_dim, num_embeddings=num_embeddings).to(device)
    coder_model.load_state_dict(torch.load(param_dir + opt + "_%d_%d_coder_best.pth"%(embedding_dim, num_embeddings)))
    coder_model.eval()

    predictor = Predictor(embedding_dim=embedding_dim, size=12).to(device)
    if os.path.exists(param_dir + opt + "_predictor_best.pth"):
        predictor.load_state_dict(torch.load(param_dir + opt + "_predictor_best.pth"))
        loss_rec = np.loadtxt(param_dir + opt + "_predictor_recorder.txt", delimiter=',').tolist()
        if len(loss_rec) <= 1:
            loss_rec = [[0, 0.0003, 100, 100]]
    else:
        loss_rec = [[0, 0.0003, 100, 100]]
    num_epoch = 2000
    eval_step = 10
    # loss_rec = [[0, 0.0003, 100, 100]]
    lr = 1e-4
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    epoch = (len(loss_rec) - 1) * 10

    while epoch < num_epoch:
        epoch += 1
        tmp_loss_rec = []
        for idx, [load, sql_npy] in enumerate(train_dl):
            optimizer.zero_grad()
            latent = predictor(load)
            _, latent_label, _, _ = coder_model.encode(sql_npy)
            latent_loss = torch.nn.functional.mse_loss(latent_label, latent)
            pre_sql = coder_model.decode(latent)
            loss = torch.nn.functional.mse_loss(pre_sql, sql_npy) + latent_loss * 0.1
            loss.backward()
            optimizer.step()
            tmp_loss_rec.append(loss.item())

        if epoch % eval_step == 0 or epoch == num_epoch - 1:
            train_loss_mean = np.mean(np.array(tmp_loss_rec))
            tmp_loss_rec = []
            tmp_loss_mse_rec = []
            for idx, [load, sql_npy] in enumerate(valid_dl):
                latent = predictor(load)
                pre_sql = coder_model.decode(latent)
                loss = torch.nn.functional.mse_loss(pre_sql, sql_npy)
                tmp_loss_rec.append(loss.item())
                if epoch%100 == 0:
                    show(pre_sql, sql_npy, load, idx=0, save=True)
            valid_loss_mean = np.mean(np.array(tmp_loss_rec))

            loss_rec.append([epoch, lr, train_loss_mean, valid_loss_mean])

            if valid_loss_mean < np.min(np.array(loss_rec[:-1])[:, 3]):
                torch.save(predictor.state_dict(), param_dir + opt + "_predictor_best.pth")
            print("Epoch %d, Train Loss: %.5f, Valid Loss: %.5f" %
                  (epoch, train_loss_mean, valid_loss_mean))
            torch.save(predictor.state_dict(), param_dir + opt + "_predictor_last.pth")
            np.savetxt(param_dir + opt + '_predictor_recorder.txt', np.array(loss_rec), delimiter=',')


def train_simCNN(opt='SDEG'):
    data_path = r'F:\DATASET\LVI_SM\npy\dataset2'
    param_dir = r'params\\'
    batch_size = 4
    num_workers = 0
    device = torch.device('cuda:0')

    data_loader = DmgData(data_path, opt, batch_size, num_workers, device)
    train_dl = data_loader.train_dataloader()
    valid_dl = data_loader.val_dataloader()

    # model = VQVAE(opt, embedding_dim=64, n_codes=512, n_hiddens=384, n_res_layers=8, downsample=(4, 8, 8), device=device).to(device)
    # coder_model =
    # #
    # coder_model.eval()

    predictor = simple_CNN(opt, embedding_dim=64).to(device)
    if os.path.exists(param_dir + opt + "_Direc_predictor_last.pth"):
        predictor.load_state_dict(torch.load(param_dir + opt + "_Direc_predictor_last.pth"))
        loss_rec = np.loadtxt(param_dir + opt + '_Direc_predictor_recorder.txt', delimiter=',').tolist()
    else:
        loss_rec = [[0, 0.0003, 100, 100]]

    num_epoch = 2000
    eval_step = 10

    lr = 1e-4
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    epoch = (len(loss_rec) - 1) * 10

    while epoch < num_epoch:
        epoch += 1
        tmp_loss_rec = []
        for idx, [load, sql_npy] in enumerate(train_dl):
            optimizer.zero_grad()
            pre_sql = predictor(load)
            loss = torch.nn.functional.mse_loss(pre_sql, sql_npy)
            loss.backward()
            optimizer.step()
            tmp_loss_rec.append(loss.item())

        if epoch % eval_step == 0 or epoch == num_epoch - 1:
            train_loss_mean = np.mean(np.array(tmp_loss_rec))
            tmp_loss_rec = []
            for idx, [load, sql_npy] in enumerate(valid_dl):
                pre_sql = predictor(load)
                loss = torch.nn.functional.mse_loss(pre_sql, sql_npy)
                tmp_loss_rec.append(loss.item())
                if epoch%100 == 0:
                    # show(pre_sql, sql_npy, load, idx=1024)
                    pass
            valid_loss_mean = np.mean(np.array(tmp_loss_rec))

            loss_rec.append([epoch, lr, train_loss_mean, valid_loss_mean])

            if valid_loss_mean < np.min(np.array(loss_rec[:-1])[:, 3]):
                torch.save(predictor.state_dict(), param_dir + opt + "_Direc_predictor_best.pth")
            print("Epoch %d, Train Loss: %.5f, Valid Loss: %.5f" %
                  (epoch, train_loss_mean, valid_loss_mean))
            torch.save(predictor.state_dict(), param_dir + opt + "_Direc_predictor_last.pth")
            np.savetxt(param_dir + opt + '_Direc_predictor_recorder.txt', np.array(loss_rec), delimiter=',')


def show(rec, lab, load, idx=1024, save=False):

    rec_np = rec[1, :, :, :].detach().cpu().numpy()
    lab_np = lab[1, :, :, :].detach().cpu().numpy()
    load_np = load[1, ].detach().cpu().numpy()
    if idx == 1024:
        for i in range(rec_np.shape[0]):
            rec_frame_np = rec_np[i, :, :]
            lab_frame_np = lab_np[i, :, :]
            plt.figure(figsize=(16, 8))
            plt.subplot(1, 2, 1)
            plt.imshow(rec_frame_np, vmin=-1, vmax=1)
            plt.colorbar()
            plt.title("Reconstructed of R%.2f_E%.2f_T%.2f" % (load_np[0], load_np[1], load_np[2]))
            plt.subplot(1, 2, 2)
            plt.imshow(lab_frame_np, vmin=-1, vmax=1)
            plt.colorbar()
            plt.title("Label of R%.2f_E%.2f_T%.2f" % (load_np[0], load_np[1], load_np[2]))
            if save:
                plt.savefig(r'results//'+str(i)+".png")
            plt.show()
    else:
        rec_frame_np = rec_np[idx, :, :]#.detach().cpu().numpy()
        lab_frame_np = lab_np[idx, :, :]#.detach().cpu().numpy()

        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(rec_frame_np, vmin=-1, vmax=1)
        plt.colorbar()
        plt.title("Reconstructed of R%.2f_E%.2f_T%.2f"%(load_np[0], load_np[1], load_np[2]))
        plt.subplot(1, 2, 2)
        plt.imshow(lab_frame_np, vmin=-1, vmax=1)
        plt.colorbar()
        plt.title("Label of R%.2f_E%.2f_T%.2f"%(load_np[0], load_np[1], load_np[2]))
        plt.show()


if __name__ == '__main__':
    train_vqvae(opt='SDEG')
    train_predictor(opt='SDEG')
    train_simCNN(opt='SDEG')
    train_vqvae(opt='MTDMG')
    train_predictor(opt='MTDMG')
    # train_simCNN(opt='MTDMG')
    train_vqvae(opt='MCDMG')
    train_predictor(opt='MCDMG')
    # train_simCNN(opt='MCDMG')
    train_vqvae(opt='FDMG')
    train_predictor(opt='FDMG')
    # train_simCNN(opt='FDMG')
    train_vqvae(opt='MDMG')
    train_predictor(opt='MDMG')
    train_simCNN(opt='MDMG')

