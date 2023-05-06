from networks import VQVAE_Simp, Predictor, simple_CNN
from data import DmgData
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def eval_vae_predictor(opt='SDEG'):
    data_path = r'F:\DATASET\LVI_SM\npy\dataset2'
    param_dir = r'D:\Codes\LiminateDMG_SM\20230318\params\\'
    batch_size = 1
    num_workers = 0
    device = torch.device('cuda:0')

    data_loader = DmgData(data_path, opt, batch_size, num_workers, device)
    valid_dl = data_loader.val_dataloader()
    embedding_dim = 16
    num_embeddings = 256

    coder_model = VQVAE_Simp(opt, embedding_dim=embedding_dim, num_embeddings=num_embeddings).to(device)
    coder_model.load_state_dict(torch.load(param_dir + opt + "_%d_%d_coder_best.pth"%(embedding_dim, num_embeddings)))
    coder_model.eval()

    predictor = Predictor(embedding_dim=embedding_dim, size=12).to(device)
    predictor.load_state_dict(torch.load(param_dir + opt + "_predictor_best.pth"))
    predictor.eval()

    for idx, [load, sql_npy] in enumerate(valid_dl):
        latent = predictor(load)
        pre_sql = coder_model.decode(latent)
        show(opt, pre_sql, sql_npy, load, method='VAE')


def eval_vae_predictor_TC(opt='MDEG'):
    data_path = r'F:\DATASET\LVI_SM\npy\dataset2'
    param_dir = r'D:\Codes\LiminateDMG_SM\20230318\params\\'
    batch_size = 1
    num_workers = 0
    device = torch.device('cuda:0')

    data_loader = DmgData(data_path, 'MDMG', batch_size, num_workers, device)
    valid_dl = data_loader.val_dataloader()
    embedding_dim = 16
    num_embeddings = 256

    coder_model = VQVAE_Simp(opt='MTDMG', embedding_dim=embedding_dim, num_embeddings=num_embeddings).to(device)
    coder_model.load_state_dict(torch.load(param_dir + 'MTDMG' + "_%d_%d_coder_best.pth"%(embedding_dim, num_embeddings)))
    coder_model.eval()

    predictor = Predictor(embedding_dim=embedding_dim, size=12).to(device)
    predictor.load_state_dict(torch.load(param_dir + 'MTDMG' + "_predictor_best.pth"))
    predictor.eval()

    coder_model_C = VQVAE_Simp(opt='MCDMG', embedding_dim=embedding_dim, num_embeddings=num_embeddings).to(device)
    coder_model_C.load_state_dict(torch.load(param_dir + 'MCDMG' + "_%d_%d_coder_best.pth"%(embedding_dim, num_embeddings)))
    coder_model_C.eval()

    predictor_C = Predictor(embedding_dim=embedding_dim, size=12).to(device)
    predictor_C.load_state_dict(torch.load(param_dir + 'MCDMG' + "_predictor_best.pth"))
    predictor_C.eval()

    for idx, [load, sql_npy] in enumerate(valid_dl):
        latent_T = predictor(load)
        pre_sql_T = coder_model.decode(latent_T)
        latent_C = predictor_C(load)
        pre_sql_C = coder_model_C.decode(latent_C)
        show_TC(pre_sql_T, pre_sql_C, sql_npy, load, method='VAE')


def eval_cnn_predictor(opt='SDEG'):
    data_path = r'F:\DATASET\LVI_SM\npy\dataset2'
    param_dir = r'D:\Codes\LiminateDMG_SM\20230318\params\\'
    batch_size = 1
    num_workers = 0
    device = torch.device('cuda:0')

    data_loader = DmgData(data_path, opt, batch_size, num_workers, device)
    valid_dl = data_loader.val_dataloader()

    predictor = simple_CNN(opt, embedding_dim=64).to(device)
    predictor.load_state_dict(torch.load(param_dir + opt + "_Direc_predictor_best.pth"))
    predictor.eval()

    for idx, [load, sql_npy] in enumerate(valid_dl):
        pre_sql = predictor(load)
        show(opt, pre_sql, sql_npy, load, method='CNN')


def show(opt, rec, lab, load, save=True, method='VAE'):

    rec_np = rec[0, :, :, :].detach().cpu().numpy()
    lab_np = lab[0, :, :, :].detach().cpu().numpy()
    load_np = load[0, ].detach().cpu().numpy()
    for i in range(rec_np.shape[0]):
        rec_frame_np = rec_np[i, :, :]
        lab_frame_np = lab_np[i, :, :]
        # plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(rec_frame_np, vmin=-1, vmax=1, cmap='jet')
        cb = plt.colorbar()
        cb.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        plt.title("Reconstructed of R%.2f_E%.2f_T%.2f" % (load_np[0], load_np[1], load_np[2]))
        plt.subplot(1, 2, 2)
        plt.imshow(lab_frame_np, vmin=-1, vmax=1, cmap='jet')
        cb2 = plt.colorbar()
        cb2.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        plt.title("Label of R%.2f_E%.2f_T%.2f" % (load_np[0], load_np[1], load_np[2]))
        if save:
            plt.savefig(r'results/'+method+'_'+opt+'_R%.2f_E%.2f_T%.2f_L%d.pdf'% (load_np[0], load_np[1], load_np[2], i))
        plt.show()


def show_TC(rec_T, rec_C, lab, load, save=True, method='VAE'):

    recT_np = rec_T[0, :, :, :].detach().cpu().numpy()
    recC_np = rec_C[0, :, :, :].detach().cpu().numpy()
    minus = (recT_np - recC_np) >= 0   # T > C 的位置
    rec_np = recT_np * minus - recC_np * (-(minus - 1))
    lab_np = lab[0, :, :, :].detach().cpu().numpy()
    load_np = load[0, ].detach().cpu().numpy()
    for i in range(rec_np.shape[0]):
        rec_frame_np = rec_np[i, :, :]
        lab_frame_np = lab_np[i, :, :]
        # plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(rec_frame_np, vmin=-1, vmax=1, cmap='jet')
        cb = plt.colorbar()
        cb.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        plt.title("Reconstructed of R%.2f_E%.2f_T%.2f" % (load_np[0], load_np[1], load_np[2]))
        plt.subplot(1, 2, 2)
        plt.imshow(lab_frame_np, vmin=-1, vmax=1, cmap='jet')
        cb2 = plt.colorbar()
        cb2.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        plt.title("Label of R%.2f_E%.2f_T%.2f" % (load_np[0], load_np[1], load_np[2]))
        if save:
            plt.savefig(r'results/'+method+'_'+"MDMG_d"+'_R%.2f_E%.2f_T%.2f_L%d.pdf'% (load_np[0], load_np[1], load_np[2], i))
        plt.show()


if __name__ == '__main__':
    # eval_vae_predictor(opt='SDEG')
    # eval_cnn_predictor(opt='SDEG')
    # eval_vae_predictor(opt='MTDMG')
    # eval_cnn_predictor(opt='MTDMG')
    # eval_vae_predictor(opt='MCDMG')
    # eval_cnn_predictor(opt='MCDMG')
    # eval_vae_predictor(opt='MDMG')
    # eval_cnn_predictor(opt='MDMG')
    eval_vae_predictor_TC(opt='MDMG')
