

import sys
import math
import os
import time
import shutil
import datetime
import pandas as pd

import click
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torchvision
import tqdm
import subprocess

import models
import datasets
import utils

from torch.distributions.normal import Normal


@click.command("ucvme")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default="DATA_DIR")
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--pretrained/--random", default=True)
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--run_test/--skip_test", default=True)
@click.option("--test_only/--run_all", default=False)

@click.option("--num_epochs", type=int, default=30)   
@click.option("--lr", type=float, default=0.0001)
@click.option("--weight_decay", type=float, default=1e-3)
@click.option("--lr_step_period", type=int, default=10)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=32)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)

@click.option("--reduced_set/--full_set", default=True)
@click.option("--rd_label", type=int, default=1000) 
@click.option("--rd_unlabel", type=int, default=9518)
@click.option("--ssl_mult", type=int, default=-1)
@click.option("--w_ulb", type=float, default=10)

@click.option("--pad_param", type=int, default=5)


@click.option("--y_mean", type=float, default=35)
@click.option("--y_std", type=float, default=11)

@click.option("--samp_fq", type=int, default=5)
@click.option("--samp_ssl", type=int, default=5)

@click.option("--drp_p", type=float, default=0.05)


def run(
    data_dir="DATA_DIR",
    output=None,
    pretrained=True,
    weights=None,
    run_test=True,
    test_only = False,

    num_epochs=30,
    lr=0.0001,
    weight_decay=1e-3,
    lr_step_period=10,
    num_workers=4,
    batch_size=32,
    device=None,
    seed=0,

    reduced_set = True,
    rd_label = 1000,
    rd_unlabel = 9518,

    ssl_mult = -1,
    w_ulb = 10,

    pad_param = 5,
    y_mean = 35,
    y_std = 11,
    samp_fq = 5,
    samp_ssl = 5,

    drp_p = 0.05
):
    

    command_args = sys.argv[:]

    print("Run with options:")
    for carg_itr in command_args:
        print(carg_itr)

    if reduced_set:
        if not os.path.isfile(os.path.join(data_dir, "FileList_ssl_{}_{}.csv".format(rd_label, rd_unlabel))):
            print("Generating new file list for ssl dataset")
            np.random.seed(0)
            
            data = pd.read_csv(os.path.join(data_dir, "FileList.csv"))
            data["SPLIT"].map(lambda x: x.upper())

            file_name_list = np.array(data[data['SPLIT']== 'TRAIN']['FileName'])
            np.random.shuffle(file_name_list)

            label_list = file_name_list[:rd_label]
            unlabel_list = file_name_list[rd_label:]

            data['SSL_SPLIT'] = "UNLABELED"
            data.loc[data['FileName'].isin(label_list), 'SSL_SPLIT'] = "LABELED"

            data.to_csv(os.path.join(data_dir, "FileList_ssl_{}_{}.csv".format(rd_label, rd_unlabel)),index = False)

    ssl_mult_choice = ssl_mult

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    def worker_init_fn(worker_id):                            
        # print("worker id is", torch.utils.data.get_worker_info().id)
        # https://discuss.pytorch.org/t/in-what-order-do-dataloader-workers-do-their-job/88288/2
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    # Set default output directory
    if output is None:
        assert 1==2, "need output option"

    os.makedirs(output, exist_ok = True)
    bkup_tmstmp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "gpu":
        device = torch.device("cuda")
    elif device == "cpu":
        device = torch.device("cpu")
    else:
        assert 1==2, "wrong parameter for device"


    model = models.resnet50_unc(pretrained=pretrained, drp_p = drp_p)
    model = torch.nn.DataParallel(model)

    model_1 = models.resnet50_unc(pretrained=pretrained, drp_p = drp_p)
    model_1 = torch.nn.DataParallel(model_1)

    model.to(device)
    model_1.to(device)


    if weights is not None:
        checkpoint = torch.load(weights)
        if checkpoint.get('state_dict'):
            model.load_state_dict(checkpoint['state_dict'])
        elif checkpoint.get('state_dict_0'):
            model.load_state_dict(checkpoint['state_dict_0'])
        else:
            assert 1==2, "state dict not found"


    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    optim_1 = torch.optim.Adam(model_1.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler_1 = torch.optim.lr_scheduler.StepLR(optim_1, lr_step_period)

    mean, std = utils.get_mean_and_std(datasets.UTKdta(root=data_dir, split="train"))
    print("mean std", mean, std)

    kwargs = {"target_type": ['age'],
              "mean": mean,
              "std": std
              }

    # Set up datasets and dataloaders
    dataset = {}
    dataset_trainsub = {}
    if reduced_set:
        dataset_trainsub['lb'] = datasets.UTKdta(root=data_dir, split="train", **kwargs, pad=pad_param, ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel), ssl_type = 1, ssl_mult = ssl_mult_choice)
        dataset_trainsub['unlb_0'] = datasets.UTKdta(root=data_dir, split="train", **kwargs, pad=pad_param, ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel), ssl_type = 2)
    else:
        assert 1==2, "not possible"

    dataset['train'] = dataset_trainsub
    dataset["val"] = datasets.UTKdta(root=data_dir, split="val", **kwargs, ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel))

    with open(os.path.join(output, "log.csv"), "a") as f:

        f.write("Run timestamp: {}\n".format(bkup_tmstmp))

        epoch_resume = 0
        bestLoss = float("inf")
        try:
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'], strict = False)
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])

            model_1.load_state_dict(checkpoint['state_dict_1'], strict = False)
            optim_1.load_state_dict(checkpoint['opt_dict_1'])
            scheduler_1.load_state_dict(checkpoint['scheduler_dict_1'])

            np_rndstate_chkpt = checkpoint['np_rndstate']
            trch_rndstate_chkpt = checkpoint['trch_rndstate']

            np.random.set_state(np_rndstate_chkpt)
            torch.set_rng_state(trch_rndstate_chkpt)

            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        if test_only:
            num_epochs = 0

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:

                start_time = time.time()

                if device.type == "cuda":
                    for i in range(torch.cuda.device_count()):
                        torch.cuda.reset_peak_memory_stats(i)

                
                ds = dataset[phase]
                if phase == "train":
                    dataloader_lb = torch.utils.data.DataLoader(
                        ds['lb'], batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"), worker_init_fn=worker_init_fn)
                    dataloader_unlb_0 = torch.utils.data.DataLoader(
                        ds['unlb_0'], batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"), worker_init_fn=worker_init_fn)
                    


                    loss_tr, loss_reg_0, loss_reg_1, cps, cps_l, cps_s, yhat_0, yhat_1, y, mean_0_ls, mean_1_ls, var_0_ls, var_1_ls = run_epoch(model, 
                                                                                                                                model_1, 
                                                                                                                                dataloader_lb, 
                                                                                                                                dataloader_unlb_0, 
                                                                                                                                phase == "train", 
                                                                                                                                optim, 
                                                                                                                                optim_1, 
                                                                                                                                device, 
                                                                                                                                w_ulb = w_ulb, 
                                                                                                                                y_mean = y_mean, 
                                                                                                                                y_std = y_std, 
                                                                                                                                samp_fq = samp_fq, 
                                                                                                                                samp_ssl = samp_ssl)

                    r2_value_0 = sklearn.metrics.r2_score(y, yhat_0)
                    r2_value_1 = sklearn.metrics.r2_score(y, yhat_1)

                    f.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                                phase,
                                                                loss_tr,
                                                                r2_value_0,
                                                                r2_value_1,
                                                                time.time() - start_time,
                                                                y.size,
                                                                sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                                batch_size,
                                                                loss_reg_0,
                                                                cps))
                    f.flush()
                
                    with open(os.path.join(output, "train_pred_{}.csv".format(epoch)), "w") as f_trnpred:
                        for clmn in range(mean_0_ls.shape[1]):
                            f_trnpred.write("m_0_{},".format(clmn))
                        for clmn in range(mean_1_ls.shape[1]):
                            f_trnpred.write("m_1_{},".format(clmn))
                        for clmn in range(var_0_ls.shape[1]):
                            f_trnpred.write("v_0_{},".format(clmn))
                        for clmn in range(var_1_ls.shape[1]):
                            f_trnpred.write("v_1_{},".format(clmn))
                        f_trnpred.write("\n".format(clmn))
                        
                        for rw in range(mean_0_ls.shape[0]):
                            for clmn in range(mean_0_ls.shape[1]):
                                f_trnpred.write("{},".format(mean_0_ls[rw, clmn]))
                            for clmn in range(mean_1_ls.shape[1]):
                                f_trnpred.write("{},".format(mean_1_ls[rw, clmn]))
                            for clmn in range(var_0_ls.shape[1]):
                                f_trnpred.write("{},".format(var_0_ls[rw, clmn]))
                            for clmn in range(var_1_ls.shape[1]):
                                f_trnpred.write("{},".format(var_1_ls[rw, clmn]))
                            f_trnpred.write("\n".format(clmn))

                
                else:
    
                    ds = dataset[phase]
                    dataloader = torch.utils.data.DataLoader(
                        ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))                        
                    
                    loss_valit, yhat, y, var_hat, var_e, var_a, mean_0_ls, var_0_ls = run_epoch_val(model = model, model_1 = model_1, dataloader = dataloader, train = False, optim = None, device = device, block_size=None, y_mean = y_mean, y_std = y_std, samp_fq = samp_fq)

                    r2_value = sklearn.metrics.r2_score(y, yhat)
                    loss = loss_valit

                    with open(os.path.join(output, "z_{}_epch{}_prd.csv".format(phase, epoch)), "a") as pred_out:
                        pred_out.write("yhat,y,var_hat, var_e, var_a\n")
                        for pred_itr in range(y.shape[0]):
                            pred_out.write("{},{},{},{},{}\n".format(yhat[pred_itr],
                            y[pred_itr], 
                            var_hat[pred_itr], 
                            var_e[pred_itr], 
                            var_a[pred_itr]))
                        pred_out.flush()
                    
                    with open(os.path.join(output, "val_predmcd0_{}.csv".format(epoch)), "w") as f_trnpred:
                        for clmn in range(mean_0_ls.shape[1]):
                            f_trnpred.write("m_0_{},".format(clmn))
                        for clmn in range(var_0_ls.shape[1]):
                            f_trnpred.write("v_0_{},".format(clmn))
                        f_trnpred.write("\n".format(clmn))
                        
                        for rw in range(mean_0_ls.shape[0]):
                            for clmn in range(mean_0_ls.shape[1]):
                                f_trnpred.write("{},".format(mean_0_ls[rw, clmn]))
                            for clmn in range(var_0_ls.shape[1]):
                                f_trnpred.write("{},".format(var_0_ls[rw, clmn]))
                            f_trnpred.write("\n".format(clmn))


                    f.write("{},{},{},{},{},{},{},{},{},{},{}".format(epoch,
                                                                phase,
                                                                loss,
                                                                r2_value,
                                                                time.time() - start_time,
                                                                y.size,
                                                                sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                                batch_size,
                                                                0,
                                                                0))
            
                    

                    f.write("\n")
                    f.flush()


            
            scheduler.step()
            scheduler_1.step()

            best_model_loss = loss_valit

            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'state_dict_1': model_1.state_dict(),
                'best_loss': bestLoss,
                'loss': loss,
                "best_model_loss": best_model_loss,
                'r2': r2_value,
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
                'opt_dict_1': optim_1.state_dict(),
                'scheduler_dict_1': scheduler_1.state_dict(),
                'np_rndstate': np.random.get_state(),
                'trch_rndstate': torch.get_rng_state()
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            
            if best_model_loss < bestLoss:
                print("saved best because {} < {}".format(best_model_loss, bestLoss))
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = best_model_loss


        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'], strict = False)
            model_1.load_state_dict(checkpoint['state_dict_1'], strict = False)

            f.write("Best validation loss {} from epoch {}, R2 {}\n".format(checkpoint["best_model_loss"], checkpoint["epoch"], checkpoint["r2"]))
            f.flush()

        if run_test:

            split_list = ["test", "val"]
            
            for split in split_list: 

                dataloader = torch.utils.data.DataLoader(
                    datasets.UTKdta(root=data_dir, split=split, **kwargs, ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel)),
                    batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"), worker_init_fn=worker_init_fn)
                total_loss, yhat, y, _, _, _, _, _ = run_epoch_val(model = model, model_1 = model_1, dataloader = dataloader, train = False, optim = None, device = device, block_size=None, y_mean = y_mean, y_std = y_std, samp_fq = samp_fq)

                f.write("{} - {} (one clip) R2:   {:.3f}\n".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), split, sklearn.metrics.r2_score(y, yhat)))
                f.write("{} - {} (one clip) MAE:  {:.2f}\n".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), split, sklearn.metrics.mean_absolute_error(y, yhat)))
                f.write("{} - {} (one clip) RMSE: {:.2f}\n".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), split, sklearn.metrics.mean_squared_error(y, yhat)**0.5))
                f.flush()







def run_epoch(model, 
            model_1, 
            dataloader_lb, 
            dataloader_unlb_0, 
            train, 
            optim, 
            optim_1, 
            device, 
            block_size=None, 
            run_dir = None, 
            test_val = None, 
            w_ulb = 10,  
            y_mean = 35, 
            y_std = 11, 
            samp_fq = 5, 
            samp_ssl = 5):
    
    model.train(train)
    model_1.train(train)

    total = 0  
    total_reg = 0 
    total_reg_1 = 0

    total_cps = 0
    total_cps_0 = 0
    total_cps_1 = 0


    n = 0 


    yhat_0 = []
    yhat_1 = []
    y = []

    mean2s_0_stack_ls = []
    mean2s_1_stack_ls = []
    var1s_0_stack_ls = []
    var1s_1_stack_ls = []

    start_frame_record = []
    vidpath_record = []

    torch.set_grad_enabled(train)

    total_itr_num = len(dataloader_lb)

    dataloader_lb_itr = iter(dataloader_lb)
    dataloader_unlb_0_itr = iter(dataloader_unlb_0)

    for train_iter in range(total_itr_num):
        (X_ulb_0, outcome_ulb) = dataloader_unlb_0_itr.next()

        X_ulb_0 = X_ulb_0.to(device)

        all_output_unlb_0_pred_0, var_unlb_0_pred_0 = model(X_ulb_0)
        all_output_unlb_1_pred_0, var_unlb_1_pred_0 = model_1(X_ulb_0)
        
        mean1s_0 = []
        mean2s_0 = []
        var1s_0 = []

        mean1s_1 = []
        mean2s_1 = []
        var1s_1 = []

        X_ulb_in = X_ulb_0

        with torch.no_grad():
            for samp_ssl_itr in range(samp_ssl):
                mean1_raw_0, var1_raw_0 = model(X_ulb_in)
                mean1_0 = mean1_raw_0.view(-1)
                var1_0 = var1_raw_0.view(-1)

                mean1s_0.append(mean1_0** 2)
                mean2s_0.append(mean1_0)
                var1s_0.append(var1_0)

                mean1_raw_1, var1_raw_1 = model_1(X_ulb_in)
                mean1_1 = mean1_raw_1.view(-1)
                var1_1 = var1_raw_1.view(-1)

                mean1s_1.append(mean1_1** 2)
                mean2s_1.append(mean1_1)
                var1s_1.append(var1_1)


        mean2s_0_stack = torch.stack(mean2s_0, dim=1).to("cpu").detach().numpy()
        mean2s_0_stack_ls.append(mean2s_0_stack)
        var1s_0_stack = torch.stack(var1s_0, dim=1).to("cpu").detach().numpy()
        var1s_0_stack_ls.append(var1s_0_stack)

        mean1s_0_ = torch.stack(mean1s_0, dim=0).mean(dim=0)
        mean2s_0_ = torch.stack(mean2s_0, dim=0).mean(dim=0)
        var1s_0_ = torch.stack(var1s_0, dim=0).mean(dim=0)

        mean2s_1_stack = torch.stack(mean2s_1, dim=1).to("cpu").detach().numpy()
        mean2s_1_stack_ls.append(mean2s_1_stack)
        var1s_1_stack = torch.stack(var1s_1, dim=1).to("cpu").detach().numpy()
        var1s_1_stack_ls.append(var1s_1_stack)

        mean1s_1_ = torch.stack(mean1s_1, dim=0).mean(dim=0)
        mean2s_1_ = torch.stack(mean2s_1, dim=0).mean(dim=0)
        var1s_1_ = torch.stack(var1s_1, dim=0).mean(dim=0)


        all_output_unlb_0_pslb = mean2s_0_
        all_output_unlb_1_pslb = mean2s_1_

        avg_mean01 = (all_output_unlb_0_pslb + all_output_unlb_1_pslb)/2
        avg_var01 = (var1s_0_ + var1s_1_)/2

        loss_mse_cps_0 = ((all_output_unlb_0_pred_0.view(-1) - avg_mean01)**2)
        loss_mse_cps_1 = ((all_output_unlb_1_pred_0.view(-1) - avg_mean01)**2)

        loss_cmb_cps_0 = 0.5 * (torch.mul(torch.exp(-avg_var01), loss_mse_cps_0) + avg_var01 )
        loss_cmb_cps_1 = 0.5 * (torch.mul(torch.exp(-avg_var01), loss_mse_cps_1) + avg_var01 )

        loss_reg_cps0 = loss_cmb_cps_0.mean()
        loss_reg_cps1 = loss_cmb_cps_1.mean()

        
        var_loss_ulb_0 = ((var_unlb_0_pred_0.view(-1) - avg_var01)**2).mean()
        var_loss_ulb_1 = ((var_unlb_1_pred_0.view(-1) - avg_var01)**2).mean()


        loss_reg_cps = (loss_reg_cps0 + loss_reg_cps1) + (var_loss_ulb_0 + var_loss_ulb_1)

        (X, outcome ) = dataloader_lb_itr.next()


        y.append(outcome.detach().cpu().numpy())
        X = X.to(device)

        outcome = outcome.to(device)


        all_output = model(X)
        all_output_1 = model_1(X)                    
        

        mean_raw, var_raw = all_output
        mean = mean_raw.view(-1)
        var = var_raw.view(-1)

        mean_1_raw, var_1_raw = all_output_1
        mean_1 = mean_1_raw.view(-1)
        var_1 = var_1_raw.view(-1)



        loss_mse = (mean - (outcome - y_mean) / y_std) ** 2
        loss1 = torch.mul(torch.exp(-(var + var_1) / 2), loss_mse)
        loss2 = (var + var_1) / 2
        loss = .5 * (loss1 + loss2)

        loss_reg_0 = loss.mean()
        yhat_0.append(all_output[0].view(-1).to("cpu").detach().numpy() * y_std + y_mean)

        
        

        loss_mse_1 = (mean_1 - (outcome - y_mean) / y_std) ** 2
        loss1_1 = torch.mul(torch.exp(-(var + var_1) / 2), loss_mse_1)
        loss2_1 = (var + var_1) / 2
        loss_1 = .5 * (loss1_1 + loss2_1)

        loss_reg_1 = loss_1.mean()
        yhat_1.append(all_output_1[0].view(-1).to("cpu").detach().numpy() * y_std + y_mean)


        loss_reg = (loss_reg_0 + loss_reg_1)

        loss = loss_reg + w_ulb * loss_reg_cps + ((var_1 - var) ** 2).mean()

        
        if train:
            optim.zero_grad()
            optim_1.zero_grad()
            loss.backward()
            optim.step()
            optim_1.step()

        total += loss.item() * outcome.size(0)
        total_reg += loss_reg_0.item() * outcome.size(0)
        total_reg_1 += loss_reg_1.item() * outcome.size(0)

        total_cps += loss_reg_cps.item() * outcome.size(0)
        total_cps_0 += loss_reg_cps0.item() * outcome.size(0)
        total_cps_1 += loss_reg_cps1.item() * outcome.size(0)

        n += outcome.size(0)

        if train_iter % 10 == 0:
            print("phase {} itr {}/{}: ls {:.2f}({:.2f}) rg0 {:.4f} ({:.2f}) rg1 {:.4f} ({:.2f}) cps {:.4f} ({:.2f}) cps0 {:.4f} ({:.2f}) cps1 {:.4f} ({:.2f})".format(train,
                train_iter, total_itr_num, 
                total / n, loss.item(), 
                total_reg/n, loss_reg_0.item(), 
                total_reg_1/n, loss_reg_1.item(), 
                total_cps/n, loss_reg_cps.item(),
                total_cps_0/n, loss_reg_cps0.item(),
                total_cps_1/n, loss_reg_cps1.item()), flush = True)


    yhat_0 = np.concatenate(yhat_0)
    yhat_1 = np.concatenate(yhat_1)
        

    y = np.concatenate(y)

    mean2s_0_stack_ls = np.concatenate(mean2s_0_stack_ls)
    mean2s_1_stack_ls = np.concatenate(mean2s_1_stack_ls)
    var1s_0_stack_ls = np.concatenate(var1s_0_stack_ls)
    var1s_1_stack_ls = np.concatenate(var1s_1_stack_ls)

    return total / n, total_reg / n, total_reg_1 / n, total_cps / n, total_cps_0 / n, total_cps_1 / n, yhat_0, yhat_1, y, mean2s_0_stack_ls, mean2s_1_stack_ls, var1s_0_stack_ls, var1s_1_stack_ls







def run_epoch_val(model, 
                model_1, 
                dataloader, 
                train, 
                optim, 
                device, 
                block_size=None, 
                y_mean = 35, 
                y_std = 11, 
                samp_fq = 5):


    model.train(False)
    model_1.train(False)

    total = 0 
    n = 0   

    yhat = []
    y = []

    var_hat = []
    var_e = []
    var_a = []

    mean2s_0_stack_ls = []
    var1s_0_stack_ls = []
    mean2s_0_stack_ls_m1 = []
    var1s_0_stack_ls_m1 = []

    mean2s_0_stack_ls_avg = []
    var1s_0_stack_ls_avg = []

    with torch.no_grad():
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (X, outcome) in dataloader:

                y.append(outcome.numpy())
                X = X.to(device)
                outcome = outcome.to(device)

                mean1s = []
                mean2s = []
                var1s = []


                mean1s_m1 = []
                mean2s_m1 = []
                var1s_m1 = []

                for samp_itr in range(samp_fq):
                    all_ouput = model(X)
                    mean1_raw, var1_raw = all_ouput

                    mean1 = mean1_raw.view(-1)
                    var1 = var1_raw.view(-1)

                    mean1s.append(mean1** 2)
                    mean2s.append(mean1)
                    var1s.append(torch.exp(var1))


                    all_ouput_m1 = model_1(X)
                    mean1_raw_m1, var1_raw_m1 = all_ouput_m1

                    mean1_m1 = mean1_raw_m1.view(-1)
                    var1_m1 = var1_raw_m1.view(-1)

                    mean1s_m1.append(mean1_m1** 2)
                    mean2s_m1.append(mean1_m1)
                    var1s_m1.append(torch.exp(var1_m1))


                mean2s_0_stack = torch.stack(mean2s, dim=1).to("cpu").detach().numpy()
                mean2s_0_stack_ls.append(mean2s_0_stack)
                var1s_0_stack = torch.stack(var1s, dim=1).to("cpu").detach().numpy()
                var1s_0_stack_ls.append(var1s_0_stack)

                mean1s_ = torch.stack(mean1s, dim=0).mean(dim=0)
                mean2s_ = torch.stack(mean2s, dim=0).mean(dim=0)
                var1s_ = torch.stack(var1s, dim=0).mean(dim=0)


                mean2s_0_stack_m1 = torch.stack(mean2s_m1, dim=1).to("cpu").detach().numpy()
                mean2s_0_stack_ls_m1.append(mean2s_0_stack_m1)
                var1s_0_stack_m1 = torch.stack(var1s_m1, dim=1).to("cpu").detach().numpy()
                var1s_0_stack_ls_m1.append(var1s_0_stack_m1)

                mean2s_0_stack_ls_avg.append((mean2s_0_stack + mean2s_0_stack_m1) / 2)
                var1s_0_stack_ls_avg.append((var1s_0_stack + var1s_0_stack_m1) / 2)


                mean1s_m1_ = torch.stack(mean1s_m1, dim=0).mean(dim=0)
                mean2s_m1_ = torch.stack(mean2s_m1, dim=0).mean(dim=0)
                var1s_m1_ = torch.stack(var1s_m1, dim=0).mean(dim=0)


                var2 = mean1s_ - mean2s_ ** 2
                var_ = var1s_ + var2
                var_norm = var_ / var_.max()         


                var2_m1 = mean1s_m1_ - mean2s_m1_ ** 2
                var_m1_ = var1s_m1_ + var2_m1
                var_m1_norm = var_m1_ / var_m1_.max()      



                yhat.append(((mean2s_ + mean2s_m1_) / 2).to("cpu").detach().numpy() * y_std + y_mean)
                var_hat.append(((var_norm + var_m1_norm) / 2).to("cpu").detach().numpy())
                var_e.append(((var2 + var2_m1) / 2).to("cpu").detach().numpy())
                var_a.append(((var1s_ + var1s_m1_) / 2).to("cpu").detach().numpy())

                loss = torch.nn.functional.mse_loss( (mean2s_ + mean2s_m1_) / 2 , (outcome - y_mean) / y_std )

                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                total += loss.item() * X.size(0)
                n += X.size(0)

                pbar.set_postfix_str("{:.2f} ({:.2f})".format(total / n, loss.item()))
                pbar.update()

    yhat = np.concatenate(yhat)
    var_hat = np.concatenate(var_hat)
    var_e = np.concatenate(var_e)
    var_a = np.concatenate(var_a)
    y = np.concatenate(y)

    mean2s_0_stack_ls_avg = np.concatenate(mean2s_0_stack_ls_avg)
    var1s_0_stack_ls_avg = np.concatenate(var1s_0_stack_ls_avg)

    return total / n, yhat, y, var_hat, var_e, var_a, mean2s_0_stack_ls_avg, var1s_0_stack_ls_avg




if __name__ == '__main__':
    run()