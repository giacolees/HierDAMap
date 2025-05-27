import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import time
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from data.nus_mapping import compile_data_mapping
from Our.LSS_model_da import LiftSplatShoot_mask_mix
from tools import *
from torch.optim import SGD, AdamW, Adam
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from Our.confusion import BinaryConfusionMatrix, singleBinary
from Our.loss import SegmentationLoss
from utils import distributed_context, get_rank, get_latest_model_path, save_checkpoint
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import jaccard_score, confusion_matrix
import re

score_th = 0.5

def get_val_info(model, valloader, device, use_tqdm=True):
    total_intersect = 0.0
    total_union = 0
    confusion_c = BinaryConfusionMatrix(1)

    loader = tqdm(valloader) if use_tqdm else valloader

    with torch.no_grad():
        for batch in loader:
            allimgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs,_ = batch

            preds, _, _, _, _, _, pred_car = model(allimgs.to(device,), rots.to(device),trans.to(device), intrins.to(device),
                         post_rots.to(device),post_trans.to(device), lidars.to(device), )

            # iou
            intersect, union, _ = get_batch_iou_mapping(onehot_encoding_mapping(preds),  binimgs['iou'].to(device).long())
            total_intersect += intersect
            total_union += union
            scores_c = pred_car.cpu().sigmoid()
            confusion_c.update(scores_c > score_th, (binimgs['car']) > 0)

    iou = total_intersect / (total_union+1e-7)
    miou = torch.mean(iou[1:])

    return {'iou': iou,'miou': miou,'miou_car': confusion_c.mean_iou}

def train(logdir, grid_conf, data_aug_conf, version, dataroot, nsweeps, domain_gap, source, target, bsz, nworkers, lr, weight_decay, nepochs,
          max_grad_norm=5.0, gpuid=0, use_tqdm=True):

    if not os.path.exists(logdir):
        os.mkdir(logdir)

    logging.basicConfig(filename=os.path.join(logdir, "results.log"),
                        filemode='a',
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    # Set up data loaders
    strainloader, ttrainloader, tvalloader, sampler_strain, sampler_ttrain, sampler_tval = compile_data_mapping(
        version, dataroot, data_aug_conf, grid_conf, nsweeps, domain_gap, source, target, bsz,
        nworkers, flip=True, is_distributed=dist.is_initialized()
    )
    
    datalen = min(len(strainloader), len(ttrainloader))
    if (dist.is_initialized() == False):
        strainloader = tqdm(strainloader)
        ttrainloader = tqdm(ttrainloader)

    # Initialize models
    device = torch.device(f'cuda:{gpuid}')
    student_model = LiftSplatShoot_mask_mix(grid_conf, data_aug_conf, outC=4)
    teacher_model = LiftSplatShoot_mask_mix(grid_conf, data_aug_conf, outC=4)
    
    # Move models to device
    student_model.to(device)
    teacher_model.to(device)

    # Set up distributed training if needed
    if dist.is_initialized():
        # Important: Use SyncBatchNorm to synchronize BN statistics across processes
        student_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student_model)
        teacher_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model)
        
        # Use find_unused_parameters=False for better performance if all parameters are used
        student_model = DDP(student_model, device_ids=[gpuid], find_unused_parameters=True)
        teacher_model = DDP(teacher_model, device_ids=[gpuid], find_unused_parameters=True)
        
        # Get local rank for logging
        local_rank = get_rank()
        print(f"Initialized process with rank {local_rank}")

    # Set up loss functions
    depth_weights = torch.Tensor([2.4, 1.2, 1.0, 1.1, 1.2, 1.4, 1.8, 2.3, 2.7, 3.5,
                                  3.6, 3.9, 4.8, 5.8, 5.4, 5.3, 5.0, 5.4, 5.3, 5.9,
                                  6.5, 7.0, 7.5, 7.5, 8.5, 10.3, 10.9, 9.8, 11.5, 13.1,
                                  15.1, 15.1, 16.3, 16.3, 17.8, 19.6, 21.8, 24.5, 24.5, 28.0,
                                  28.0]).to(device)
    loss_depth = DepthLoss(depth_weights).to(device)
    loss_pv = Dice_Loss().to(device)
    loss_final = SegmentationLoss(class_weights=[1.0, 2.0, 2.0, 2.0], weight=1.0).to(device)
    loss_mini = Dice_Loss().to(device)
    loss_car = Dice_Loss().to(device)
    
    # Set up optimizer and scheduler
    opt = AdamW(student_model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = StepLR(opt, 10, 0.1)
    
    # Configuration for consistency ramp-up
    uplr = int(datalen * nepochs * 0.6)
    
    # Initial weight and counters
    wp = 0.5
    loss_nan_counter = 0
    
    # Find latest checkpoint and resume if available
    start_epoch = 1
    checkpoint_path = None
    
    # Only rank 0 searches for checkpoints to avoid file system conflicts
    if not dist.is_initialized() or get_rank() == 0:
        checkpoint_path, latest_epoch = get_latest_model_path(logdir)
        logger.info(f"Found checkpoint: {checkpoint_path} from epoch {latest_epoch}")

    # Broadcast checkpoint path from rank 0 to other processes
    if dist.is_initialized():
        # Create a list for broadcasting
        checkpoint_list = [checkpoint_path]
        dist.broadcast_object_list(checkpoint_list, src=0)
        checkpoint_path = checkpoint_list[0]
        
        # Ensure all processes are synchronized
        dist.barrier()
    
    # Load checkpoint if available
    if checkpoint_path is not None:
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load checkpoint to CPU first to avoid OOM issues with large models
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state dicts
        if dist.is_initialized():
            # For DDP, load to module first
            student_model.module.load_state_dict(checkpoint['student_model'])
            teacher_model.module.load_state_dict(checkpoint['teacher_model'])
        else:
            student_model.load_state_dict(checkpoint['student_model'])
            teacher_model.load_state_dict(checkpoint['teacher_model'])
            
        # Load optimizer and scheduler states
        opt.load_state_dict(checkpoint['optimizer'])
        sched.load_state_dict(checkpoint['scheduler'])
        
        # Load other training states
        start_epoch = checkpoint['epoch'] + 1
        loss_nan_counter = checkpoint.get('loss_nan_counter', 0)
        wp = checkpoint.get('wp', 0.5)
        
        logger.info(f"Resumed training from epoch {start_epoch}")
    
    # Ensure start_epoch is synced across processes
    if dist.is_initialized():
        start_epoch_tensor = torch.tensor([start_epoch], device=device)
        dist.broadcast(start_epoch_tensor, src=0)
        start_epoch = start_epoch_tensor.item()
        
        # Additional sync barrier
        dist.barrier()

    # Start training loop
    for epo in range(start_epoch, nepochs + 1):
        # Synchronize at epoch start
        if dist.is_initialized():
            dist.barrier()
            local_rank = get_rank()
            logger.info(f"Starting Epoch {epo} for rank: {local_rank}")
            
            # Set different seeds for each process to ensure data diversity
            sampler_strain.set_epoch(epo)
            sampler_ttrain.set_epoch(epo)
            sampler_tval.set_epoch(epo)
        
        # Set models to correct modes
        student_model.train()
        teacher_model.eval()  # Teacher model always in eval mode
        
        # Ensure teacher params don't receive gradients
        for param in teacher_model.parameters():
            param.requires_grad_(False)
        
        iteration = (epo - 1) * datalen
        
        # Training loop for current epoch
        for batch_idx, (batch_strain_data, batch_ttrain_data) in enumerate(zip(strainloader, ttrainloader)):
            
            # Process source domain data
            imgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs_s, seg_mask_s = batch_strain_data
            
            # Move data to device
            imgs_gpu = imgs.to(device, non_blocking=True)
            rots_gpu = rots.to(device, non_blocking=True)
            trans_gpu = trans.to(device, non_blocking=True)
            intrins_gpu = intrins.to(device, non_blocking=True)
            post_rots_gpu = post_rots.to(device, non_blocking=True)
            post_trans_gpu = post_trans.to(device, non_blocking=True)
            
            # Reshape segmentation mask
            seg_mask = rearrange(seg_mask_s.mean(dim=-1), 'b n c h w -> (b n) c h w')
            seg_mask_gpu = seg_mask.to(device, non_blocking=True)
            
            # Reshape lidar data
            lidars_i = rearrange(lidars, 'b n c h w -> (b n) c h w') 
            lidars_i_gpu = lidars_i.to(device, non_blocking=True)
            
            # Forward pass for source domain (student)
            preds, depth, pv_out, x_bev, x_mini, mask_mini, pred_car = student_model(
                imgs_gpu, rots_gpu, trans_gpu, intrins_gpu, post_rots_gpu, post_trans_gpu,
                lidars_i_gpu, camdroup=False, seg_mask=seg_mask_gpu
            )
            
            # Process binary images
            binimgs = binimgs_s['index'].to(device, non_blocking=True)
            binimgs_car = binimgs_s['car'].to(device, non_blocking=True).unsqueeze(1)
            
            # Calculate source domain losses
            loss_f = loss_final(preds, binimgs)
            loss_f_car = loss_car(pred_car, binimgs_car)
            loss_p = wp * loss_pv(pv_out, seg_mask_gpu)

            loss_bev_t = 0.0
            loss_t = 0.0
            iou_ss_t = torch.zeros(4)
            iou_st_t = torch.zeros(4)
            iou_tt_t = torch.zeros(4)
            miou_car_ss_t = 0.0
            miou_car_st_t = 0.0
            miou_car_tt_t = 0.0
            
            # Handle depth loss
            lidars_gpu = lidars.to(device, non_blocking=True)
            loss_d, d_isnan = loss_depth(depth, lidars_gpu)
            loss_d = 0.1 * loss_d
            
            # Process target domain data
            un_image, un_rots, un_trans, un_intrins, un_post_rots, un_post_trans, un_lidars, un_binimgs_t, \
                un_img_ori, un_post_rots_ori, un_post_trans_ori, seg_mask_t, seg_mask_o = batch_ttrain_data
            
            # Move target data to GPU
            un_image_gpu = un_image.to(device, non_blocking=True)
            un_rots_gpu = un_rots.to(device, non_blocking=True)
            un_trans_gpu = un_trans.to(device, non_blocking=True)
            un_intrins_gpu = un_intrins.to(device, non_blocking=True)
            un_post_rots_gpu = un_post_rots.to(device, non_blocking=True)
            un_post_trans_gpu = un_post_trans.to(device, non_blocking=True)
            
            un_img_ori_gpu = un_img_ori.to(device, non_blocking=True)
            un_post_rots_ori_gpu = un_post_rots_ori.to(device, non_blocking=True)
            un_post_trans_ori_gpu = un_post_trans_ori.to(device, non_blocking=True)
            
            un_binimgs = un_binimgs_t['index'].to(device, non_blocking=True)
            un_binimgs_car = un_binimgs_t['car'].to(device, non_blocking=True).unsqueeze(1)
            
            # Teacher prediction on target domain (no_grad)
            with torch.no_grad():
                preds_un_ori, _, _, x_bev_un_ori, _, mask_mini_un_ori, pred_car_ori = teacher_model(
                    un_img_ori_gpu, un_rots_gpu, un_trans_gpu, un_intrins_gpu,
                    un_post_rots_ori_gpu, un_post_trans_ori_gpu, None
                )
            
            # Process segmentation mask for target
            seg_mask_un = rearrange(seg_mask_t.mean(dim=-1), 'b n c h w -> (b n) c h w')
            seg_mask_un_gpu = seg_mask_un.to(device, non_blocking=True)
            
            # Student prediction on target domain
            preds_un, _, pv_out_un, x_bev_un, x_mini_un, mask_mini_un, pred_car_un = student_model(
                un_image_gpu, un_rots_gpu, un_trans_gpu, un_intrins_gpu,
                un_post_rots_gpu, un_post_trans_gpu, None,
                camdroup=False, seg_mask=seg_mask_un_gpu
            )
            
            # Calculate target domain losses
            loss_p_un = wp * loss_pv(pv_out_un, seg_mask_un_gpu)
            
            # Calculate consistency weights using sigmoid ramp-up
            it = batch_idx + (epo - 1) * datalen
            weightofcon = sigmoid_rampup(it, uplr)
            w1 = min(0.002 + 0.1 * weightofcon, 0.1)
            
            # BEV consistency loss - using detached teacher outputs
            # No need to detach again as we already did in the with torch.no_grad() block
            loss_bev = (
                0.1 * w1 * F.mse_loss(x_bev_un, x_bev_un_ori) + 
                w1 * F.mse_loss(preds_un.sigmoid(), preds_un_ori.sigmoid()) +
                w1 * F.mse_loss(pred_car_un.sigmoid(), pred_car_ori.sigmoid())
            )
            
            # New tensors for combined mini BEV processing
            x_mini_combined = torch.cat((x_mini, x_mini_un), dim=1)
            mask_mini_combined = torch.cat((mask_mini, mask_mini_un), dim=1)
            
            # Calculate mini BEV loss
            loss_mbs = 0.1 * w1 * loss_mini(x_mini_combined, mask_mini_combined)
            
            # Combine all losses
            loss = loss_p + loss_p_un + loss_bev + loss_f + loss_f_car + 0.1 * loss_mbs
            if not d_isnan:
                loss = loss + loss_d
            else:
                loss_nan_counter += 1
            

            # Optimization step
            opt.zero_grad()
            loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_grad_norm)
            
            # Update weights
            opt.step()
            
            # EMA update for teacher model
            alpha = min(1.0 - 1.0 / (iteration + 1), 0.999)  # when iteration>=999, alpha==0.999
            
            if dist.is_initialized():
                dist.barrier()

            # Perform EMA update for teacher model
            with torch.no_grad():
                # Get state dictionaries with proper handling for DDP
                if dist.is_initialized():
                    student_dict = student_model.module.state_dict()
                    teacher_dict = teacher_model.module.state_dict()
                else:
                    student_dict = student_model.state_dict()
                    teacher_dict = teacher_model.state_dict()
                
                # Update teacher parameters using EMA
                for k in teacher_dict.keys():
                    teacher_dict[k] = teacher_dict[k] * alpha + student_dict[k] * (1 - alpha)

                # Load updated weights back to teacher model
                if dist.is_initialized():
                    teacher_model.module.load_state_dict(teacher_dict)
                else:
                    teacher_model.load_state_dict(teacher_dict)

            if dist.is_initialized():
                dist.barrier()

            #Student Model on Source 
            _, _, iou_ss = get_batch_iou_mapping(onehot_encoding_mapping(preds), binimgs_s['iou'].to(device))
            #Student Model on Target
            _, _, iou_st = get_batch_iou_mapping(onehot_encoding_mapping(preds_un), un_binimgs_t['iou'].to(device))
            #Teacher Model on Target
            _, _, iou_tt = get_batch_iou_mapping(onehot_encoding_mapping(preds_un_ori), un_binimgs_t['iou'].to(device))
            #Student Model on Source car
            iou_car_ss, miou_car_ss = singleBinary(pred_car.sigmoid() > score_th, binimgs_car > 0, dim=1)
            #Student Model on Target car
            iou_car_st, miou_car_st = singleBinary(pred_car_un.sigmoid() > score_th, un_binimgs_car > 0, dim=1)
            #Teacher Model on Target car
            iou_car_tt, miou_car_tt = singleBinary(pred_car_ori.sigmoid() > score_th, un_binimgs_car > 0, dim=1)

            loss_bev_t += loss_bev.item()
            loss_t += loss.item()
            iou_ss_t += iou_ss
            iou_st_t += iou_st
            iou_tt_t += iou_tt
            miou_car_ss_t += miou_car_ss
            miou_car_st_t += miou_car_st
            miou_car_tt_t += miou_car_tt
            
            # Logging (only from main process)
            if (not dist.is_initialized() or get_rank() == 0) and batch_idx > 0 and batch_idx % 17 == 0:

                # Log information
                logger.info(f"EVAL[{int(epo):>3d}]: [{batch_idx:>4d}/{datalen}]:    "
                            f"lr: {opt.param_groups[0]['lr']:>.2e}   "
                            f"w1: {w1:>7.4f}   "
                            f"alpha: {alpha}   "
                            f"loss: {loss_t:>7.4f}   "
                            f"loss_bev: {loss_bev_t:>7.4f}   "
                            f"mIOU_car_ss: {(miou_car_ss_t/batch_idx):>7.4f}   "
                            f"mIOU_car_st: {(miou_car_st_t/batch_idx):>7.4f}   "
                            f"mIOU_car_tt: {(miou_car_tt_t/batch_idx):>7.4f}   "
                            f"IOU Student on Source: {np.array2string((iou_ss_t/batch_idx).cpu().numpy(), precision=3, floatmode='fixed')}   "
                            f"IOU Student on Target: {np.array2string((iou_st_t/batch_idx).cpu().numpy(), precision=3, floatmode='fixed')}   "
                            f"IOU Teacher on Target : {np.array2string((iou_tt_t/batch_idx).cpu().numpy(), precision=3, floatmode='fixed')}   "
                            )
            
            # Save intermediate checkpoint every 50 batches or at a predefined frequency
            if (not dist.is_initialized() or get_rank() == 0) and batch_idx > 0 and batch_idx % 100 == 0:
                # Create intermediate checkpoint
                save_checkpoint(
                    logdir,
                    epo,
                    batch_idx,
                    student_model,
                    teacher_model,
                    opt,
                    sched,
                    loss_nan_counter,
                    wp,
                    is_distributed=dist.is_initialized(),
                    checkpoint_name=f"checkpoint_{epo}_batch_{batch_idx}.pt"
                )
                logger.info(f"Saved intermediate checkpoint at epoch {epo}, batch {batch_idx}")
            
            iteration += 1
        
        # Step scheduler at the end of epoch
        sched.step()
        
        # Save checkpoint at the end of each epoch
        if not dist.is_initialized() or get_rank() == 0:
            # Create full checkpoint
            save_checkpoint(
                logdir,
                epo,
                0,  # batch_idx=0 for end of epoch
                student_model,
                teacher_model,
                opt,
                sched,
                loss_nan_counter,
                wp,
                is_distributed=dist.is_initialized(),
                checkpoint_name=f"checkpoint_{epo}.pt"
            )
            logger.info(f"Saved checkpoint at end of epoch {epo}")
        
        # Validation after each epoch (only on main process)
        if not dist.is_initialized() or get_rank() == 0:
            # Run validation
            val_info = get_val_info(teacher_model, tvalloader, device)
            
            # Log validation results
            logger.info(f"TargetVAL[{epo:>2d}]:    "
                        f"mIOU_car: {val_info['miou_car']:>7.4f}  "
                        f"TargetIOU: {np.array2string(val_info['iou'][1:].cpu().numpy(), precision=5, floatmode='fixed')}  "
                        f"mIOU: {val_info['miou']:>7.4f}  "
                        )
            
            # Save final model for evaluation (different from checkpoints)
            mname = os.path.join(logdir, f"model{epo}.pt")
            print('saving model for evaluation', mname)
            
            # Save state dict with proper handling for DDP
            if dist.is_initialized():
                state_dict = teacher_model.module.state_dict()
            else:
                state_dict = teacher_model.state_dict()
                
            torch.save(state_dict, mname)
        
        # Synchronize before next epoch
        if dist.is_initialized():
            dist.barrier()

        

if __name__ == '__main__':
    version =   'v1.0-trainval-t'#
    grid_conf = {
        'xbound': [-30.0, 30.0, 0.15],
        'ybound': [-15.0, 15.0, 0.15],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [4.0, 45.0, 1.0],
    }
    data_aug_conf = {'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                              'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                     'Ncams': 6, 'up_scale': 4, 'H': 900, 'W': 1600,
                     'rand_resize': True, 'resize_lim': (0.20, 0.235), 'bot_pct_lim': (0.0, 0.22),
                     'rand_flip': True,
                     'rot': True, 'rot_lim': (-5.4, 5.4),
                     'color_jitter': True, 'color_jitter_conf': [0.2, 0.2, 0.2, 0.1],
                     'GaussianBlur': False, 'gaussion_c': (0, 2),
                     'final_dim': (128, 352),  # (224,480),#
                     'Aug_mode': 'hard',  # 'simple',#
                     'backbone': "efficientnet-b0",
                     }
    b = 12
    lr = 1e-3*b/4
    source_name_list = ['boston', 'singapore', 'day', 'dry']
    source = source_name_list[0]
    target = source_name_list[1]


    if sys.argv[1] == "False":
        is_distributed = False
    elif sys.argv[1] == "True":
        is_distributed = True
    else:
        RuntimeError("No valid input for is_distributed")

    # Use context manager for automatic cleanup

    with distributed_context(is_distributed=is_distributed) as ctx:
        rank = ctx['rank']
        world_size = ctx['world_size']
        current_gpu = ctx['gpu']
        idx_gpu = ctx['idx_gpu']
        device = ctx['device']
        # Print GPU info
        print(f"Process {rank} using GPU: {current_gpu} ({torch.cuda.get_device_name(current_gpu)})")

        train(logdir='./ours_mapping_' + source+'_'+target+'_8', version=version, dataroot='/davinci-1/home/dnazarpour/spoke_5/data/nuscenes',
            grid_conf=grid_conf, data_aug_conf=data_aug_conf,
            domain_gap=True, source=source, target=target, nsweeps=3,
            bsz=b, nworkers=6, lr=lr, weight_decay=1e-2, nepochs=30, gpuid=idx_gpu, use_tqdm=True)

        if rank == 0:
            print("Benchmark finished.")
