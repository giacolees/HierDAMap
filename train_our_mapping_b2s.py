import os
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
from utils import distributed_context

score_th = 0.5
torch.autograd.set_detect_anomaly(True)

def get_val_info(model, valloader, device, use_tqdm=True):
    total_intersect = 0.0
    total_union = 0
    confusion_c = BinaryConfusionMatrix(1)
    print('running eval...')
    loader = tqdm(valloader) if use_tqdm else valloader
    with torch.no_grad():
        for batch in loader:
            allimgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs,_ = batch

            preds, _, _, _, _, _, pred_car = model(allimgs.to(device), rots.to(device),trans.to(device), intrins.to(device),
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
          max_grad_norm=5.0, gpuid=0, is_distributed=False, use_tqdm=True):
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    logging.basicConfig(filename=os.path.join(logdir, "results.log"),
                        filemode='w',
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    # Set up data loaders
    strainloader, ttrainloader, tvalloader, sampler_strain, sampler_ttrain = compile_data_mapping(
        version, dataroot, data_aug_conf, grid_conf, nsweeps, domain_gap, source, target, bsz,
        nworkers, flip=True, is_distributed=is_distributed
    )
    
    datalen = min(len(strainloader), len(ttrainloader))

    if use_tqdm and dist.get_rank()==0:
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
    if is_distributed:
        # Important: Use SyncBatchNorm to synchronize BN statistics across processes
        student_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student_model)
        teacher_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model)
        
        # Use find_unused_parameters=False for better performance if all parameters are used
        student_model = DDP(student_model, device_ids=[gpuid], find_unused_parameters=True)
        teacher_model = DDP(teacher_model, device_ids=[gpuid], find_unused_parameters=True)
        
        # Get local rank for logging
        local_rank = dist.get_rank()
        logger.info(f"Initialized process with rank {local_rank}")

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
    opt = AdamW(student_model.parameters(), lr=lr)
    sched = StepLR(opt, 10, 0.1)
    
    # Configuration for consistency ramp-up
    uplr = int(datalen * nepochs * 0.6)
    
    # Initial weight and counters
    wp = 0.5
    loss_nan_counter = 0

    # Start training loop
    for epo in range(1, nepochs + 1):
        # Synchronize at epoch start
        if is_distributed:
            dist.barrier()
            local_rank = dist.get_rank()
            logger.info(f"Starting Epoch {epo} for rank: {local_rank}")
            
            # Set different seeds for each process to ensure data diversity
            sampler_strain.set_epoch(epo)
            sampler_ttrain.set_epoch(epo)
        
        # Set models to correct modes
        student_model.train()
        teacher_model.eval()  # Teacher model always in eval mode
        
        # Ensure teacher params don't receive gradients
        for param in teacher_model.parameters():
            param.requires_grad_(False)
        
        iteration = (epo - 1) * datalen
        
        # Training loop for current epoch
        for batch_idx, (batch_strain_data, batch_ttrain_data) in enumerate(zip(strainloader, ttrainloader)):
            t0 = time.time()
            
            # Process source domain data (student)
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
                # Important: ALWAYS detach teacher outputs to prevent gradient flow
                preds_un_ori, _, _, x_bev_un_ori, _, mask_mini_un_ori, pred_car_ori = teacher_model(
                    un_img_ori_gpu, un_rots_gpu, un_trans_gpu, un_intrins_gpu,
                    un_post_rots_ori_gpu, un_post_trans_ori_gpu, None
                )
                
                # Explicitly detach all teacher outputs to ensure no gradient flow
                preds_un_ori = preds_un_ori.detach()
                x_bev_un_ori = x_bev_un_ori.detach()
                pred_car_ori = pred_car_ori.detach()
            
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
            
            # Perform EMA update for teacher model
            with torch.no_grad():

                # Get state dictionaries with proper handling for DDP
                if is_distributed:
                    student_dict = student_model.module.state_dict()
                    teacher_dict = teacher_model.module.state_dict()
                else:
                    student_dict = student_model.state_dict()
                    teacher_dict = teacher_model.state_dict()
                
                # Update teacher parameters using EMA
                for k in teacher_dict.keys():
                    if k.find('num_batches_tracked') == -1:  # Skip batch norm statistics
                        teacher_dict[k] = teacher_dict[k] * alpha + student_dict[k] * (1 - alpha)
                
                # Load updated weights back to teacher model
                if is_distributed:
                    teacher_model.module.load_state_dict(teacher_dict)
                else:
                    teacher_model.load_state_dict(teacher_dict)

            if is_distributed:
                dist.barrier()
            
            # Logging (only from main process)
            if (not is_distributed or dist.get_rank() == 0) and batch_idx == datalen-1:
                # Calculate IoU metrics
                _, _, iou = get_batch_iou_mapping(onehot_encoding_mapping(preds), binimgs_s['iou'].to(device))
                _, _, iou_un = get_batch_iou_mapping(onehot_encoding_mapping(preds_un_ori), un_binimgs_t['iou'].to(device))
                
                # Calculate car IoU
                iou_car, miou_car = singleBinary(pred_car_ori.sigmoid() > score_th, un_binimgs_car > 0, dim=1)
                
                # Log information
                logger.info(f"EVAL[{int(epo):>3d}]: [{batch_idx:>4d}/{datalen}]:    "
                            f"lr {opt.param_groups[0]['lr']:>.2e}   "
                            f"w1 {w1:>7.4f}   "
                            f"loss: {loss.item():>7.4f}   "
                            f"loss_bev: {loss_bev.item():>7.4f}   "
                            f"mIOU_car: {miou_car:>7.4f}  "
                            f"tIOU: {np.array2string(iou[1:].cpu().numpy(), precision=3, floatmode='fixed')}  "
                            f"IOU_un: {np.array2string(iou_un[1:].cpu().numpy(), precision=3, floatmode='fixed')}  "
                            )
            
            iteration += 1
        
        # Step scheduler at the end of epoch
        sched.step()
        
        # Validation after each epoch (only on main process)
        if not is_distributed or dist.get_rank() == 0:
            # Run validation
            val_info = get_val_info(teacher_model, tvalloader, device)
            
            # Log validation results
            logger.info(f"TargetVAL[{epo:>2d}]:    "
                        f"mIOU_car: {val_info['miou_car']:>7.4f}  "
                        f"TargetIOU: {np.array2string(val_info['iou'][1:].cpu().numpy(), precision=3, floatmode='fixed')}  "
                        f"mIOU: {val_info['miou']:>7.4f}  "
                        )
            
            # Save model checkpoint
            mname = os.path.join(logdir, f"model{epo}.pt")
            print('saving', mname)
            
            # Save state dict with proper handling for DDP
            if is_distributed:
                state_dict = teacher_model.module.state_dict()
            else:
                state_dict = teacher_model.state_dict()
                
            torch.save(state_dict, mname)
        
        # Synchronize before next epoch
        if is_distributed:
            dist.barrier()


if __name__ == '__main__':
    version =  'v1.0-mini'
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
    b = 10
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

        train(logdir='./ours_mapping_' + source+'_'+target+'_2', version=version, dataroot='./nuscenes_mini',
            grid_conf=grid_conf, data_aug_conf=data_aug_conf,
            domain_gap=True, source=source, target=target, nsweeps=3,
            bsz=b, nworkers=6, lr=lr, weight_decay=1e-2, nepochs=50, gpuid=idx_gpu, is_distributed=is_distributed, use_tqdm=True)

        if rank == 0:
            print("Benchmark finished.")
