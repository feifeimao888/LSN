import os

import torch
from nets.training_strategy import (Dice_loss, Focal_Loss,  
                                    DeepSupervision_Loss)
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, 
                  epoch_step_val, gen, gen_val, UnFreeze_Epoch, cuda, dice_loss, focal_loss, cls_weights, 
                  num_classes, fp16, scaler, save_period, save_dir, local_rank=0, 
                  lambda_boundary = 0.0001, boundary_loss_flag = True, start_boundary=None,
                  select_val = True ):
    total_loss      = 0
    total_f_score   = 0

    val_loss        = 0
    val_f_score     = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{UnFreeze_Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            print ("快开f16，或者--------------")

        else:
            from torch.amp import autocast
            with autocast('cuda'):

                outputs = model_train(imgs)

                if isinstance(outputs, list):
                    all_loss = DeepSupervision_Loss(
                        outputs,
                        pngs,
                        labels,
                        weights,
                        num_classes,
                        lambda_boundary=lambda_boundary,
                        boundary_loss_flag=boundary_loss_flag,
                        epoch=epoch,
                        start_boundary = start_boundary,
                    )

                else:
                    if epoch < 1:
                        print("未使用深监督")


                with torch.no_grad():
                    #-------------------------------#
                    #   计算f_score
                    #-------------------------------#

                    if isinstance(outputs, list):
                        _f_score = f_score(outputs[-1], labels)
                    else:
                        _f_score = f_score(outputs, labels)

            scaler.scale(all_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss      += all_loss.item()
        total_f_score   += _f_score.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)







# #   ================ 验证 =======================
    if select_val:
        if local_rank == 0:
            pbar.close()
            print('Finish Train')
            print('Start Validation')
            pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{UnFreeze_Epoch}',postfix=dict,mininterval=0.3)

        model_train.eval()
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            imgs, pngs, labels = batch
            with torch.no_grad():
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs    = imgs.cuda(local_rank)
                    pngs    = pngs.cuda(local_rank)
                    labels  = labels.cuda(local_rank)
                    weights = weights.cuda(local_rank)


                outputs = model_train(imgs)

                if isinstance(outputs, list):

                    all_loss = DeepSupervision_Loss(
                        outputs,
                        pngs,
                        labels,
                        weights,
                        num_classes,
                        lambda_boundary=lambda_boundary,
                        boundary_loss_flag=boundary_loss_flag,
                        epoch=epoch,
                        start_boundary=start_boundary
                    )

                else:
                    print("未使用深监督")


                if isinstance(outputs, list):
                    _f_score = f_score(outputs[-1], labels)
                else:
                    _f_score = f_score(outputs, labels)

                val_loss    += all_loss.item()
                val_f_score += _f_score.item()
                
            if local_rank == 0:
                pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                    'f_score'   : val_f_score / (iteration + 1),
                                    'lr'        : get_lr(optimizer)})
                pbar.update(1)
                
        if local_rank == 0:
            pbar.close()
            print('Finish Validation')
            loss_history.append_loss(epoch + 1, total_loss/ epoch_step, val_loss/ epoch_step_val)
            eval_callback.on_epoch_end(epoch + 1, model_train)
            print('Epoch:'+ str(epoch+1) + '/' + str(UnFreeze_Epoch))
            print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
            

            #-----------------------------------------------#
            if (epoch + 1) % save_period == 0 or epoch + 1 == UnFreeze_Epoch:
                torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

            if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
                print('Save best model to best_epoch_weights.pth')
                torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
                
            torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))



    else:
        if local_rank == 0:
            pbar.close()
            print('Finish Train')
            
            loss_history.append_loss(epoch + 1, total_loss / epoch_step, total_loss / epoch_step)
            
            print('Epoch:'+ str(epoch+1) + '/' + str(UnFreeze_Epoch))
            print('Total Loss: %.3f' % (total_loss / epoch_step))




            save_filename = 'ep%03d-loss%.3f.pth' % ((epoch + 1), total_loss / epoch_step)
            save_path = os.path.join(save_dir, save_filename)
            torch.save(model.state_dict(), save_path)
            print(f'Model saved: {save_filename}')
