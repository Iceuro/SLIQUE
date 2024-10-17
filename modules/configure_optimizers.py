from modules import scheduler as sch
import torch

def configure_optimizers(args, optimizer, cur_iter=-1):
    iters = args.iters







    # if args.reload:
    #     fl = torch.load(args.model_path + 'CONTRIQUE.tar')
    #     optimizer.load_state_dict(fl['optimizer'])
    #     cur_iter = fl['scheduler']['last_epoch'] - 1
    
    if args.lr_schedule == 'warmup-anneal':
        scheduler = sch.LinearWarmupAndCosineAnneal(
            optimizer,
            args.warmup,
            iters,
            last_epoch=cur_iter,
        )
    elif args.lr_schedule == 'linear':
        scheduler = sch.LinearLR(optimizer, iters, last_epoch=cur_iter)
    elif args.lr_schedule == 'const':
        scheduler = sch.LinearWarmupAndConstant(
            optimizer,
            args.warmup,
            iters,
            last_epoch=cur_iter,
        )
    else:
        raise NotImplementedError
    
    # if args.reload:
    #     scheduler.load_state_dict(fl['scheduler'])
    
    return  scheduler