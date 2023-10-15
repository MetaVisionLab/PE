import torch
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel

from dassl.engine import TRAINER_REGISTRY
from dassl.engine.dg.vanilla import Vanilla
from dassl.engine.trainer import SimpleNet
from dassl.metrics import compute_accuracy
from dassl.modeling import BACKBONE_REGISTRY
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
from dassl.utils import mixup_data, cutmix_data, load_pretrained_weights
from share import share_dict


def split_params(root, fix_names, block):
    unfixed_params = []
    fixed_params = []
    def contains(s, names):
        for n in names:
            if n in s:
                return True
        return False
    def walk(node, fp, ufp):
        if isinstance(node, block) and node.is_me():
            fnames = []
            for name, param in node.named_parameters():
                if contains(name, fix_names):
                    fp.append(param)
                    fnames.append((node.fix_id, name))
                else:
                    ufp.append(param)
            print(f'Fixing:\t{fnames}')
        else:
            children = list(node.children())
            if len(children) > 0:
                for ch in children:
                    walk(ch, fp, ufp)
            else:
                ufp += list(node.parameters())
    walk(root, fixed_params, unfixed_params)
    del walk, contains
    return fixed_params, unfixed_params

@TRAINER_REGISTRY.register()
class DRT_DG_Mixup(Vanilla):
    """Dynamic transfer network with dynamic network.

    xxx.
    """
    scfg = None

    def __init__(self, cfg):
        super().__init__(cfg)
        self.mix_type = 'None'
        self.dy_flag = False
        self.st_flag = False
        pass

    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg
        self.block_ver = cfg.MODEL.BACKBONE.NAME.split('_')[-1]
        self.BLOCK_REF = BACKBONE_REGISTRY.get(f'Block_{self.block_ver}')
        print('Building model')
        self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        self.swa_model = AveragedModel(self.model)
        print('# params: {:,}'.format(count_num_param(self.model)))

        def count_params(params):
            return sum(p.numel() for p in params)

        # Aggregate part
        self.optim_agg = build_optimizer(self.model, cfg.OPTIM)
        print(f'total params of agg params:{count_params(list(self.model.parameters()))}')
        self.sched_agg = build_lr_scheduler(self.optim_agg, cfg.OPTIM)
        self.register_model('agg_model', self.model, self.optim_agg, self.sched_agg)
        self.register_model('swa_model', self.swa_model, None, None)

    def my_train(self, start_epoch, max_epoch):
        """Generic training loops."""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        self.before_train()
        self.dy_flag = False  # Disable domain-specific(dynamic) training
        self.st_flag = False  # Disable domain-invariant(static) training
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            # Warm up for Agg
            # if self.epoch < int(share_dict['args'].fix_when * self.max_epoch):
            #     self.run_epoch()
            # else:
            self.dy_flag = True # Enable domain-specific(dynamic) training
            self.st_flag = True # Enable domain-invariant(static) training
            self.run_epoch()
            self.dy_flag = False # Disable
            self.st_flag = False # Disable
            self.after_epoch()
        self.after_train()

    def train(self):
        self.my_train(self.start_epoch, self.max_epoch)

    def model_inference(self, input):
        return self.swa_model(input)

    def forward_backward(self, batch):
        input, label_a, label_b, lam = self.parse_batch_train(batch)
        loss_st, loss_dy, loss_agg = 0, 0, 0
        # End warming up for Agg
        # Start training domain-specific(dynamic) or domain-invariant(static) part
        if self.dy_flag or self.st_flag:
            # if self.st_flag:
            self.BLOCK_REF.fix('meta')
            output = self.model(input)
            loss_st = F.cross_entropy(output, label_a)
            self.BLOCK_REF.unfix()
            self.BLOCK_REF.fix('conv')
            output = self.model(input)
            loss_dy = F.cross_entropy(output, label_a)
            self.BLOCK_REF.unfix()

            self.optim_agg.zero_grad()
            loss = loss_dy + loss_st
            loss.backward()
            self.optim_agg.step()
            # self.optim_dy.step()
        else:
            output = self.model(input)
            loss_agg = F.cross_entropy(output, label_a)
            loss = loss_agg
            self.model_backward_and_update(loss, 'agg_model')
        # Update bn statistics for the swa_model
        with torch.no_grad():
            self.swa_model(input)

        loss_summary = {
            'loss_st': loss_st.item() if self.st_flag else 0,
            'loss_dy': loss_dy.item() if self.dy_flag else 0,
            'loss_agg': loss_agg.item() if not (self.dy_flag or self.st_flag) else 0,
            'loss_all': loss.item(),
            'acc': compute_accuracy(output, label_a)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.swa_model.update_parameters(self.model)
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)

        if self.mix_type == 'mixup':
            input_x, label_a, label_b, lam = mixup_data(input, label, alpha=1.0)
        elif self.mix_type == 'cutmix':
            input_x, label_a, label_b, lam = cutmix_data(input, label, alpha=1.0)
        elif self.mix_type == 'None':
            input_x, label_a, label_b, lam = mixup_data(input, label, alpha=0)

        return input_x, label_a, label_b, lam
