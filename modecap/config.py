from yacs.config import CfgNode as CN

_C = CN()
_C.device = 'cuda'
_C.distributed = False
_C.log_time = 20
_C.checkpoint_time = 2500
_C.save_dir = 'output'
_C.data_dir = 'data'
_C.num_workers = 8
_C.num_positions = 30 
_C.samples_per_gpu = 64
_C.model_path = None

_C.num_modes = 64
_C.hidden_size = 768
_C.hidden_dropout_prob = 0.1
_C.num_mask_decoder_layers = 2
_C.num_decoder_layers = 3
_C.num_img_encoder_layers = 3
_C.num_mode_encoder_layers = 6

_C.solver = CN()
_C.solver.lr = 2e-4
_C.solver.weight_decay = 1e-2
_C.solver.betas = (0.9, 0.999)
_C.solver.grad_clip = 1.0

_C.scheduler = CN()
_C.scheduler.warmup_steps = 2000
_C.scheduler.max_steps = 135000

_C.loss = CN()
_C.loss.label_smoothing = 0.1
_C.loss.commitment_cost = 0.25
