import torch.nn as nn
import numpy as np

X2CT = {
    'global': {
        'img_size': 128,
        'batch_size': 2,
    },
    # learning rate
    'optimizer': {
        'gen_lr': 1e-3,
        'betas': (0, 0.9),
    },
    # loss hyperparameter
    'process': {
        'class': 'Pre3DProcess',
        'kwargs': {
            'CT_edge_lambda': 0.05,
            'nerf_lambda': 0.001,
        }
    },
    # network parameters
    'generator': {
        'class': 'Renderer',
        'kwargs': {
            'step_length_path': './DRR_Parameters/step_length.npy',
            'idxs_path': './DRR_Parameters/idxs.npy',
            'target_path': './DRR_Parameters/target.npy',
            'source_path': './DRR_Parameters/source.npy',
            'coor_path': './DRR_Parameters/coords_3D.npy',
            'mean_path': './DRR_Parameters/mean_CT.npy',
            'drr_height': 128,
            'drr_weight': None,
            'device': 'cuda',
            'representation_kwargs': {
                'hidden_dim': 34,
                'norm_layer': nn.BatchNorm3d,
                'norm_layer2d': nn.BatchNorm2d,
                'input_dim': 2,
                'input_coor_dim': 1,
                'depths': [3, 3, 9, 3],
                'spacing': np.array([2.5, 2.5, 2.5]),
                'sdr': 949 // 2,
                'del_size': 5,
                'offset_points': 2
            },
        }
    },
    # path for CT and X-ray images
    'dataset': {
        'class': 'X2CT',
        'kwargs': {
            'CT_PATH': './crop_image/train',
            'XRAY_PATH': './drr_Xray/train',
            'MEAN_XRAY_PATH': './DRR_Parameters/mean_xray.npy',
            'STD_XRAY_PATH': './DRR_Parameters/std_xray.npy'
        }
    },
    # test snapshot
    'snapshot': {
        'test_dir': './drr_Xray/test/LIDC-IDRI-0001.npy',
        'MEAN_XRAY_PATH': './DRR_Parameters/mean_xray.npy',
        'STD_XRAY_PATH': './DRR_Parameters/std_xray.npy'
    }
}
