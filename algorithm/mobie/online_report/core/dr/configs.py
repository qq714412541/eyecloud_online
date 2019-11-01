FLAGS = {
    'network': '',
    'transfer_learning': False,
}
network_configs = {
    'Inception-ResNet-V2': {
        'INPUT_WIDTH': 299,
        'INPUT_HEIGHT': 299,
        'INPUT_DEPTH': 3,
        'pretrained_model_path': '../models/inception_resnet_v2_2016_08_30.ckpt'
    }
}
