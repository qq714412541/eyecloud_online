FLAGS = {
    'network': '',
    'transfer_learning': False,
}
network_configs = {
    'Inception-V1': {
        'INPUT_WIDTH': 224,
        'INPUT_HEIGHT': 224,
        'INPUT_DEPTH': 3,
        'INPUT_MEAN': 127.5,
        'INPUT_STD': 127.5,
        'pretrained_model_path': '../models/inception_v1.ckpt'
    },
}
