import os


def initial_config(record_dir='./', model_dir='./models'):
    return {
        'direction': {
            'model_path': os.path.join(model_dir, 'side_model')
        },
        'lesion': {
            'prefix': {
                'exu': 0,
                'hem': 1,
                "mic": 2
            },
            'model_path': os.path.join(model_dir, 'lesion/3lesion.h5'),
            'output_dir': os.path.join(record_dir, 'lesion/result/'),
            'plot_dir': os.path.join(record_dir, 'lesion/plot/'),
            'threshold': 0.5,
            'color': 'green'
        },
        'macular': {
            'path_od_pb': os.path.join(model_dir, 'macular/frozen_graph_od.pb'),
            'path_fovea_pb': os.path.join(model_dir, 'macular/frozen_graph_fovea.pb'),
            'path_to_images_output': os.path.join(record_dir, 'macular/result/'),
            'width': 64,
            'height': 64,
            'mean_disc_radius': 193.38
        },
        'vessel': {
            'model_architecture_path': os.path.join(model_dir, 'vessel/bnUNet_architecture.json'),
            'model_weight_path': os.path.join(model_dir, 'vessel/bnUNet_best_weights.h5'),
            'model_name': 'bnUNet',
            'dataset_name': 'vessel',
            'save_path': os.path.join(record_dir, 'vessel/result/'),
            'patch_height': 64,
            'patch_width': 64,
            'stride_height': 15,
            'stride_width': 15,
            'seg_num': 1
        },
        "iqa": {
            "model_path": os.path.join(model_dir, 'iqa/'),
            "std": 64.7685,
            "mean": 51.4170,
            "image_size": 299
        },
        "isa": {
            "model_path": os.path.join(model_dir, 'isa/'),
            "std": 67.9021,
            "mean": 68.2146,
            "image_size": 299
        },
        "general_grading": {
            "model_path": os.path.join(model_dir, 'grading/o_O_large.pt'),
            "input_size": 448,
            "device": "cpu"
        },
        'topcon_detecting': {
            'model_path': os.path.join(model_dir, 'dr/inc_res_v2/model-270890'),
            'num_classes': 2,
            'lesion_num': 2,
            "lesion_types": [0, 1],
            'true_class': 0,
            'label2class': {
                '0': 0,
                '1': 1
            },
            'network': 'Inception-ResNet-V2',
            'FLAGS': {
                'network': 'Inception-ResNet-V2',
                'transfer_learning': False
            }
        },
        'type': {
            'model_path': os.path.join(model_dir, 'type/model-8214'),
            'num_classes': 2,
            'true_class': 0,
            'label2class': {
                '0': 0,
                '1': 1
            },
            'network': 'Inception-V1',
            'FLAGS': {
                'network': 'Inception-V1',
                'transfer_learning': False
            }
        },
        'topcon_grading': {
            'model_path': os.path.join(model_dir, 'grading/inc_res_v2/model-10803'),
            'num_classes': 4,
            'lesion_num': 2,
            "lesion_types": [0, 1],
            'label2class': {
                '0': 1,
                '1': 3,
                '2': 2,
                '3': 4
            },
            'network': 'Inception-ResNet-V2',
            'FLAGS': {
                'network': 'Inception-ResNet-V2',
                'transfer_learning': False
            }
        },
        'vesselAnalysis': {
            'path_to_images_output': os.path.join(record_dir, 'vesselAnalysis/result/qua.jpg'),
            'path_to_save_counter_diameter': os.path.join(record_dir, 'vesselAnalysis/result/diameter.jpg'),
            'path_to_save_counter_length': os.path.join(record_dir, 'vesselAnalysis/result/length.jpg'),
            "path_to_save_coordinate": os.path.join(record_dir, 'vesselAnalysis/result/Coordinate.csv')
        },
        'cupSegmentation': {
            "detect_model_path": os.path.join(model_dir, 'cupSegmentation/detector/'),
            "segment_model_path": os.path.join(model_dir, 'cupSegmentation/segmentor/'),
            "save_path": os.path.join(record_dir, 'cupSegmentation/cup/'),
            "std": 76.641,
            "mean": 133.49,
            "segment_size": 256,
            "disc_size": 640,
            "ROI_size": 800
        },
        'report': {
            'vessel_analysis': False,
            'length_normal_mean': 2.13,
            'density_normal_mean': '14.98%',
            'diameter_normal_mean': 0.19,
            'a-length_compare': '2.13±1.029',
            'a-density_compare': '14.98%±4.23%',
            'a-diameter_compare': '0.19±0.0523',
            'a-normal_length_histogram': os.path.join(record_dir, 'template/vessel_normal/length_normal.jpg'),
            'a-normal_diameter_histogram': os.path.join(record_dir, 'template/vessel_normal/diameter_normal.jpg'),
            'watermark1': os.path.join(record_dir, 'temp/watermark1.pdf'),
            'watermark2': os.path.join(record_dir, 'temp/watermark2.pdf'),
            'watermark3': os.path.join(record_dir, 'temp/watermark3.pdf'),
            'watermark4': os.path.join(record_dir, 'temp/watermark4.pdf'),
            'template1': os.path.join(record_dir, 'template/template/template1.pdf'),
            'template2': os.path.join(record_dir, 'template/template/template2.pdf'),
            'template3': os.path.join(record_dir, 'template/template/template3.pdf'),
            'template4': os.path.join(record_dir, 'template/template/template4.pdf'),
        },
        'global': {
            'temp_file_dir': os.path.join(record_dir, 'temp/'),
            'output_path': os.path.join(record_dir, 'results'),
            'path_to_report': os.path.join(record_dir, 'report.pdf'),
            'photochoose': 1
        }
    }


# A stupid method
def change_output_dir(CONFIG, record_dir):
    CONFIG['global']['path_to_report'] = os.path.join(record_dir, 'report.pdf')
    CONFIG['lesion']['output_dir'] = makdirs(os.path.join(record_dir, 'lesion/'))
    CONFIG['lesion']['plot_dir'] = makdirs(os.path.join(record_dir, 'lesion/'))
    CONFIG['macular']['path_to_images_output'] = makdirs(os.path.join(record_dir, 'macular/'))
    CONFIG['vessel']['save_path'] = makdirs(os.path.join(record_dir, 'vessel/'))
    CONFIG['vesselAnalysis']['path_to_images_output'] = makdirs(os.path.join(record_dir, 'vesselAnalysis/qua.jpg'))
    CONFIG['vesselAnalysis']['path_to_save_counter_diameter'] = makdirs(os.path.join(record_dir, 'vesselAnalysis/diameter.jpg'))
    CONFIG['vesselAnalysis']['path_to_save_counter_length'] = makdirs(os.path.join(record_dir, 'vesselAnalysis/length.jpg'))
    CONFIG['cupSegmentation']['save_path'] = makdirs(os.path.join(record_dir, 'cup_segmentation/'))


def makdirs(path):
    dir_ = os.path.split(path)[0]
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    return path
