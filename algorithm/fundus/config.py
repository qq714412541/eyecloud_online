import os


def initial_config(record_dir='./', model_dir='./models'):
    return {
        'direction': {
            'model_path': os.path.join(model_dir, 'side_model')
        },
        'lesion': {
            'prefix': {
                'exu': 0,
                'hem': 1
            },
            'model_path': os.path.join(model_dir, 'lesion/combined50.h5'),
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
        'dr': {
            'model_path': os.path.join(model_dir, 'dr/inc_res_v2/model-270890'),
            'num_classes': 2,
            'lesion_num': 2,
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
            'model_path': os.path.join(model_dir, 'type/model-3219'),
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
        'grading': {
            'model_path': os.path.join(model_dir, 'grading/inc_res_v2/model-10803'),
            'num_classes': 4,
            'lesion_num': 2,
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
            'path_to_save_counter_length': os.path.join(record_dir, 'vesselAnalysis/result/length.jpg')
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
            'a-normal_diameter_histogram': os.path.join(record_dir, 'template/vessel_normal/diameter_normal.jpg')
        },
        'pdf': {
            'watermark1': './temp/watermark1.pdf',
            'watermark2': './temp/watermark2.pdf',
            'watermark3': './temp/watermark3.pdf',
            'watermark4': './temp/watermark4.pdf',
            'template1': './template/template/template1.pdf',
            'template2': './template/template/template2.pdf',
            'template3': './template/template/template3.pdf',
            'template4': './template/template/template4.pdf',
        },
        'global': {
            'temp_file_dir': './temp/',
            'output_path': './results',
            'path_to_report': os.path.join(record_dir, 'report.pdf')
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


def makdirs(path):
    dir_ = os.path.split(path)[0]
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    return path
