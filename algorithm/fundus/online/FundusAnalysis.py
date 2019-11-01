import os
import json
import datetime
import shutil
import traceback
from time import localtime, strftime

from ..config import initial_config, change_output_dir

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
exts = ['.jpg', '.JPG', '.jpeg', '.JPEG']
convert_exts = ['.png', '.PNG']
timeid_format = r'%Y%m%d%H%M%S'
status_code = {
    0: 'Success',
    1: 'Not Fundus Images',
    2: 'Optical Disc Unclear'
}


class FundusAnalysis:
    def initial(self, record_dir, model_dir):
        import tensorflow as tf

        from ..core.type.inference import Inference as TypeInference
        from ..core.direction.LeNet5_evaluate import Direction
        from ..core.macular.main import Macular
        from ..core.lesion.lesion_detection import Lesion
        from ..core.vessel.vessel import Vessel
        from ..core.dr.inference import Inference
        from ..core.vesselAnalysis.vessel_analysis import VesselAnalysis

        CONFIG = initial_config(record_dir=record_dir, model_dir=model_dir)
        self.CONFIG = CONFIG

        # initial
        self.temp_dir = CONFIG['global']['temp_file_dir']
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        self.output_path = CONFIG['global']['output_path']
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            self.type_detection = TypeInference(CONFIG['type'])
            self.type_detection.initial()
            self.direction = Direction(CONFIG['direction'])
            self.dr_detection = Inference(CONFIG['dr'])
            self.dr_detection.initial()
            self.grader = Inference(CONFIG['grading'])
            self.grader.initial()
            self.lesion = Lesion(CONFIG['lesion'])
            self.lesion.initial()
            self.macular = Macular(CONFIG['macular'])
            self.macular.initial()
            self.vesselAnalysis = VesselAnalysis(CONFIG['vesselAnalysis'])
        self.vessel = Vessel(CONFIG['vessel'])

        self.graph = tf.get_default_graph()

    def analysis(self, images, va=True, ignore_exception=True):
        reports = []
        exceptions = {}
        statuses = {}
        for idx in range(len(images)):
            # create unique dir
            timeid = strftime(timeid_format, localtime())
            image_path = images[idx]
            image_name = os.path.split(image_path)[1]
            image_prefix = os.path.splitext(image_name)[0]
            unique_dir = os.path.join(self.output_path, '{}_{}'.format(timeid, image_prefix))
            assert not os.path.exists(unique_dir)
            os.makedirs(unique_dir)
            change_output_dir(self.CONFIG, unique_dir)
            try:
                with self.graph.as_default():
                    report = self._analysis_one_shot(
                        va,
                        image_path,
                        image_name,
                    )
                    reports.append(report)
            except Exception as e:
                if ignore_exception:
                    exceptions[image_path] = traceback.format_exc()
                    print('=============={}: exception raise=============='.format(image_path))
                    print(exceptions[image_path])
                    print('=============={}: exception raise=============='.format(image_path))
                    continue
                else:
                    raise e

            if report['status']:
                statuses[image_path] = report['status']
            print('{}：分析完成'.format(image_name))

        return reports, exceptions, statuses

    def _analysis_one_shot(self, va, image_path, image_name):
        # Initial callback message
        report = self.CONFIG['report'].copy()
        report['status'] = 0
        report['output_images'] = {}
        report['patient-image'] = image_path

        image_full_name = os.path.split(image_path)[-1]
        f_name, ext = os.path.splitext(image_full_name)
        if ext in convert_exts:
            import cv2 as cv
            origin_img = cv.imread(image_path)
            temp_file = os.path.join(self.temp_dir, f_name + '.jpg')
            cv.imwrite(temp_file, origin_img)
        else:
            temp_file = os.path.join(self.temp_dir, f_name + ext)
            shutil.copy(image_path, temp_file)

        image_path = temp_file
        report['report-time'] = strftime(timeid_format, localtime())
        jobs_num = 8 if va else 6

        # Type detection
        print('{}：正在判断是否为眼底照'.format(image_name))
        type_prob, type_ = self.type_detection.classify(image_path)
        if type_ == 1:
            report['status'] = 1
            callback_bar(100)
            return report

        # eye direction detection
        print('{}：正在进行眼别识别'.format(image_name))
        direction, _ = self.direction.evalutate(image_path)
        report['eye'] = direction

        # macular and disc detection
        print('{}：正在进行黄斑及视盘识别'.format(image_name))
        result = self.macular.detection(image_path)
        macular_path, spot, disc_radius, theta, disc_spot, width, flag1, flag2 = result
        x, y = spot
        report['output_images']['macular_image'] = macular_path
        if disc_radius > 0:
            report['disc_diameter'] = str(round(disc_radius, 2))
            no_disc = False
        else:
            report['disc_diameter'] = '视盘半径模糊'
            disc_radius = self.CONFIG['macular']['mean_disc_radius']
            no_disc = True
        report['distance'] = '超过' if flag1 else '不超过'
        report['macular_center_coordinate'] = '({}, {})'.format(x, y)
        report['angle'] = str(round(theta, 2))
        report['standard'] = '符合' if flag1 and flag2 else '不符合'

        # lesion detection
        print('{}：正在进行病变区域划分'.format(image_name))
        exu_result, hem_result, lesion_result = self.lesion.lesion_locate(
            image_path, spot, disc_radius)
        lesion_nums, mean_dist, plot_path, output_path = exu_result
        report['exudation'] = '有' if lesion_nums > 0 else '无'
        report['1-exudation'] = str(round(mean_dist, 2)) if mean_dist > 0 else '-'
        report['output_images']['exudation_image'] = output_path
        report['output_images']['exudation_histogram'] = plot_path

        lesion_nums, mean_dist, plot_path, output_path = hem_result
        report['bleed'] = '有' if lesion_nums > 0 else '无'
        report['1-bleed'] = str(round(mean_dist, 2)) if mean_dist > 0 else '-'
        report['output_images']['bleed_image'] = output_path
        report['output_images']['bleed_histogram'] = plot_path

        # diabetic retinopathy detection
        print('{}：正在进行DR诊断'.format(image_name))
        prob, class_ = self.dr_detection.classify(image_path, lesion_result)
        report['DR_prob'] = '{}%'.format(round(prob*100, 2))
        is_dr = (class_ == self.CONFIG['dr']['true_class'])
        report['dr'] = '有' if is_dr else '无'

        # Staging if is dr
        if is_dr:
            print('{}：正在进行分级'.format(image_name))
            grade_prob, grade = self.grader.classify(image_path, lesion_result)
            report['stage'] = str(grade)
            report['level1_prob'] = '{}%'.format(
                round(grade_prob*prob*100, 2))
        else:
            report['stage'] = '0'
            report['level1_prob'] = report['DR_prob']

        # vessel analysis
        if va:
            if no_disc:
                report['status'] = 2
                return report
            else:
                report['vessel_analysis'] = True

            # vessel segmentation
            print('{}：正在进行血管分割'.format(image_name))
            image_bw_path = self.vessel.predict(image_path)
            report['output_images']['retinal_vessel_image'] = image_bw_path

            # vessel data computing
            print('{}：正在进行血管分析'.format(image_name))
            stat, path_qua, path_diameter, path_length = self.vesselAnalysis.analysis(
                image_path, image_bw_path, disc_spot, width)
            report['output_images']['quadrant_segmentation_image'] = path_qua
            report['output_images']['a-patient_length_histogram'] = path_length
            report['output_images']['a-patient_diameter_histogram'] = path_diameter

            report['2-length'] = str(round(stat['mean_length'], 2))
            var = round(stat['mean_length'] - report['length_normal_mean'], 2)
            report['2-length_compare'] = '+{}'.format(
                var) if var > 0 else '{}'.format(var)
            report['2-diameter'] = str(round(stat['mean_diameter'], 2))
            var = round(stat['mean_diameter'] -
                        report['diameter_normal_mean'], 2)
            report['2-diameter_compare'] = '+{}'.format(
                var) if var > 0 else '{}'.format(var)
            report['2-density'] = stat['density']
            var = round(float(stat['density'][:-1]) -
                        float(report['density_normal_mean'][:-1]), 2)
            report['2-density_compare'] = '+{}%'.format(
                var) if var > 0 else '{}%'.format(var)

            report['a-length'] = '{}±{}'.format(
                round(stat['mean_length'], 2), round(stat['std_length'], 4))
            report['a-density'] = stat['density']
            report['vessel-point'] = str(round(stat['iqa'], 1))
            report['vessel-quality'] = stat['quality']
            report['a-diameter'] = '{}±{}'.format(
                round(stat['mean_diameter'], 2), round(stat['std_diameter'], 4))

        return report


if __name__ == "__main__":
    fundusAnalysis = FundusAnalysis()
    fundusAnalysis.initial(record_dir='./', model_dir='/home/algorithms/online/models')
    report, _, _ = fundusAnalysis.analysis(
        ['./test.JPG']
    )
    import pprint
    pprint.pprint(report[0])
