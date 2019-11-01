import os
import json
import datetime
import shutil
import traceback
from time import localtime, strftime

from .config import initial_config, change_output_dir
from .core.pdf.write import generate

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
        from keras.backend.tensorflow_backend import set_session

        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allocator_type = 'BFC'
        tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.3
        tfconfig.gpu_options.allow_growth = True
        set_session(tf.Session(config=tfconfig))

        from .core.type.inference import Inference as TypeInference
        from .core.direction.LeNet5_evaluate import Direction
        from .core.macular.main import Macular
        from .core.lesion.lesion_detection import Lesion
        from .core.vessel.vessel import Vessel
        from .core.grading.grading import Grader
        from .core.dr.inference import Inference
        from .core.vesselAnalysis.vessel_analysis import VesselAnalysis
        from .core.iqa.iqa_inference import Inference as IQAInference
        from .core.isa.isa_inference import Inference as ISAInference
        from .core.cupSegmentation.inference import Inference as CupInference

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

            self.general_grader = Grader(CONFIG['general_grading'])
            self.general_grader.initial()

            self.topcon_detector = Inference(CONFIG['topcon_detecting'])
            self.topcon_detector.initial()

            self.topcon_grader = Inference(CONFIG['topcon_grading'])
            self.topcon_grader.initial()

            self.lesion = Lesion(CONFIG['lesion'])
            self.lesion.initial()

            self.iqaInference = IQAInference(CONFIG['iqa'])
            self.iqaInference.initial()

            self.isaInference = ISAInference(CONFIG['isa'])
            self.isaInference.initial()

            self.cupInference = CupInference(CONFIG['cupSegmentation'])
            self.cupInference.initial()

            self.macular = Macular(CONFIG['macular'])
            self.macular.initial()

            self.vesselAnalysis = VesselAnalysis(CONFIG['vesselAnalysis'])
        self.vessel = Vessel(CONFIG['vessel'])

        self.graph = tf.get_default_graph()

    def analysis(self, images, patient_names, report_ids, va=True, ignore_exception=True):
        assert len(images) == len(patient_names) == len(report_ids), 'images, patient_names and report_ids should have same size.'
        reports = []
        exceptions = {}
        statuses = {}
        for idx in range(len(images)):
            # create unique dir
            timeid = strftime(timeid_format, localtime())
            image_path = images[idx]
            patient_name = patient_names[idx]
            report_id = report_ids[idx]
            image_name = os.path.split(image_path)[1]
            image_prefix = os.path.splitext(image_name)[0]
            dir_name = '{}_{}'.format(timeid, image_prefix)
            unique_dir = os.path.join(self.output_path, dir_name)
            assert not os.path.exists(unique_dir)
            os.makedirs(unique_dir)
            change_output_dir(self.CONFIG, unique_dir)
            print(self.CONFIG)
            path_to_report = self.CONFIG['global']['path_to_report']
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

            statuses[image_path] = report['status']
            if not report['status']:
                report['patient-id'] = patient_name
                report['report-id'] = report_id
                report['report_path'] = path_to_report
                generate(report, path_to_report)
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

        # Type detection
        print('{}：正在判断是否为眼底照'.format(image_name))
        type_prob, type_ = self.type_detection.classify(image_path)
        if type_ == 1:
            report['status'] = 1
            return report

        print('{}：正在进行图像质量检测'.format(image_name))
        score, grade = self.isaInference.classify(image_path)
        report['norm-point'] = str(score // 10)
        report['norm-quality'] = grade
        score, grade = self.iqaInference.classify(image_path)
        report['image-point'] = str(score // 10)
        report['image-quality'] = grade

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
        exu_result, hem_result, mic_result, lesion_result = self.lesion.lesion_locate(
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

        lesion_nums, mean_dist, plot_path, output_path = mic_result
        report['microaneurysms'] = '有' if lesion_nums > 0 else '无'
        report['1-microaneurysms'] = str(round(mean_dist, 2)) if mean_dist > 0 else '-'
        report['output_images']['microaneurysms_image'] = output_path
        report['output_images']['microaneurysms_histogram'] = plot_path

        # cup segmentation
        print('{}：正在进行视杯视盘分割'.format(image_name))
        cup_image, cdr, disc_area, cup_area= self.cupInference.run(image_path)
        report['cdr'] = str(round(cdr, 4))
        report['disc_area'] = disc_area
        report['cup_area'] = cup_area
        report['output_images']['optic_image'] = cup_image

        # diabetic retinopathy detection
        print('{}：正在进行DR诊断'.format(image_name))
        if self.CONFIG['global']['photochoose'] == 1:
            prob, class_ = self.topcon_detector.classify(image_path, lesion_result)
            is_dr = (class_ == self.CONFIG['topcon_detecting']['true_class'])
            dr_prob = '{}%'.format(round(prob*100, 2)) if is_dr else '{}%'.format(round((1-prob)*100, 2))
            report['DR_prob'] = dr_prob
            report['dr'] = '有' if is_dr else '无'

            # Staging if is dr
            if is_dr:
                print('{}：正在进行分级'.format(image_name))
                grade_prob, grade = self.topcon_grader.classify(image_path, lesion_result)
                report['stage'] = str(grade)
                report['level1_prob'] = '{}%'.format(
                    round(grade_prob*prob*100, 2))
            else:
                report['stage'] = '0'
                report['level1_prob'] = '{}%'.format(round(prob*100, 2))
        else:
            grade_prob, grade = self.general_grader.grade(image_path)
            report['stage'] = str(grade)
            report['level1_prob'] = '{}%'.format(round(grade_prob*100, 2))
            if grade == 0:
                report['dr'] = '无'
                report['DR_prob'] = '{}%'.format(round(100-grade_prob*100, 2))
            else:
                report['dr'] = '有'
                report['DR_prob'] = '{}%'.format(round(grade_prob*100, 2))

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
    fundusAnalysis.initial(record_dir='./', model_dir='./models')
    report, _, _ = fundusAnalysis.analysis(
        images=['./test2.jpg'],
        patient_names=['yuanzhe'],
        report_ids=['I am ID']
    )
    postpdf = report[0]['report_path']
    print(postpdf)
    import pprint
    pprint.pprint(report[0])
