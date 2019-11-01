### FundusAnalysis.py

```python
class FundusAnalysis:
	def initial(self, record_dir, model_dir):
        """
        Initial models and config. A time-consuming process.
        Args:
            record_dir: str. The path to store output results.
            model_dir: str. The path to load models.
        """
    
    def analysis(self, images, va=True, ignore_exception):
        """
        Analyse batch of images.
        Args:
            images: list(str). The path of images to analysis.
            va: bool. If do the vessel analysis.
            ignore_exception: bool. If continue the batch process when any exception 					raises. True for application and False for testing.
        Returns:
            reports: list(dict). List of report. The detail of key and value is shown 					below.
            exceptions: dict. Key is image_path and value is exception traceback if any 				exception raises.
            statuses: dict. Key is image_path and the value is status code.
                Status Code:
                    0: Success
                    1: Not Fundus Images
                    2: Optical Disc Unclear
        """
    
    def _analysis_one_shot(self, va, image_path, image_name):
        """
        Private function. Should not be called.
        Args:
        	va: bool. If do the vessel analysis.
        	image_path: str. The path of image.
        	image_name: str. The name of image shown in the log.
        Returns:
        	report: dict. The result of analysis.
        """
```



### Report

- 无特别指出数据类型的value都为str
- Keys:
  - report_path: 报告生成后保存的路径
  - report_id: 报告ID，目前使用目录的名称
  - patient_id: 病人ID，目前使用输入图片的名称
  - 1-bleed：出血处距离黄斑中心平均ORD
  - 1-exudation：渗出处距离黄斑中心平均ORD
  - 2-density：血管密度
  - 2-density_compare：与常人血管密度差值
  - 2-diameter：血管直径
  - 2-diameter_compare：与常人血管直径差值
  - 2-length：血管长度
  - 2-length_compare：与常人血管长度差值
  - DR_prob：得糖尿病眼病的概率，即置信度
  - a-density：同"2-density"
  - a-density_compare：常人血管密度，格式为{14.98%±4.23%}
  - a-diameter：血管直径，相较于"2-diameter"，在均值后增加了标准差。格式为{0.13±0.0415}
  - a-diameter_compare：常人血管直径，格式同"a-diameter"
  - a-length：血管长度，相较于“2-length”，在均值后增加了标准差。格式同"a-diameter"
  - a-length_compare：常人血管长度，格式同"a-diameter"
  - a-normal_diameter_histogram：常人血管直径分布直方图路径
  - a-normal_length_hisrogram：常人血管长度分布直方图路径
  - angle：黄斑中心和视盘中心的夹角，用于判断拍摄质量（0°-180°）
  - bleed：是否有出血现象（有或无）
  - density_normal_mean：常人血管密度均值
  - diameter_normal_mean：**float.** 常人血管直径均值
  - disc_diameter：视盘直径
  - distance：黄斑中心与视盘中心夹角是否超过2个视盘半径距离，用于判断拍摄质量（超过或不超过）
  - dr：是否患有糖尿病眼病（有或无）
  - exudation：是否有渗出现象（有或无）
  - eye：拍摄眼睛方向（左眼或右眼）
  - length_normal_mean：**float.** 常人血管长度均值
  - level1_prob：糖尿病眼病患病程度分级置信度
  - macular_center_coordinate：黄斑中心像素坐标
  - output_images
    - a-patient_diameter_histogram：血管直径分布直方图路径
    - a-patient_length_histogram：血管长度分布直方图路径
    - bleed_histogram：出血处与黄斑中心距离分布直方图路径
    - bleed_image：出血位置检测结果图路径
    - exudation_histogram：渗出处与黄斑中心距离分布直方图路径
    - exudation_image：渗出位置检测结果图路径
    - macular_image：黄斑中心定位和视盘检测结果图
    - quadrant_segmentation_image：血管分析ROI区域划分图
    - retinal_vessel_image：血管分割结果图
  - patient-image：输入图像的路径
  - report-time：分析时间
  - stage：糖尿病眼病患病程度分级结果（0-4）
  - standard：输入眼底照是否符合标准（符合或不符合）
  - status：**int.** report状态码
    - 0：成功
    - 1：非眼底照
    - 2：视盘模糊，无法进行血管分析
  - vessel-point：眼底照血管质量评估分数
  - vessel-quality：眼底照血管质量评估定级
  - vessel_analysis：**bool.** 是否有进行血管分析