# -*- coding: utf-8 -*-
import datetime

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from PyPDF2 import PdfFileWriter, PdfFileReader

def generate(config, save_path):
    # support chinese
    font_chinese = 'STSong-Light' # from Adobe's Asian Language Packs
    pdfmetrics.registerFont(UnicodeCIDFont(font_chinese))

    # Create the watermark from an image
    w = 595.27
    h = 841.89
    # page1
    c1 = canvas.Canvas(config['watermark1'], (w, h))
    c1.setFont(font_chinese, size=11)
    # get current time
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime('%H:%M')
    # Add content
    c1.drawString(137, h-134, config['patient-id'])
    c1.drawString(160, h-159, config['report-id'])
    c1.drawString(147, h-184, date)
    c1.drawString(147, h-209, time)
    c1.drawString(122, h-234, config['eye'])
    # image quality content
    offset = 0
    if config['vessel_analysis']:
        c1.drawCentredString(304, h-282, config['vessel-point'])
        c1.drawCentredString(326, h-282, config['vessel-quality'])
        c1.drawCentredString(448, h-282, config['norm-point'])
        c1.drawCentredString(470, h-282, config['norm-quality'])
        c1.drawCentredString(182, h-298, config['image-quality'])
        c1.drawCentredString(205, h-298, config['image-point'])
        c1.drawCentredString(400, h-298, config['distance'])
        c1.drawCentredString(310, h-316, config['angle'])
        c1.drawCentredString(369, h-316, config['standard'])
    else:
        c1.drawCentredString(326, h-282, config['norm-point'])
        c1.drawCentredString(348, h-282, config['norm-quality'])
        c1.drawCentredString(448, h-282, config['image-quality'])
        c1.drawCentredString(470, h-282, config['image-point'])
        c1.drawCentredString(257, h-298, config['distance'])
        c1.drawCentredString(169, h-316, config['angle'])
        c1.drawCentredString(225, h-316, config['standard'])
    # diabetic retinopathy
    c1.drawCentredString(215, h-339-offset, config['dr'])
    if config['dr'] == u'有':
        c1.drawCentredString(262, h-340-offset, config['stage'])
        c1.drawCentredString(391, h-340-offset, config['bleed'])
        c1.drawCentredString(151, h-357-offset, config['1-bleed'])
        c1.drawCentredString(221, h-356-offset, config['exudation'])
        c1.drawCentredString(378, h-357-offset, config['1-exudation'])
    if config['vessel_analysis']:
        if config['dr'] == u'有':
            offset += 18
        c1.drawCentredString(250, h-364-offset, config['2-length'])
        c1.drawCentredString(409, h-364-offset, config['2-length_compare'])
        c1.drawCentredString(124, h-381-offset, config['2-density'])
        c1.drawCentredString(260, h-381-offset, config['2-density_compare'])
        c1.drawCentredString(378, h-381-offset, config['2-diameter'])
        c1.drawCentredString(151, h-397-offset, config['2-diameter_compare'])
    c1.drawImage(config['patient-image'], 153, h-658, width=2880//10, height=2136//10)
    c1.save()

    # page2
    c2 = canvas.Canvas(config['watermark2'], (w, h))
    c2.setFont(font_chinese, size=11)
    # Add content
    c2.drawImage(config['output_images']['macular_image'], 100, h-245, width=2880//18, height=2136//18)

    c2.drawCentredString(447, h-141, config['DR_prob'])
    c2.drawCentredString(310, h-176, config['stage'])
    c2.drawCentredString(447, h-176, config['level1_prob'])
    c2.drawCentredString(447, h-211, config['disc_diameter'])
    c2.drawCentredString(447, h-241, config['macular_center_coordinate'])

    c2.drawImage(config['output_images']['optic_image'], 120, h - 431, width=126, height=126)
    c2.drawCentredString(447, h - 322, config['disc_area'])
    c2.drawCentredString(447, h - 357, config['cup_area'])
    c2.drawCentredString(447, h - 407, config['cdr'])
    c2.save()


    # page3
    if config['dr'] == u'有':
        c3 = canvas.Canvas(config['watermark3'], (w, h))
        c3.setFont(font_chinese, size=11)
        c3.drawImage(config['output_images']['microaneurysms_image'], 113, h-341, width=2880//18, height=2136//18)
        c3.drawImage(config['output_images']['microaneurysms_histogram'], 307, h-351, width=187, height=140)
        c3.drawImage(config['output_images']['bleed_image'], 113, h-522, width=2880//18, height=2136//18)
        c3.drawImage(config['output_images']['bleed_histogram'], 307, h-532, width=187, height=140)
        c3.drawImage(config['output_images']['exudation_image'], 113, h-703, width=2880//18, height=2136//18)
        c3.drawImage(config['output_images']['exudation_histogram'], 307, h-713, width=187, height=140)
        c3.save()

    if config['vessel_analysis']:
        # page4
        c4 = canvas.Canvas(config['watermark4'], (w, h))
        c4.setFont(font_chinese, size=11)
        # Add content
        c4.drawImage(config['output_images']['retinal_vessel_image'], 113, h-273, width=2880//18, height=2136//18)
        c4.drawImage(config['output_images']['quadrant_segmentation_image'], 320, h-273, width=2880//18, height=2136//18)
        c4.drawCentredString(274, h-447, config['a-density'])
        c4.drawCentredString(428, h-447, config['a-density_compare'])
        c4.drawCentredString(274, h-464, config['a-length'])
        c4.drawCentredString(428, h-464, config['a-length_compare'])
        c4.drawImage(config['output_images']['a-patient_length_histogram'], 198, h-588, width=150, height=116)
        c4.drawImage(config['a-normal_length_histogram'], 354, h-588, width=150, height=116)
        c4.drawCentredString(274, h-604, config['a-diameter'])
        c4.drawCentredString(428, h-604, config['a-diameter_compare'])
        c4.drawImage(config['output_images']['a-patient_diameter_histogram'], 198, h-729, width=150, height=116)
        c4.drawImage(config['a-normal_diameter_histogram'], 354, h-729, width=150, height=116)
        c4.save()
        '''
        # page4
        c4 = canvas.Canvas('./backup/watermark4.pdf', (w, h))
        c4.setFont(font_chinese, size=11)
        # Add content
        c4.drawCentredString(254, h-125, config['b-density'])
        c4.drawCentredString(428, h-125, config['b-density_compare']) 
        c4.drawCentredString(254, h-142, config['b-length'])
        c4.drawCentredString(428, h-142, config['b-length_compare'])
        c4.drawImage(config['b-patient_length_histogram'], 178, h-253, width=152, height=106)
        c4.drawImage(config['b-patient_length_histogram'], 349, h-253, width=152, height=106)
        c4.drawCentredString(254, h-267, config['b-diameter'])
        c4.drawCentredString(428, h-267, config['b-diameter_compare'])
        c4.drawImage(config['b-patient_diameter_histogram'], 177, h-379, width=154, height=108)
        c4.drawImage(config['b-patient_diameter_histogram'], 346, h-379, width=154, height=108)
        c4.drawCentredString(254, h-459, config['c-density'])
        c4.drawCentredString(428, h-459, config['c-density_compare']) 
        c4.drawCentredString(254, h-476, config['c-length'])
        c4.drawCentredString(428, h-476, config['c-length_compare'])
        c4.drawImage(config['c-patient_length_histogram'], 178, h-587, width=152, height=106)
        c4.drawImage(config['c-patient_length_histogram'], 349, h-587, width=152, height=106)
        c4.drawCentredString(254, h-601, config['c-diameter'])
        c4.drawCentredString(428, h-601, config['c-diameter_compare'])
        c4.drawImage(config['c-patient_diameter_histogram'], 177, h-713, width=154, height=106)
        c4.drawImage(config['c-patient_diameter_histogram'], 346, h-713, width=154, height=106)
        c4.save()
        '''
    # Get the watermark file you just created
    watermark1 = PdfFileReader(open(config['watermark1'], "rb"))
    watermark2 = PdfFileReader(open(config['watermark2'], "rb"))

    if config['dr'] == u'有':
        watermark3 = PdfFileReader(open(config['watermark3'], "rb"))
    if config['vessel_analysis']:
        watermark4 = PdfFileReader(open(config['watermark4'], "rb"))
        #watermark4 = PdfFileReader(open("./backup/watermark4.pdf", "rb"))
        if config['dr'] != u'有':
            watermark3 = watermark4
    # Get our files ready
    output_file = PdfFileWriter()

    # Number of pages in input document
    page_count = 2
    if config['vessel_analysis']:
        if config['dr'] == u'有':
            input_file = PdfFileReader(open(config['template1'], "rb"))
            page_count += 1
        else:
            input_file = PdfFileReader(open(config['template4'], "rb"))
        page_count += 1
    else:
        if config['dr'] == u'有':
            input_file = PdfFileReader(open(config['template2'], "rb"))
            page_count += 1
        else:
            input_file = PdfFileReader(open(config['template3'], "rb"))

    # Go through all the input file pages to add a watermark to them
    for page_number in range(page_count):
        print("Watermarking page {} of {}".format(page_number, page_count))
        # merge the watermark with the page
        input_page = input_file.getPage(page_number)
        if page_number == 0:
            input_page.mergePage(watermark1.getPage(0))
        elif page_number == 1:
            input_page.mergePage(watermark2.getPage(0))
        elif page_number == 2:
            input_page.mergePage(watermark3.getPage(0))
        else:
            input_page.mergePage(watermark4.getPage(0))
        # add page from input file to output document
        output_file.addPage(input_page)

    # finally, write "output" to document-output.pdf
    with open(save_path, 'wb') as outputStream:
        output_file.write(outputStream)

if __name__ == '__main__':
    config = dict()
    config['vessel_analysis'] = True

    config['patient-id'] = 'yijin'
    config['report-id'] = 'yijin-001'
    config['eye'] = u'左眼'
    config['norm-point'] = '7'
    config['norm-quality'] = u'优'
    config['image-point'] = '8'
    config['image-quality'] = u'优'
    config['vessel-point'] = '6'
    config['vessel-quality'] = u'良'
    config['distance'] = u'不超过'
    config['angle'] = '12.2'
    config['standard'] = u'符合'
    config['dr'] = u'无'
    config['stage'] = '1'
    config['bleed'] = u'有'
    config['1-bleed'] = '10'
    config['exudation'] = '有'
    config['1-exudation'] = '10'
    config['2-length'] = '1.26'
    config['2-length_compare'] = '-0.87'
    config['2-density'] = '13.07%'
    config['2-density_compare'] = '-1.91%'
    config['2-diameter'] = '0.13'
    config['2-diameter_compare'] = '-0.06'
    config['origin_image'] = 'test3.jpeg'
    config['macular_image'] = 'test3.jpeg'
    config['optic_image'] = 'result.png'
    config['DR_prob'] = '95.5%'
    config['level1_prob'] = '92.9%'
    config['disc_diameter'] = '15'
    config['macular_center_coordinate'] = '(1440, 1052)'
    config['microaneurysms_image'] = 'test3.jpeg'
    config['microaneurysms_histogram'] = 'test3.jpeg'
    config['bleed_image'] = 'test3.jpeg'
    config['bleed_histogram'] = 'test3.jpeg'
    config['exudation_image'] = 'test3.jpeg'
    config['exudation_histogram'] = 'test3.jpeg'
    config['retinal_vessel_image'] = 'test3.jpeg'
    config['quadrant_segmentation_image'] = 'test3.jpeg'
    config['a-length'] = '100'
    config['a-length_compare'] = '90/0.5'
    config['a-density'] = '20'
    config['a-density_compare'] = '22/0.4'
    config['a-diameter'] = '50'
    config['a-diameter_compare'] = '45/0.3'
    config['a-patient_length_histogram'] = 'test3.jpeg'
    config['a-normal_length_histogram'] = 'test3.jpeg'
    config['a-patient_diameter_histogram'] = 'test3.jpeg'
    config['a-normal_diameter_histogram'] = 'test3.jpeg'
    config['b-length'] = '100'
    config['b-length_compare'] = '90/0.2'
    config['b-density'] = '20'
    config['b-density_compare'] = '22/0.15'
    config['b-diameter'] = '50'
    config['b-diameter_compare'] = '45/0.23'
    config['b-patient_length_histogram'] = 'test3.jpeg'
    config['b-normal_length_histogram'] = 'test3.jpeg'
    config['b-patient_diameter_histogram'] = 'test3.jpeg'
    config['b-normal_diameter_histogram'] = 'test3.jpeg'
    config['c-length'] = '100'
    config['c-length_compare'] = '90/0.5'
    config['c-density'] = '20'
    config['c-density_compare'] = '22/0.15'
    config['c-diameter'] = '50'
    config['c-diameter_compare'] = '48/0.26'
    config['c-patient_length_histogram'] = 'test3.jpeg'
    config['c-normal_length_histogram'] = 'test3.jpeg'
    config['c-patient_diameter_histogram'] = './test3.jpeg'
    config['c-normal_diameter_histogram'] = './test3.jpeg'

    config['watermark1'] = './watermark1.pdf'
    config['watermark2'] = './watermark2.pdf'
    config['watermark3'] = './watermark3.pdf'
    config['watermark4'] = './watermark4.pdf'

    config['template1'] = './test_v5.pdf'
    config['template2'] = './test_v5.pdf'
    config['template3'] = './test_v5.pdf'
    config['template4'] = './test_v5.pdf'




    config['disc_area'] = '30156.57'
    config['cup_area'] = '12593.30'
    config['cdr'] = '0.6736'
    generate(config, './test.pdf')




