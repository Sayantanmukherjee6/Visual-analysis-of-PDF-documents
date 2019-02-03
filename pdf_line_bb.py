#!/usr/bin/env python
# coding: utf-8
# Author: Debraj Ghosh, Sayantan Mukherjee  

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb
from PyPDF2 import PdfFileReader
import cv2
import subprocess
from pdf2jpg import pdf2jpg
import pytesseract
import re
from pythonRLSA import rlsa


flg = 0
no_of_rows = 0
row_word_y = []
row_word_height = []


def get_page_layout(pdf_file_path, page_num):
	"""
	Function to return page width and height.
	"""
	wb_pdf = PdfFileReader(open(pdf_file_path, 'rb'))
	page_layout = wb_pdf.getPage(page_num)['/MediaBox']
	if '/Rotate' in wb_pdf.getPage(page_num) and wb_pdf.getPage(page_num)['/Rotate'] == 90:#identify whether the page is portrait or landscape 
		page_width = float(page_layout[3])
		page_height = float(page_layout[2])
	else:
		page_width = float(page_layout[2])
		page_height = float(page_layout[3])
	return page_width,page_height

def get_page_image_from_pdf(page_num, image_file_name, pdf_file_path):
	"""
	Converting a pdf page into an Image for processing
	"""
	inputpath = pdf_file_path
	outputpath = "images/"
	# To convert single page
	pdf2jpg.convert_pdf2jpg(inputpath, outputpath, pages=str(page_num))
	im = cv2.imread(
		"images/"+pdf_file_path+"/"+str(page_num)+"_"+pdf_file_path+".jpg",
		 cv2.IMREAD_COLOR
		)
	cv2.imwrite(image_file_name, im)
	return cv2.imread(image_file_name, 0)

def get_thresh_bin_image(image):
	"""
	Function to create binary image
	"""
	(thresh, image_binary) = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	return thresh,image_binary

def get_rlsa_output(image):
	"""
	Function to return rlsa output after running rlsa on the binary iamge
	"""
	image_rlsa_horizontal = rlsa.rlsa(image, 1, 0, 50)# performing rlsa algorithm on the binary image 
	image_rlsa_horizontal_inverted = cv2.bitwise_not(image_rlsa_horizontal)# inverting the image 
	return image_rlsa_horizontal_inverted

def get_cca_output(image):
	"""
	Function to return statitics of the rlsa based on connected component analysis
	"""
	n_comp, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
	return n_comp,labels,stats,centroids

def get_block_stats(stats, centroids):
	"""
	Convert stats into a dataframe.
	"""
	stats_columns = ["left", "top", "width", "height", "area"]
	block_stats = pd.DataFrame(stats, columns=stats_columns)
	block_stats['centroid_x'], block_stats['centroid_y'] = centroids[:, 0], centroids[:, 1]
	# Ignore the label 0 since it is the background
	block_stats.drop(0, inplace=True)
	return block_stats

def get_text_data_line(row, page_num, pdf_file_path, horizontal_ratio, vertical_ratio, wb_page_image):
	"""
	Bounding box function for a single line in pdf
	"""
	x = (row['left'] * horizontal_ratio)
	y = (row['top'] * vertical_ratio)
	width = (row['width'] * horizontal_ratio) + 5
	hieght = (row['height'] * vertical_ratio) + 5
	wb_page_image1 = wb_page_image
	global no_of_rows, row_word_y, row_word_height, flg
	if (int(row['height']) > 15 and int(row['height']) < 100):
		if flg ==1: 
			if int(row['top']) <= (sum(row_word_y)/len(row_word_y) + sum(row_word_height)/len(row_word_height)):
				row_word_y.append(int(row['top']))
				row_word_height.append(int(row['height']))
				cv2.waitKey(0)
				lineimg = wb_page_image[int(row['top']-4):int(row['top']) + int(row['height']+8), int(row['left'])-10: int(row['left']) + int(row['width'])+10]#cropping the current line from the page image 
				get_text(lineimg) 
				cv2.rectangle(wb_page_image, (int(row['left']), int(row['top'])), (int(row['left']) + int(row['width']), int(row['top']) + int(row['height'])), (0, 0, 255), 5)#drawing the bounding box on the current line 
				cv2.imshow('table_words', cv2.resize(wb_page_image, (528, 713)))
			else:
				no_of_rows = no_of_rows + 1         
				flg = 0
				row_word_y, row_word_height = [], []
		if flg==0:
			row_word_y.append(int(row['top']))
			row_word_height.append(int(row['height']))
			flg = 1 
			cv2.waitKey(0)
			lineimg = wb_page_image[int(row['top']-4):int(row['top']) + int(row['height']+8), int(row['left']-10): int(row['left']) + int(row['width']+20)]#cropping the line image 
			get_text(lineimg)
			cv2.rectangle(wb_page_image, (int(row['left']), int(row['top'])), (int(row['left']) + int(row['width']), int(row['top']) + int(row['height'])), (0, 0, 255), 5)#bounding box on line 
			cv2.imshow('table_words', cv2.resize(wb_page_image, (528, 713)))

def get_text(image):
	"""
	Print text present inside line bounding box in console
	"""
	config = ('-l eng --oem 1 --psm 3')
	text = pytesseract.image_to_string(image, config=config)
	text_list = text.splitlines()
	print(text_list)	


if __name__=="__main__":
	"""
	Main function to process the pdf file
	"""
	pdf_file_path = "test.pdf"
	page_num = 0

	page_width,page_height=get_page_layout(pdf_file_path,page_num)
	wb_page_image = get_page_image_from_pdf(page_num, 'kn_sample_image.jpg', pdf_file_path)
	cv2.imshow('original', cv2.resize(wb_page_image, (528, 713)))
	cv2.waitKey(0)

	image_height, image_width = wb_page_image.shape
	horizontal_ratio = page_width / image_width
	vertical_ratio = page_height / image_height

	(thresh, image_binary) = get_thresh_bin_image(wb_page_image)
	rlsa_inverted_image=get_rlsa_output(image_binary)
	cv2.imwrite('rlsa.jpg', rlsa_inverted_image)
	cv2.imshow('rlsa', cv2.resize(rlsa_inverted_image, (528, 713)))
	cv2.waitKey(0)

	n_comp, labels, stats, centroids = get_cca_output(rlsa_inverted_image)

	block_stats = get_block_stats(stats, centroids)
	block_stats['right'] = block_stats.left + block_stats.width
	block_stats['bottom'] = block_stats.top + block_stats.height

	text_columns = ['text', 'text_length', 'comma_separated_numbers_present', 'is_text', 'number']
	block_stats.apply(get_text_data_line, axis=1, args=[page_num, pdf_file_path, horizontal_ratio, vertical_ratio, wb_page_image])