#!/usr/bin/env python
# coding: utf-8

# In[104]:


from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
# Copied from Classification
import os, re
from underthesea import word_tokenize
from langdetect import detect
import unicodedata
from tkinter import *
import ntpath
import pandas as pd

# Copied from NaiveBayes
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from fractions import Fraction

import scraper
from time import sleep

import DatasetGenerator as segmentation
import random
# In[110]:


# Class: Classification
class Classification():
	
	def __init__(self):
		self.list_stopwords = []
		self.words = []
		self.data = []
		list_category = ['doi-song','du-lich', 'giai-tri', 'giao-duc', 'khoa-hoc', 'kinh-doanh', 'phap-luat', 'so-hoa', 'suc-khoe', 'the-gioi', 'the-thao', 'thoi-su']
		list_accent_category = ['Đời sống', 'Du lịch', 'Giải trí', 'Giáo dục', 'Khoa học', 'Kinh doanh', 'Pháp luật', 'Số hóa', 'Sức khỏe', 'Thế giới', 'Thể thao', 'Thời sự']
		self.category = dict(zip(list_category, list_accent_category))
		self.prop = []
	
	# Get stopwords list
	def getStopWordsList(self):
		filename = 'vietnamese-stopwords.txt'
		filepath = "./stopword/" + filename
		# print(os.path.exists(filepath)) --> TRUE
		if(os.path.exists(filepath)):
			with open(filepath, "r", encoding='utf-8') as f:
				words = f.readlines()
				for stopword in words:
					if stopword:
						self.list_stopwords.append(stopword.strip())
				f.close()
				
	def printStopwordsList(self):
		print(len(self.list_stopwords))
		
	def readfile(self):
		try:
			self.file = open(self.testfile, "r", encoding='utf-8')
		except:
			print("File is not exist.")
			
	# Check if string is number
	def is_number(self, s):
		try:
			float(s)
			return True
		except ValueError:
			pass

		try:
			unicodedata.numeric(s)
			return True
		except (TypeError, ValueError):
			pass

		for char in s:
			if char.isdigit():
				return True
		
		return False
	
	# Cleaning a string text
	def text_cleaner(self, text):
		
		#print("Stopwords detected: " + str(len(self.list_stopwords)))

		punc_file = "./stopword/punctuation.txt"
		punc_list = []
		with open(punc_file, "r", encoding='utf-8') as f:
			punctuations = f.readlines()
			for punc in punctuations:
				punc_list.append(punc.strip())
		no_punc = "".join([w for w in text if w not in punc_list])

		no_punc = no_punc.strip()

		#print("punctuation removed: " + str(no_punc))

		no_punc = re.sub(" +", " ", no_punc)

		#print("spaces removed: " + str(no_punc))

		line_wtokenized = word_tokenize(no_punc)

		#print("word segmentation: " + str(line_wtokenized))

		cleaned_text = [c.lower() for c in line_wtokenized if not (c.lower() in self.list_stopwords)]

		#print("Stopword removed: " + str(cleaned_text))

		#is_vietnamese = [c for c in cleaned_text if not (len(c)==1)]

		result = [c for c in cleaned_text if not(self.is_number(c))]

		#print("character, number removed: " + str(result))

		return result
	
	def cleaner_to_file(self, filename):
		filename = str(filename)
		with open(filename, "r", encoding='utf-8') as f:
			lines = f.read()
			
			from underthesea import sent_tokenize
			
			sentences = sent_tokenize(lines)
			sentences = sent_tokenize(lines)
			if len(sentences) > 0:
				for line in sentences:
					cleaned_text = self.text_cleaner(line)
					for word in cleaned_text:
						self.words.append(word)
			f.close()
		#createOutputPath()
		#global _PATH, _TYPE
		#opath = "./data/cleaned/input/"
		#for type in _TYPE:
			#current_path = _PATH + type
			#output_path = opath + type
			# get all files from each category folder
			#files = os.listdir(current_path)
			#for file in files:
				#current_file = current_path + "/" + file
				#output_file = output_path + "/" + file

				#print("\nReading file: " + current_file)
				#print("\nOutput success: " + output_file)

				#with open(current_file, "r", encoding='utf-8') as rf, open(output_file, "w", encoding='utf-8') as wf:
					#lines = rf.readlines()
					#if lines:
						#for line in lines:
							#cleaned_text = text_cleaner(line)
							#for word in cleaned_text:
								#wf.write(word + "\n")
					#rf.close()
					#wf.close()
					
	def createTestFile(self):
		#for type in _TYPE:
		
		# Lấy các keys (từ) đã có trong file train
		train_file = "./data/train/train_data.csv"
		df = pd.read_csv(train_file)
		keys = df.columns.tolist()
		keys = keys[:-1]
		
		tf_idf = dict.fromkeys(keys, 0)

		for word in self.words:
			word = word.strip()
			if word in keys:
				tf_idf[word] += 1
			
		self.data.append(list(tf_idf.values()))
		
		df_csv = pd.DataFrame(self.data, columns = keys)
		df_csv.to_csv(r'./data/test/test.csv', index=False, encoding='utf-8-sig')
		# print(current_path)
		#files = os.listdir(current_path)
		# print(files)
		#print("\nREADING CATEGORY: " + type)
		#for file in files:
			#print("\nfile: " + file)
			#current_file = current_path + "/" + file
			# print(current_file)
			#with open(current_file, "r", encoding='utf-8') as f:
				#lines = f.readlines()
				#if lines:
					#for line in lines:
						#line = line.strip()
						#tf_idf[line] += 1
				#f.close()
			#data.append(list(tf_idf.values()))
			
	def add_accent(self, result):		
		return self.category[result]
	
	def classfy_text(self):
		filename = './data/train/train_data.csv'

		df = pd.read_csv(filename)

		label = df.columns[-1:]
		name_label = label[0]

		list_label = df.iloc[:, -1]

		content = df.drop(['Decision'], axis=1)

		label_numpy = np.array(list_label.values.tolist())

		content_numpy = content.values
	
		#testfile = './data/test/test.csv'

		#test_df = pd.read_csv(testfile)

		test_line = np.array(list(self.data))
		
		## call MultinomialNB
		clf = MultinomialNB()

		# training 
		clf.fit(content_numpy, label_numpy)
		
		result = clf.predict(test_line)[0]
		#prop = clf.predict_proba(test_line)
		return self.add_accent(result)
	


# In[106]:
def crawl():
	type_list = scraper.scrap()
	messagebox.showinfo("Thông báo", "Đã tải xong dữ liệu")
	

def classify():
	execute = Classification()
	execute.getStopWordsList()
	
	# Lấy đường dẫn file
	filename = str(data_entry.get())
	# gọi hàm cleaner_to_file để tách từ và lưu vào list words của class
	execute.cleaner_to_file(filename)
	# gọi hàm createTestFile() để tạo file test.csv, có các cột là các từ đã xử lý (không có cột Decision)
	execute.createTestFile()
	# gọi hàm classfy_text() để trả về kết quả dự đoán
	result = execute.classfy_text()
	#print("Văn bản thuộc thể loại: " + result)
	messagebox.showinfo("Kết quả phân loại", "Văn bản thuộc thể loại: " + str(result))
	'''
	pro_str = ""
	for i in range(len(prop)):
		pro_str += (prop[i] + "\n")
	messagebox.showinfo("Xác suất dự đoán từng thể loại là:", pro_str)
	'''
	

def textSegmentation():
	segmentation.main()
	messagebox.showinfo("Kết quả", "Đã chuẩn hóa văn bản")
# In[107]:

def trainTestSplit():
	path = "./data/dataset/dataset_sample.csv"

	#filename = path_leaf(path)
	filename = "dataset_sample.csv"

	df = pd.read_csv(path)

	label = df.columns[-1:]
	name = label[0]

	rows, cols = df.shape

	test_size = int(float(rows) * 0.3)

	indices = df.index.tolist()

	test_indices = random.sample(population=indices, k=test_size)

	test_df = df.loc[test_indices]

	train_df = df.drop(test_indices)

	filename_train = "./data/train/train_data.csv"
	train_df.to_csv(filename_train, index=None, encoding='utf-8-sig')

	test_df = test_df.drop(columns=['Decision'])
	filename_test = "./data/test/test_data.csv"
	test_df.to_csv(filename_test, index=None, encoding='utf-8-sig')

	file_test_indices = "./data/test_indices.txt"
	with open(file_test_indices, "w", encoding='utf-8') as f:
		for index in test_indices:
			f.write(str(index) + "\n")
		f.close()
	
	
	test_line = test_df.values.tolist()
	
	tested_label = []
	
	for i in range(0, len(test_line)):
		test = np.array([test_line[i]])
		tested_label.append(clf.predict(test)[0])

	indices_file = './data/test_indices.txt'
	with open(indices_file, "r", encoding='utf-8') as f:
		indices = f.readlines()
		list_indices = []
		if indices:
			for indice in indices:
				if indice:
					list_indices.append(indice.strip())
		f.close()
	
	# Test File Train
	origin_df = pd.read_csv(origin_filename)

	origin_label = []
	for index in list_indices:
		index = int(index)
		origin_label.append(origin_df.iloc[index, -1])

	numerator = 0
	denominator = len(origin_label)
	for i in range(len(tested_label)):
		if tested_label[i] == origin_label[i]:
			numerator += 1
	prob = (numerator / denominator) * 100

	messagebox.showinfo("Xác suất test file: ", prob)

def fileDialogDF():
	gui.filename = filedialog.askopenfilename(initialdir="./",title = "Chọn file", filetype = (("txt files","*.txt"),("all files","*.*")))
	data_entry.delete(0,END)
	data_entry.insert(0,gui.filename)


gui = Tk()
# LABEL FRAME
frame = LabelFrame(gui, text = "Phân loại văn bản", bd=5, font=20)
frame.pack_propagate(0)
frame.pack(fill='both', expand='yes')

label_download = Label(frame, text="Download", width=10, font=15)
label_download.grid(row=0, column=0)

label_download = Label(frame, text="Tách câu, tách từ,...", font=15)
label_download.grid(row=1, columnspan=2, sticky=W)

label_test = Label(frame, text="Tạo file test train", font=15)
label_test.grid(row=2, column=0)

label_select = Label(frame, text="Chọn file: ", width=10, font=15)
label_select.grid(row=3, column=0)

browse = Button(frame,text="Bắt đầu download",command=crawl,font=0.5)
browse.grid(row=0, column=2)

segment = Button(frame,text="Bắt đầu chuẩn hóa",command=textSegmentation,font=0.5)
segment.grid(row=1, column=2)

test = Button(frame,text="Bắt đầu test",command=trainTestSplit,font=0.5)
test.grid(row=2, column=2)

browse = Button(frame,text="Duyệt",command=fileDialogDF,font=0.5)
browse.grid(row=3, column=2)

start = Button(frame, text="Bắt đầu", command=classify, font=15, padx=10)
start.grid(row=4, column=2)

data_entry = Entry(frame, font=30)
data_entry.grid(row=3, column=1)

gui.mainloop()


# In[ ]:




