#!/usr/bin/env python
# coding: utf-8

# In[2]:


import csv
import os
import re
import pandas as pd
import unicodedata
from underthesea import word_tokenize
from pyvi import ViTokenizer
from langdetect import detect
import requests

# In[3]:


# CONSTANT
_PATH = './data/articles/'
_TYPE = os.listdir(_PATH)


# In[4]:


# Get stopwords list
def getStopWordsList():
	filename = 'vietnamese-stopwords.txt'
	global _PATH
	filepath = "./stopword/" + filename
	# print(os.path.exists(filepath)) --> TRUE
	if(os.path.exists(filepath)):
		list_stopwords = []
		with open(filepath, "r", encoding='utf-8') as f:
			while True:
				word = f.readline()
				if word:
					list_stopwords.append(word.strip())
				else:
					break
			f.close()
	return list_stopwords


# In[5]:


# Create all category's directories
def createOutputPath():
	global _TYPE
	path = "./data/cleaned/"
	for type in _TYPE:
		current_path = path + type
		if not os.path.exists(current_path):
			os.mkdir(current_path)


def is_number(s):
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


# In[89]:


def text_cleaner(text):
		
	#print("Stopwords detected: " + str(len(self.list_stopwords)))
	list_stopwords = getStopWordsList()
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

	cleaned_text = [c.lower() for c in line_wtokenized if not (c.lower() in list_stopwords)]

	#print("Stopword removed: " + str(cleaned_text))
	
	#is_vietnamese = [c for c in cleaned_text if not (len(c)==1)]

	result = [c for c in cleaned_text if not(is_number(c))]

	#print("character, number removed: " + str(result))

	return result


# In[90]:


def cleaner_to_files():
	createOutputPath()
	global _PATH, _TYPE
	opath = "./data/cleaned/"
	for type in _TYPE:
		current_path = _PATH + type
		output_path = opath + type
		# get all files from each category folder
		files = os.listdir(current_path)
		for file in files:
			current_file = current_path + "/" + file
			output_file = output_path + "/" + file
			
			print("\nReading file: " + current_file)
			print("\nOutput success: " + output_file)
			
			with open(current_file, "r", encoding='utf-8') as rf, open(output_file, "w", encoding='utf-8') as wf:
				lines = rf.readlines()
				if lines:
					for line in lines:
						cleaned_text = text_cleaner(line)
						for word in cleaned_text:
							wf.write(word + "\n")
				rf.close()
				wf.close()


# In[91]:


#cleaner_to_files()


# In[92]:


def generate_header():
	header_columns = set()
	for type in _TYPE:
		path = "./data/cleaned/" 
		current_path = path + type
		# print(current_path)
		files = os.listdir(current_path)
		# print(files)
		#print("\nREADING CATEGORY: " + type)
		for file in files:
			#print("\nfile: " + file)
			bagOfWords = []
			current_file = current_path + "/" + file
			# print(current_file)

			with open(current_file, "r", encoding='utf-8') as f:

				lines = f.readlines()
				if lines:
					for line in lines:
						if line:
							bagOfWords.append(line.strip())
				f.close()
			header_columns = header_columns.union(set(bagOfWords)) 
	return header_columns


# In[93]:
#keys = list(generate_header())

# In[94]:
#keys.append('Decision')


# In[95]:

def createData():
	data = []
	for type in _TYPE:
		tf_idf = dict.fromkeys(keys, 0)
		tf_idf['Decision'] = type
		
		path = "./data/cleaned/" 
		current_path = path + type
		# print(current_path)
		files = os.listdir(current_path)
		# print(files)
		#print("\nREADING CATEGORY: " + type)
		for file in files:
			#print("\nfile: " + file)
			current_file = current_path + "/" + file
			# print(current_file)
			with open(current_file, "r", encoding='utf-8') as f:
				lines = f.readlines()
				if lines:
					for line in lines:
						line = line.strip()
						tf_idf[line] += 1
				f.close()
			data.append(list(tf_idf.values()))
	return data


# In[96]:

	



# In[97]:



def main():
	cleaner_to_files()
	
	keys = list(generate_header())
	keys.append('Decision')
	
	data = createData()
	df_csv = pd.DataFrame(data, columns = keys)
	df_csv.to_csv(r'./data/dataset/dataset_sample.csv', index=False, encoding='utf-8-sig')

if __name__ == "__main__":
	main()
	print("Done")
	
	
# In[ ]:




