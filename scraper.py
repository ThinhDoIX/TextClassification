from bs4 import BeautifulSoup
import requests
import os


def scrap():
	#exclude = ['cuoi', 'tam-su', 'y-kien', 'goc-nhin', 'du-lich', 'giai-tri', 'giao-duc', 'khoa-hoc', 'kinh-doanh', 'phap-luat', 'so-hoa', 'suc-khoe', 'the-gioi', 'the-thao', 'thoi-su']
	exclude = ['cuoi', 'tam-su', 'y-kien', 'goc-nhin']
	
	# number_of_pages = int(input('Number of page: '))
	number_of_pages = 5
	session = requests.session()

	main_url = 'https://vnexpress.net'
	page = session.get(main_url)

	soup = get_soup(page)
	main_menu = soup.find('nav', id='main_menu')

	type_list = get_menu_list(main_menu)

	type_list = [type_ for type_ in type_list if type_ not in exclude]

	for type_ in type_list:
		url = f'{main_url}/{type_}'
		file_number = 1

		path = get_path(type_)

		for page_number in range(number_of_pages):
			if page_number != 0:
				url = f'{main_url}/{type_}-p{page_number + 1}'
				atl_url = f'{main_url}/{type_}/p{page_number + 1}'

			page = session.get(url)
			print(page, url)

			if check_error(page.status_code):
				page = session.get(atl_url)

			try:
				soup = get_soup(page)
				soup.find('section', class_='sidebar_2').extract()
			except AttributeError:
				continue

			article_list = soup.find_all('h4', class_='title_news')

			url_list = get_url_list(article_list)

			for url in url_list:
				page = session.get(url)
				print(page, url)

				if check_error(page.status_code):
					continue

				if file_number < 10:
					file = f'{path}0{file_number}.txt'
				else:
					file = f'{path}{file_number}.txt'
				file_number += 1

				scrap_(page, file)

	return type_list


def scrap_(page, file):
	soup = get_soup(page)

	try:
		title = soup.find('h1', class_='title_news_detail').text
		description = soup.find('p', class_='description').text
	except AttributeError:
		return

	paragraphs = soup.find_all('p', class_='Normal')

	with open(file, 'w', encoding='utf-8') as f:
		f.write(title.rstrip() + '\n')
		f.write(description.rstrip() + '\n')

		for paragraph in paragraphs:
			sentences = split_sentence(paragraph.text)

			for sentence in sentences:
				f.write(sentence)


def get_soup(page):
	return BeautifulSoup(page.text, 'lxml')

def check_error(status_code):
	return False if status_code == 200 else True

def get_menu_list(menu):
	url_list = []

	a_tags = menu.find_all('a', href=True)
	for a_tag in a_tags:
		url_list.append(a_tag.get('href'))

	url_list = list(map(lambda url: url.split('/')
						[-1].split('?')[0], url_list))

	return list(x for x in url_list if len(x.split('-')) == 2)
	
def get_path(type_):
	seperator = get_seperator()

	if not os.path.exists(f'data{seperator}articles{seperator}{type_}'):
		os.makedirs(f'data{seperator}articles{seperator}{type_}')

	return f'data{seperator}articles{seperator}{type_}{seperator}'
	
def get_url_list(list_):
	url_list = []
	for element in list_:
		a_tag = element.find('a', href=True)

		if a_tag is not None:
			url_list.append(a_tag.get('href'))

	return url_list
	
def split_sentence(paragraph):
	punctuations = ('. ', '? ', '! ', '; ', ': ')

	sentence = []
	sentences = []
	words = [word.strip() + ' ' for word in paragraph.split(' ') if word]

	for word in words:
		sentence.append(word)
		if word.endswith(punctuations):
			sentences.append(''.join(sentence) + '\n')
			sentence = []

	return sentences	
	
def get_seperator():
	return '\\' if os.name == 'nt' else '/'

if __name__ == "__main__":
	scrap()
	print('Done')
