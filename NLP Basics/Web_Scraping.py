# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 18:37:45 2018

@author: akansal2
"""
#importing libraries
import requests
from bs4 import BeautifulSoup



#getting the page
page = open('C://A_stuff/Learning/Machine Learning/Udemy/NLP Basic/test.html')


#page proerties
print(page)
#print(page.content)


#parsing the data
soup = BeautifulSoup(page,'html.parser')


#BeautifulSoup properties
print(soup.prettify())   #to format/align the tag


#traversing through a page step by step
print(list(soup.children)[0])

html = list(soup.children)[0]
print(list(html.children)[3])

body = list(html.children)[3]

print(list(body.children)[1])

p = list(body.children)[1]

print(p.get_text())



#finding all instances at once
print(soup.find_all('p'))
print(soup.find_all('p')[0].get_text())

#To find only first instance
print(soup.find('p'))


#finding with class/id attribute
print(soup.find_all('p',class_ = 'outer-text'))
print(soup.find_all(class_='outer-text' ))
print(soup.find_all(id = 'first'))




#using css
print(soup.select('div p'))





#Creating a python wikipedia
print('Enter to the world of Python wikipedia...')
topic = input('enter the topic here = ')


link = 'https://en.wikipedia.org/wiki/' + topic
link = 'http://guihelp.sxc.com/V8R4M01/CLMELGJW/!SSL!/WebHelp/1_CAG/1_CAGM/WICar.htm'

#accessing the link
page = requests.get(link)
html = page.text


#accessing soup data
soup =BeautifulSoup(html,'html.parser')

print(soup.find_all('p')[0].get_text())




















































