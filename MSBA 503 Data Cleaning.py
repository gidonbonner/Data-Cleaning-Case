# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 00:51:48 2020

@author: jperols
"""

#pip install lxml
#pip install requests-html
#nltk.download('wordnet')


import numpy as np
import pandas as pd 
from bs4 import BeautifulSoup
from requests_html import HTMLSession
import re
import unicodedata
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
#from sklearn_pandas import DataFrameMapper
import matplotlib.pyplot as plt
from io import StringIO



#*****************************************************************************
#*********** Import and Combine Financial Data and Bankruptcy Data *********** 
#*****************************************************************************
file_path = 'C:/Users/bonne/Downloads/'
Financials_2011_2019 = file_path + 'Financials(2011_2019).csv'
bankruptcy_firm_years = file_path + 'bankruptcy_firm_years.csv'

#*********** Financial Data *********** 
#Import financial data, drop all firms without CIK code (as this is what we will use to
#merge the financial data with the bankruptcy data).  Change CIK code to int to make sure
#that the CIK codes in the two tables have the same data type (I will later set the bankruptcy
#CIK code to also int.  Change datadate to a datetime format.

all_firm_years = pd.read_csv(Financials_2011_2019, low_memory=False)
all_firm_years.dropna(subset=['cik'], inplace=True)
all_firm_years.cik = all_firm_years.cik.astype(int)
all_firm_years.datadate = pd.to_datetime(all_firm_years.datadate)

#Only keep some of the data (I created this list by printing all_firm_years.columns and copying and pasting the list
#and then removing the columns I did not want to keep).
all_firm_years = all_firm_years[['gvkey', 'datadate', 'fyear', 'conm', 'cik', 'act', 'ap', 'at', 'ch', 'cogs', 'csho','do', 'dvt', 'ebit', 'emp', 'ib', 'invt', 'lct', 'lt', 'ni', 'nipfc','oancf', 'opiti', 'ppent', 're', 'rect', 'revt', 'seq', 'xopr', 'xsga','prcc_c', 'mkvalt', 'au', 'auop', 'auopic']]

#Create a list of columns that are numeric columns and then loop through these columns to create
#lagged and change variables.
data_columns = ['act', 'ap', 'at', 'ch', 'cogs', 'csho','do', 'dvt', 'ebit', 'emp', 'ib', 'invt', 'lct', 'lt', 'ni', 'nipfc','oancf', 'opiti', 'ppent', 're', 'rect', 'revt', 'seq', 'xopr', 'xsga','prcc_c', 'mkvalt', 'au', 'auop', 'auopic']

for col in data_columns:
    all_firm_years['prior_' + col] = all_firm_years.groupby('gvkey')[col].shift(1)
    all_firm_years['change_' + col] = all_firm_years[col] - all_firm_years['prior_' + col]    

#*********** Bankruptcy Data *********** 
#Import bankruptcy data, keep only three columns, change the data type of CIK code to in
#to make sure it has the same datatype as CIK code in financial data (for the merge), and 
#set the data type of File date to datetime.

bankruptcy_firms = pd.read_csv(bankruptcy_firm_years, low_memory=False)
bankruptcy_firms = bankruptcy_firms[['CIK Code', 'File date', 'Bankruptcy']]
bankruptcy_firms['CIK Code'] = bankruptcy_firms['CIK Code'].astype(int)
bankruptcy_firms['File date'] = pd.to_datetime(bankruptcy_firms['File date'])

#*********** Merge Financial and Bankruptcy Data *********** 
#Both DataFrames must be sorted by the key
all_firm_years.sort_values(by=['datadate'], ascending=[True], inplace = True)
bankruptcy_firms.sort_values(by=['File date'], ascending=[True], inplace = True)

#Perform the merge on CIK code, but because a single firm can (and typically does)
#have multiple firm years in the financial data, also merge on dates.  The bankruptcy
#data contains the field 'File date' which indicates the date that the bankruptcy notice 
#was filed.  Since we are trying to predict future bankruptcies we want financial statement
#data to be prior to the file date but as close to the file date as possible.  To perform this
#merge I am using pd.merge_asof.
bankruptcy_firms = pd.merge_asof(bankruptcy_firms, all_firm_years, left_on='File date', right_on='datadate', left_by='CIK Code', right_by='cik', allow_exact_matches=False, direction='backward')

#It is possible that some firms were listed in the bankruptcy data twice, which will create duplicates 
#in the merged data.  We probably should be looking into this, but to keep this simple we will simply
#drop all duplicates from the merged data based on cik code.
bankruptcy_firms.dropna(subset=['cik'], inplace=True)
bankruptcy_firms.drop_duplicates('cik', inplace=True)

#*********** Create Matched Non-Bankruptcy Sample *********** 
#We are going to create a matched sample of firms that look similar in terms of their financials. If we look at
#a bankruptcy firm right before they are about to go bankrupt they are likely to have very poor financials and it would
#be fairly easy to discrimate such firms from most other firms.  However, if we instead try to determine which firms will go
#bankrupt in the next year from a subsample of firms that all look similar in terms of their finanicals then we have a better
#use case for a situation where having MDA information might be useful. The code in this section creates this subsample.

#First, find all non-bankruptcy firm years by performing a left outer join on gvkey (and ony include gvkey and Bankruptcy 
#columns from the bankruptcy_firms (just to make it cleaner if you want to look at it) and then only keeping
#firm years from all_firm_years where there was not match in the bankruptcy table. Note that we might find
#some firms without matches that are actually in the original bankruptcy data but that do not have a CIK code that 
#we could match or that we cannot match to the data for some other reason.  We can reduce the risk of including such
#firm years if we only include all_firm_years observations that have cik codes. This, however, also reduces the number
#of firms that can potentially be match to the bankruptcy firms (which will allow us to get better matches). In this 
#example, I decided to go with having more is better, but this could be easily adjusted by dropping all firm years 
#without a CIK code.)
non_bankruptcy_firms = pd.merge(all_firm_years, bankruptcy_firms[['gvkey', 'Bankruptcy']], how='left', on='gvkey')
non_bankruptcy_firms = non_bankruptcy_firms[non_bankruptcy_firms.Bankruptcy.isnull()]
non_bankruptcy_firms.Bankruptcy = 0
bankruptcy_firms = bankruptcy_firms[non_bankruptcy_firms.columns]

#We will be creating the matched sample based on net income (ni), change in net income, assets (at), change in assets,
#market value (mkvalt), and change in market value and all firm years must therefore have these data.
bankruptcy_firms.dropna(subset=['change_ni', 'ni', 'change_at', 'at', 'change_mkvalt', 'mkvalt'], inplace=True)
non_bankruptcy_firms.dropna(subset=['change_ni', 'ni', 'change_at', 'at', 'change_mkvalt', 'mkvalt'], inplace=True)

#Create 2d arrays of the observations with only the columns needed for matching and use 
#StandardScaler to standardize these values (and StandardScaler needs an array as input).
#I scale these values as KKN that I use later is sensitive to the magnitude of different variables
#used in the matching and I do not want to emphasize one variable more than any other.
bankruptcy_x = bankruptcy_firms[['change_ni', 'ni', 'change_at', 'at', 'change_mkvalt', 'mkvalt']]
non_bankruptcy_x = non_bankruptcy_firms[['change_ni', 'ni', 'change_at', 'at', 'change_mkvalt', 'mkvalt']]
scaler = StandardScaler()
scaler.fit(bankruptcy_x)
bankruptcy_x = scaler.transform(bankruptcy_x)
non_bankruptcy_x = scaler.transform(non_bankruptcy_x)

#Use KNN to find similar firms. For each bankruptcy firm, locate the four closes matches.
#NearestNeighbors returns a tuple containing two 2d arrays.  The first 2d array contains distances where
#each "row" contain distances to an observation that we are trying to find knn for. So when setting n_neighbors
#to 4, we end up with rows with four elements (so four columns).  If we are trying to find knn of non-bankruptcy firms
#for the bankruptcy firms then each row represent the distance between the closest four non-bankruptcy firms to a single
#bankruptcy firm.  The second 2d array is structured similar, but instead of showing the distances it has the indicies of 
#the non-bankruptcy firms that are closest to the bankruptcy firm. 
dis_inds = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(non_bankruptcy_x).kneighbors(bankruptcy_x)

#To locate the finanical data for the nearest neighbours we need to get a 1d array (rather than a 2d array) of all the indices.
#We then use these indices in a simple filter using iloc.
# dis_inds[1] selects the 2d array with indicies 
# dis_inds[1].shape[0] returns the number of rows in the 2d array, i.e., 172
# dis_inds[1].shape[1] returns the number of columns in the 2d array, i.e., 4
# the product of the number of columns and rows gives us the nunber of elements, 
# which then is passed into .reshape to create a 1D array with the same number of
# elements.

indices = dis_inds[1].reshape(dis_inds[1].shape[0]*dis_inds[1].shape[1]) 
# I could instead have used:
# dis_inds[1].reshape(len(dis_inds[1]))
# or
# dis_inds[1].reshape(-1)
# Save the matched non bankrupt firms only.
matched = non_bankruptcy_firms.iloc[indices]

#Finally, we stack the bankruptcy_data and matched non-bankruptcy data to create our bankruptcy dataset.
bankruptcy_data = pd.concat([bankruptcy_firms, matched])
bankruptcy_data['cik'] = bankruptcy_data['cik'].astype(int)

#*********************************************************************************
#******************************** Scrape MDA Data ******************************** 
#*********************************************************************************

#For each observation (bankruptcy firm years and matched non-bankruptcy firm years) in the bankruptcy dataset, we next need 
#to find the MDA from the 10K corresponding to the specific firm and year of each firm year observation.
#Note that we are only looking for 10ks as the finanical data we have is only from 10K.

#Create a list of CIK codes and years corresponding to the firm years that we are trying to find.
bankruptcy_data['datadate_year']=bankruptcy_data.datadate.dt.year
MDA_search_list = bankruptcy_data[['cik','datadate_year','datadate']].copy()
MDA_search_list['bankruptcy_data_index']=bankruptcy_data.index
MDA_search_list.sort_values(by=['datadate_year'], ascending=[True], inplace = True)


#The code below is currently not able to process html pages rendered using java script and a large majority of financial statements that are not 
#processed successfully are related to java script rendering not working.  I tried using another request method but this resulted in errors that 
#appear to be problems with the library I am using.  An easier and better solution would be to use the Master.idx file instead of the Crawler.idx 
#file as these contain text documents of the financial statements with the java script already rendered.  There are also some
#files that are not saving because they are short - these often refer to another location, e.g., appendix, where the MDA section is locate.
#This is much more difficult to fix.

#Initalize counters to make sure we don't get stuck in an endless loop (and to display how the processing is progressing).
j=0             #counter for max number of MDA sections to save
j_limit=10
#MDA_search_list.shape[0]+1     #the max number of MDA sections to save
k=0             #counter for max number of MDA sections to process
k_limit = 10
#3*MDA_search_list.shape[0]  #the max number of MDA sections to process

#Create two new columns in the bankruptcy_data dataframe to store scraped MDA text
bankruptcy_data['MDA_Text']=""
bankruptcy_data['MDA_List']=""
AllData = pd.DataFrame(columns=['company_name', 'form', 'cik', 'date_filed', 'url', 'MDA', 'MDA_List'])

session = HTMLSession()

#For each year (but only search through relevant years) and each quarter open the crawled.idx.
for Y in range(MDA_search_list.datadate_year.min(), MDA_search_list.datadate_year.max()+1): 
    #Filter the list of firm year CIK codes from the bankruptcy sample to only include firm years for the
    #year that is currently being processed.
    for Q in range(1,5):
        #Open crawler.idx for the year and quarter currently being processed. Do not import the first 6 rows (which contain header information)
        fwf_file = session.get('https://www.sec.gov/Archives/edgar/full-index/' + str(Y) + '/QTR' + str(Q) + '/crawler.idx').text
        URLs = pd.read_fwf(StringIO(fwf_file), colspecs=[(0,60), (60,72), (72,84), (84,96), (96,200)], skiprows=2)               
        URLs.columns=URLs.iloc[0] #The first row (really the 7th) of crawled.idx contains the field headers
        URLs=URLs.iloc[2:] #Exclude the field header row and the row below as this is a line with a bunch of dashes.

        #Only look for 10Ks.
        URLs = URLs[URLs['Form Type']=='10-K']
        URLs.CIK = URLs.CIK.astype(int)

        #Perform an inner join between the list of 10Ks from the year and period being processed with the list of 
        #firm years that were are searching for.  This creates a list of all URLs for 10Ks that was filed in this
        #year and quarter that are on the search list.
        URLs = pd.merge(URLs, MDA_search_list[MDA_search_list.datadate_year==Y], how='inner', left_on='CIK', right_on='cik')
        URLs.drop_duplicates(subset='cik', keep='last', inplace=True)
        
        #Start the actual downloading and processing of MDA section
        CIK_list =[]
        for company in URLs.index:
            approach=0     #used to keep track of the approach used for extracting each MD&A section 
            MDA_text = ""  #Used to store scraped MD&A text. Reset to an empty string each loop iteration.
            MDA=""
            
            #Connect to the first company filing page to get the URL to the financial statement 
            #for the company and create BeautifulSoup object to make it easy to go through table tags.
            soup = BeautifulSoup(session.get(URLs['URL'][company]).text, features="lxml")
        
            #I assume the URL is in the first table, the third row and the fifth column:
            financial_statment_URL = "https://www.sec.gov" + soup.table.contents[3].contents[5].a.get('href')
        
            #Connect to the financial statements page and create soup object
            financials_soup = BeautifulSoup(session.get(financial_statment_URL).text, features="lxml")
            
            #Remove HTML tags (extract the text elements) and normalize the text to make it easier to work 
            #with. 
            financials_text = unicodedata.normalize("NFKD", financials_soup.text)
            
            #The two objects, financials_soup and financials_text, are used to scrape MD&A data using 
            #BeautifulSoup and regext, respectively.  I use a total of three different approaches that 
            #are executed in the order that they are likely to provide the best precision in getting the 
            #MDA section only and nothing less.

            # *********** Approach 1 *************
            #The first approach searches for URLs to MD&A anchor tags (link within the document itself, typically provided in Table of Content). I am performing two versions, one that is simpler but fails when the link text is not within 
            start_anchor_href = "" 
            start_anchor_tag  = "" 
            end_anchor_href = "" 
            start_anchor_tag = financials_soup.find('a', string=re.compile('Management.{1,12}Discussion'))
            if start_anchor_tag:
               start_anchor_href = start_anchor_tag.get('href')
               end_anchor_href = start_anchor_tag.find_next('a').get('href')
               if start_anchor_href==end_anchor_href:
                  end_anchor_href = start_anchor_tag.find_next('a').find_next('a').get('href')
            

            if not start_anchor_href or not end_anchor_href or start_anchor_href == end_anchor_href:
               #The first approach searches for URLs to anchor tags provided in Table of Contents.  
               #It starts by looking in all table rows for the text Management's Discussion and then finds 
            #the start and end URL anchor names for the MDA section and the subsequent section.
               tr_list = financials_soup.find_all('tr') #get a list of all table rows
               MDARow = 0            #variable to keep track of the row where the MDA URL anchor is found
               NextSectionRow = 0    #keeping track of the row of the URL anchor for the section after MDA
               start_anchor_href = "" 
               start_anchor_href  = "" 
               for i, row in enumerate(tr_list): 
                   if re.search('Management.{1,12}Discussion', str(row), re.IGNORECASE | re.DOTALL):
                       MDARow = i
                   elif MDARow and re.search('Item', str(row), re.IGNORECASE | re.DOTALL):
                       NextSectionRow = i
                       #If the table rows with Management Discussion and Item both contain an a href tag then 
                       #get the href
                       if tr_list[MDARow].a and tr_list[NextSectionRow].a: 
                            start_anchor_href = tr_list[MDARow].a.get('href')
                            end_anchor_href = tr_list[NextSectionRow].a.get('href')
                       break
        
            #If MDA and next section anchor names were found, then create a regex search pattern that 
            #finds the first anchor and the second anchor and all the text between them.
            #The code uses re.escape because some of the anchor names have special characters that means 
            #something specific in regex (to be able to use regex but still interpret the anchor as 
            #regular text rather than special regex characters all non-alphanumeric characters can be back 
            #slashed, but it is not possible to do this manually as we do not know what the anchor names 
            #are - this is why I need the function re.espace to do this for me.
            if start_anchor_href and end_anchor_href:
                #The a href tag contains a character (#, which is used to indicate a link to an anchor in the same document that is create with a name tag) before the actual anchor name that is not 
                #part of the anchor name. So need to remove this character.
                start_anchor_href = start_anchor_href[1:]
                end_anchor_href = end_anchor_href[1:]
                
                #This uses regex to find all the text between the start_anchor name tag and the end_anchor tag, which are located using beautifulsoup. I have to use {} to find the HTML name element of the tax because Beautiful Soup uses the name argument to contain the name of the tag itself (it's a reserved key word).
                if financials_soup.find('a', {"name":start_anchor_href}) and financials_soup.find('a', {"name":end_anchor_href}): 
                   MDA = re.findall(str(financials_soup.find('a', {"name":start_anchor_href})) + ".*" + str(financials_soup.find('a', {"name":end_anchor_href})), str(financials_soup), re.I | re.DOTALL)
                elif financials_soup.find('a', {"id":start_anchor_href}) and financials_soup.find('a', {"id":end_anchor_href}): 
                   MDA = re.findall(str(financials_soup.find('a', {"id":start_anchor_href})) + ".*" + str(financials_soup.find('a', {"id":end_anchor_href})), str(financials_soup), re.I | re.DOTALL)
                elif re.search("<a(?!.{1,100}href).{1,100}?" + re.escape(str(start_anchor_href)), str(financials_soup), re.IGNORECASE | re.DOTALL) and re.search("<a(?!.{1,100}href).{1,100}?" + re.escape(str(end_anchor_href)), str(financials_soup), re.IGNORECASE | re.DOTALL):
                   MDA = re.findall("<a(?!.{1,100}href).{1,100}?" + re.escape(str(start_anchor_href)) + ".*?<a(?!.{1,100}href).{1,100}?" + re.escape(str(end_anchor_href)), str(financials_soup), re.I | re.DOTALL)
                if MDA:
                   MDA_text = str(BeautifulSoup(str(MDA), features="lxml").get_text())
                   approach=1    
                   
# =============================================================================
#                 if ((j>=j_limit) | (k>=k_limit)):
#                     break
#                 k+=1
#                 if len(MDA_text)>500:
#                     j+=1                 
#                 
#                     #If the file is an addendum then the code needs to reduce j by one to make sure that we save the correct number of files that we want.  I create a list to keep track of 
#                     #of CIK codes that we have already saved in the period being processed.  If a file is saved with an already existing CIK code then 
#                     #j is reduced by 1. If the file is saved with a new CIK code then the CIK code is added to the list. Note that if the same firm can appear multiple times
#                     #then the code needs to be changed to check and keep track of year in addition to CIK.
#                     if URLs['CIK'][company] in CIK_list:
#                         j-=1
#                         print('\nAddendum: ' + financial_statment_URL + '\n')
#                     else:
#                         CIK_list += [URLs['CIK'][company]]
#                         print('Processed: '+ str(k))
#                         print('Saved: '+ str(j))
#                         print('Approach Used: '+ str(approach))
#                         print('URL: ' + financial_statment_URL)
#                         print('')
#                     AllData.loc[j-1, 'company_name'] = URLs['Company Name'][company]
#                     AllData.loc[j-1, 'form'] = URLs['Form Type'][company]
#                     AllData.loc[j-1, 'cik'] = URLs['CIK'][company]
#                     AllData.loc[j-1, 'date_filed'] = URLs['Date Filed'][company]
#                     AllData.loc[j-1, 'datadate'] = URLs['datadate'][company]
#                     AllData.loc[j-1, 'bankruptcy_data_index'] = URLs['bankruptcy_data_index'][company]
#                     AllData.loc[j-1, 'url'] = URLs['URL'][company]
#                     AllData.loc[j-1, 'MDA'] = MDA_text
#                 else:
#                     print('\nNot Saved: ' + financial_statment_URL + '\n')
#             #Need these so that the outer loops do not run one more time.
#             if ((j>=j_limit) | (k>=k_limit)):
#                 break
#         if ((j>=j_limit) | (k>=k_limit)):
#             break
# =============================================================================



# *********** Approach 2 *************
            #If it was not possible to find anchors using the approach above, then the code attempts to find the regex pattern below (explained here):
            #  "item.{0,2}[0-9]{1,2}\.." The pattern should start by the word item, followed by 0 to 2 any characters, followed by 1 or 2 digits followed by a period
            #  ".{1,12}"                 The word item should be followed by 1 to 12 characters of any type (we typically but not always have a period and a space and sometimes unicodes) 
            #  "management.{0,6}"        This should be followed by the word Management and 0 to 6 of any character to account for apostrophe (potentially as unicode)
            #  "s? discussion"           This is then followed by an optional s (the ? makes s optional) and the word discussion
            #  "(?!.{1,200}item)"        Followed by a negative lookahead (?1 )that is saying that we do not want to see the word item within 1 to 200 characters of (item.{1,12}management.{0,6}s? discussion 
            #                            The word Item this close to Item. x Management's Discussion would for example indicate that we are in the table of contents)
            #  " .*?"                    Followed by any character and any number of characters, the ? makes the * non greedy, which means it attempts to find as few characters as possible given the rest pattern to
            #                            the right of ?.  This means that regex will return everything until the first occurrence of the next pattern (rather than the last occurrence, which is what happens 
            #                            with ".*".  The code uses parentheses around the rest of the regex statement to say that all characters should be returned until the first occurrence of everything inside
            #                            the parentheses.
            #  "\..{1,10}item"           Return all text between the first part of the regex and a period that is followed by 1 to 10 characters and then the word item.  I am searching for . Item as sentences in the middle
            #                            of the MDA section rarely starts with the word Item but the end of the MDA section almost always ends with a period and this period is typically followed by the word Item (which starts
            #                            the next section.
            #  ".{1,10}(quantitative|financial statement"  
            #                            The word item should be followed by 1 to 10 characters and then either the word quantitative or financial statement.
            #However, not only will this pattern match any combination of the terms, it will also store those matches into match groups for later inspection.
            #To not store all match groups, each “or” group must be prefixed with ?=.
            #  "re.I | re.DOTALL"        The re.I argument tells regext to not care about case. The re.DOTALL tells regex to also consider newline characters to be considered any character (essentially makes wildcards
            #                            able to cross lines.
# =============================================================================
#             #I use [0] because findall stores the results in a list and I want to element (string) only.
#             if not MDA_text and re.findall("(item.{0,2}[0-9]{1,2}\..{1,12}management.{0,6}s? discussion(?!.{1,200}item).*?(\..{1,10}item.{1,10}(quantitative|financial statement)))", financials_text, re.I | re.DOTALL):  
#                 MDA_text = re.findall("(item.{0,2}[0-9]{1,2}\..{1,12}management.{0,6}s? discussion(?!.{1,200}item).*?\..{1,10}item.{1,10}(?=quantitative|financial statement))", financials_text, re.I | re.DOTALL)[0]
# 
#                 approach=2
#                 
# # *********** Approach 3 *************
#             #If the search for a specific section name fails, then the code simply looks for item followed by Management's
#             #Discussion, followed by some text and then the word item.  The problem with this statement is that the MDA
#             #section itself is likely to include the word item and this would then cut off the MDA section. To improve
#             #this code I would likely not use this code until a tried a few more specific patterns to try to figure out
#             #where the MDA section ends.
#             #s? checks for zero or more of s, so s is optional
#             if not MDA_text and re.findall("(item.{0,2}[0-9]{1,2}\..{1,12}management.{0,6}s? discussion(?!.{1,200}item).*?(\..{1,10}item))", financials_text, re.I | re.DOTALL):
#                 MDA_text = re.findall("(item.{0,2}[0-9]{1,2}\..{1,12}management.{0,6}s? discussion(?!.{1,200}item).*?\..{1,10}item)", financials_text, re.I | re.DOTALL)[0]
# 
#                 approach=3
# =============================================================================

# Another potential approach could have looked for bolded Management's Discussion text to indicate the
# start of MDA text (see below) and then bolded Item to indicate the end of the text (there are, however, other ways
#to bold text in HTML documents that would have to be accounted for). Most text is, however, processed
#using approach 1 and I think approach 2 is even accurate enough for now...

#bold_list = financials_soup.findAll('b', text = re.compile('Management.{1,12}Discussion'))[-1]

# *********** Keeping Track of Number of Loops and Saving MDA Sections *************
        
            #I am breaking when I have found j MDA sections that have at least 500 characters or if k financials 
            #have been processed. The code only saves the MDA section if it has at least 500 characters.  The break
            #code should be removed (or edited) to process more financial statements
            if ((j>=j_limit) | (k>=k_limit)):
                break
            k+=1
            if len(MDA_text)>500:
                j+=1                 
                
                #If the file is an addendum then the code needs to reduce j by one to make sure that we save the correct number of files that we want.  I create a list to keep track of 
                #of CIK codes that we have already saved in the period being processed.  If a file is saved with an already existing CIK code then 
                #j is reduced by 1. If the file is saved with a new CIK code then the CIK code is added to the list. Note that if the same firm can appear multiple times
                #then the code needs to be changed to check and keep track of year in addition to CIK.
                if URLs['CIK'][company] in CIK_list:
                    j-=1
                    print('\nAddendum: ' + financial_statment_URL + '\n')
                else:
                    CIK_list += [URLs['CIK'][company]]
                    print('Processed: '+ str(k))
                    print('Saved: '+ str(j))
                    print('Approach Used: '+ str(approach))
                    print('URL: ' + financial_statment_URL)
                    print('')
                AllData.loc[j-1, 'company_name'] = URLs['Company Name'][company]
                AllData.loc[j-1, 'form'] = URLs['Form Type'][company]
                AllData.loc[j-1, 'cik'] = URLs['CIK'][company]
                AllData.loc[j-1, 'date_filed'] = URLs['Date Filed'][company]
                AllData.loc[j-1, 'datadate'] = URLs['datadate'][company]
                AllData.loc[j-1, 'bankruptcy_data_index'] = URLs['bankruptcy_data_index'][company]
                AllData.loc[j-1, 'url'] = URLs['URL'][company]
                AllData.loc[j-1, 'MDA'] = MDA_text
            else:
                print('\nNot Saved: ' + financial_statment_URL + '\n')
        #Need these so that the outer loops do not run one more time.
        if ((j>=j_limit) | (k>=k_limit)):
            break
    if ((j>=j_limit) | (k>=k_limit)):
        break


#********************************************************************************
#******************************** Clean MDA Data ******************************** 
#********************************************************************************
#Clean the data in AllData.MDA
for text in AllData.MDA:
    text = unicodedata.normalize("NFKD", text)
    
    text = re.sub(r'\r', ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r'\\xa0', ' ', text)
    text = re.sub(r'\s', ' ', text)
    
    text = re.sub(r"\'s", "", text)
    text = re.sub(r'\\\\xa', '', text)
    text = re.sub(r'<a name[^>]*>', '', text)
    text = re.sub(r'^[\\<>]\S*\s', '', text)
    text = re.sub(r'\((.{0,24})[^()]\)', '', text)
    text = re.sub(r'(\w[A-Z]\w*)', '', text)
    text = re.sub(r'(?<!.)(\w*[A-Z]\w*)', '', text)
    text = re.sub(r'millions?|thousands?|per share', '', text)
    text = re.sub(r'\b[A-Za-z]\b', '', text)
    text = re.sub(r'[^A-Za-z0-9 ]', '', text)
    
    text = re.sub(r'([ \t]+)', ' ', text)
    
    text = text.upper()
    
    MDA_list = text.split(' ')
    
    
    stopwords = file_path + 'stopwords.txt'
    stopwords = open(file_path + 'stopwords.txt','r')
    with open(file_path + 'stopwords.txt', 'r') as file:
        stopwords = file.read().replace('\n', ' ')
        stopwords_list = stopwords.split(' ')
        MDA_list = [word for word in MDA_list if word not in stopwords_list]
        
        MDA_list = re.sub(r'[,]', ' ', str(words))
        MDA = re.sub(r"'", '', MDA_list)
        AllData['MDA'] = MDA
        AllData['MDA_list'] = MDA_list
        
       