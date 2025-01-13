# Script Name: Preprocessing_and_Topic_Modeling_Functions
# Author: Joshua (Jay) Wimhurst
# Date Created: 10/10/2023
# Date Last Edited: 1/13/2025

################################ DESCRIPTION ##################################
# This script contains all pre-processing and topic modeling functions. These
# functions select files for pre-processing, extract main text from each of
# them, save the pre-processed text to the Document Details database, selects
# the texts of interest to the user, and finally enlists a Latent Dirichlet 
# Allocation algorithm to construct maps of common words and phrases within the
# text. Imperfections in text pre-processing come from inconsistent PDF 
# formatting between journal houses and government repositories. However, the
# outcome is a quantitative and visual summary of common topics pertaining to 
# socio-environmental system challenges facing of the Mississippi River Basin.
###############################################################################

# Necessary modules (some may have to be installed first)
import csv
import fitz
import gensim.corpora as corpora
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import shutil
import string
from collections import Counter
from gensim import models
from itertools import chain, combinations
from netgraph import Graph, get_circular_layout, get_bundled_edge_paths
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from operator import itemgetter
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from wordcloud import WordCloud

# =============================================================================
#                         PRE-PROCESSING FUNCTIONS
# =============================================================================

########################## RUN TEXT PRE-PROCESSING ############################

# User is asked whether to pre-process any text first, or progress
# immediately to topic modeling
def preprocessText(values,message):
    while True:
        x = input(message)
        if x in values:
            preprocessText.yesNo = x
            break
        else:
            print("Invalid value: options are " + str(values))
    return preprocessText
preprocessText(["Y","N"],'''\nWould you like to pre-process the PDFs first '''
               '''before topic modeling them? (Y/N): \n''')

########################### CREATE PDF FILE LIST ##############################

# Create a list of all PDF file names from which main text will be
# extracted and pre-processed
def pdfFileList(databaseFilepath):
    
    # User input for whether pre-processing is performed on all or only new PDFs
    # IMPORTANT: Document details must be added to the database first!!!!
    def preprocessNewOnly(values,message):
        while True:
            x = input(message)
            if x in values:
                preprocessNewOnly.yesNo = x
                break
            else:
                print("Invalid value: options are " + str(values))
        return preprocessNewOnly
    preprocessNewOnly(["Y","N"],'''\nWould you like to only pre-process texts '''
                  '''that you haven't added to the database yet? (Y/N): \n''')
    # Assign user input to variable for use in later functions
    global yesNo
    yesNo = preprocessNewOnly.yesNo
    
    # The Document Details database (authors, year, spatial extent,
    # search criteria, main text) is opened as a pandas dataframe; assigned
    # globally for later use
    global database
    database = pd.read_excel(databaseFilepath, index_col = 0)
    
    # Only new files that haven't been pre-processed yet have their names kept
    if yesNo == "Y":
        fileList = []
        for i in range(len(database)):
            if str(database["Filename"][i]) != "nan" and str(database["Preprocessed Text"][i]) == "nan":
                fileList.append(database["Filename"][i])
    
    # Otherwise use all filenames, derived from filepath to the PDFs
    else:
        fileList = database["Filename"].dropna().tolist()
        print("User has decided to pre-process all PDF documents.")
        
    return fileList

########################## PDF TO TEXT CONVERSION ############################

# The original PDF file must be converted into text
def pdfToText(fileToConvert):
    pdfPages = []
    # Iterate through each page in the document
    for page in fileToConvert:
        # Headers and footers also deleted based on falling outside each
        # page's bounding box
        rect = page.rect
        height = 50
        clip = fitz.Rect(20, height, rect.width-20, rect.height-height)
        # Extract the text from each page and add to the empty pdfPages list
        text = page.get_text(clip=clip, flags=fitz.TEXT_PRESERVE_LIGATURES)
        pdfPages.append(text)
    # List of page indices
    pageNumbers = list(range(len(pdfPages)))
    
    # Variable placeholder for contents page
    contentsPages = []    
    # This loop catches every use of the word "Contents" in the main text. Will
    # catch the Table of Contents pages and delete them from the list of page
    # numbers (a specific condition is provided for the Elsevier journals; the
    # other conditions catch use in the main text)
    for i in range(len(pdfPages)):    
        if (re.search(r'\bContents\b', pdfPages[i]) or
            re.search(r'\bCONTENTS\b', pdfPages[i]) or
            re.search(r'\bC O N T E N T S\b', pdfPages[i]) or
            re.search(r'\bTABLEOFCONTENTS\b', pdfPages[i]) or
            re.search(r'\bContents lists available\b'.lower(), pdfPages[i].lower())):
            # If the numbers aren't consecutive, then "Contents" also appeared 
            # in the main text and the loop breaks
            if len(contentsPages) >= 1:
                if i-1 == contentsPages[-1]:
                    contentsPages.append(i)
                else:
                    break
            else:
                contentsPages.append(i)
    # Delete contents page numbers if they only occurred in the main text,
    # assuming that a page number at least one third the size of the total
    # page count is in the main text
    contentsPages = [x for x in contentsPages if (x <= max(pageNumbers)/3)]
    
    # Delete all page numbers prior to (and including) the contents page, but
    # not if it's an Elsevier journal
    if len(contentsPages) > 0 and "Contents lists available" not in pdfPages[max(contentsPages)]:  
        pdfPages = pdfPages[max(contentsPages)+1:]
        pageNumbers = pageNumbers[max(contentsPages)+1:]

    # Same again but this time catching every use of the word "Abstract" in
    # the main text. All page numbers before the page on which the Abstract
    # occurs must be deleted. Usage of the word "abstract" in the main text
    # is skipped
    abstractPages = []
    for i in range(len(pdfPages)):
        if (re.search(r'\bAbstract\b', pdfPages[i]) or
            re.search(r'\bABSTRACT\b', pdfPages[i]) or
            re.search(r'\bA B S T R A C T\b', pdfPages[i]) or
            re.search(r'\ba b s t r a c t\b', pdfPages[i])):
            # Don't save if the word Abstract if it occurs in the second half
            # of the main text
            if i+min(pageNumbers) <= max(pageNumbers)*0.4:
                abstractPages.append(i)

    # Delete all content that comes before the Abstract on the same page,
    # then delete all prior pages
    if len(abstractPages) > 0:
        if "Abstract" in pdfPages[min(abstractPages)]:
            pdfPages[min(abstractPages)] = re.split("Abstract",pdfPages[min(abstractPages)])[1]  
        if "ABSTRACT" in pdfPages[min(abstractPages)]:
            pdfPages[min(abstractPages)] = re.split("ABSTRACT",pdfPages[min(abstractPages)])[1]  
        if "A B S T R A C T" in pdfPages[min(abstractPages)]:
            pdfPages[min(abstractPages)] = re.split("A B S T R A C T",pdfPages[min(abstractPages)])[1]  
        if "a b s t r a c t" in pdfPages[min(abstractPages)]:
            pdfPages[min(abstractPages)] = re.split("a b s t r a c t",pdfPages[min(abstractPages)])[1]  
            
        pdfPages = pdfPages[min(abstractPages):]
        pageNumbers = pageNumbers[min(abstractPages):]
        
    # If there is no abstract, then the same code instead catches every use of
    # "Introduction" in the main text, and deletes page numbers before its
    # first usage after the Contents page. Also must account for the word
    # often appearing in the References list and not as a subheading
    else:
        introPages = []
        for i in range(len(pdfPages)):
            if (re.search(r'\bIntroduction\b', pdfPages[i]) or
                re.search(r'\bINTRODUCTION\b', pdfPages[i]) and not
               re.search(r'\bReferences\b', pdfPages[i]) and not
               re.search(r'\bREFERENCES\b', pdfPages[i])):
               # Introduction may appear in the References list but on a 
               # subsequent page if the list is long enough
               if i+min(pageNumbers) <= max(pageNumbers)*0.8:
                   introPages.append(i)
        # Delete everything before the Intro that's on the same page,
        # then all prior pages
        if len(introPages) > 0:
            if "Introduction" in pdfPages[min(introPages)]:
                pdfPages[min(introPages)] = re.split("Introduction",pdfPages[min(introPages)])[1]
            if "INTRODUCTION" in pdfPages[min(introPages)]:
                pdfPages[min(introPages)] = re.split("INTRODUCTION",pdfPages[min(introPages)])[1]
            pdfPages = pdfPages[min(introPages):]
            pageNumbers = pageNumbers[min(introPages):]
    
    # Next find every occurrence of the word "References" or "Literature Cited"
    # in the main text. Looking to delete all page numbers after the final 
    # usage of that word/phrase.
    referencePages = []
    for i in range(len(pdfPages)):
        if (re.search(r'\bReferences\b', pdfPages[i]) or 
        re.search(r'\bREFERENCES\b', pdfPages[i]) or 
        re.search(r'\br e f e r e n c e s\b', pdfPages[i]) or  
        re.search(r'\bR E F E R E N C E S\b', pdfPages[i]) or   
        re.search(r'\bCited Literature\b', pdfPages[i]) or
        re.search(r'\bCITED LITERATURE\b', pdfPages[i]) or
        re.search(r'\bLiterature Cited\b', pdfPages[i]) or
        re.search(r'\bLiterature cited\b', pdfPages[i]) or
        re.search(r'\bLITERATURE CITED\b', pdfPages[i]) or
        re.search(r'\bLiterature reviewed\b', pdfPages[i]) or
        re.search(r'\bLITERATURE REVIEWED\b', pdfPages[i]) or
        re.search(r'\bReferences Cited\b', pdfPages[i]) or
        re.search(r'\bREFERENCES CITED\b', pdfPages[i]) or
        re.search(r'\bBibliography\b', pdfPages[i]) or
        re.search(r'\bBIBLIOGRAPHY\b', pdfPages[i]) or
        re.search(r'\bSelect Bibliography\b', pdfPages[i]) or
        re.search(r'\bSELECT BIBLIOGRAPHY\b', pdfPages[i])):
            referencePages.append(i)
            
    # Delete all pages following the title of the References section, assumed 
    # to be the largest recorded page number in the list
    if len(referencePages) > 0:
        pdfPages = pdfPages[:max(referencePages)+1]
        pageNumbers = pageNumbers[:max(referencePages)+1]
        # Delete the contents of the References themselves as well
        if "References" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("References",pdfPages[max(referencePages)])[0]
        if "REFERENCES" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("REFERENCES",pdfPages[max(referencePages)])[0]
        if "R E F E R E N C E S" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("R E F E R E N C E S",pdfPages[max(referencePages)])[0]
        if "r e f e r e n c e s" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("r e f e r e n c e s",pdfPages[max(referencePages)])[0]
        if "Cited Literature" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("Cited Literature",pdfPages[max(referencePages)])[0]
        if "CITED LITERATURE" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("CITED LITERATURE",pdfPages[max(referencePages)])[0]
        if "Literature Cited" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("Literature Cited",pdfPages[max(referencePages)])[0]
        if "Literature cited" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("Literature cited",pdfPages[max(referencePages)])[0]
        if "LITERATURE CITED" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("LITERATURE CITED",pdfPages[max(referencePages)])[0]
        if "References Cited" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("References Cited",pdfPages[max(referencePages)])[0]
        if "REFERENCES CITED" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("REFERENCES CITED",pdfPages[max(referencePages)])[0]
        if "Bibliography" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("Bibliography",pdfPages[max(referencePages)])[0]
        if "BIBLIOGRAPHY" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("BIBLIOGRAPHY",pdfPages[max(referencePages)])[0]
        if "Select Bibliography" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("Select Bibliography",pdfPages[max(referencePages)])[0]
        if "SELECT BIBLIOGRAPHY" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("SELECT BIBLIOGRAPHY",pdfPages[max(referencePages)])[0]

    # Delete all Appendices as well. These are usually removed by deleting
    # everything after the References list, but documents do not always
    # contain References, and sometimes Appendices come first. 
    appendixPages = []
    for i in range(len(pdfPages)):
        if (re.search(r'\bAppendix\b', pdfPages[i]) or
            re.search(r'\bAPPENDIX\b', pdfPages[i])):
            # Similar to "Abstract", sometimes "Appendix" occurs in the main
            # text, so skip its usage by only looking for it toward the end
            # of the document
            if i+min(pageNumbers) >= max(pageNumbers)*0.8:
                appendixPages.append(i)
    # Delete all pages following the first Appendix
    if len(appendixPages) > 0:
        pdfPages = pdfPages[:min(appendixPages)+1]
        pageNumbers = pageNumbers[:min(appendixPages)+1]
        # Delete the contents of the Appendix itself as well
        if "Appendix" in pdfPages[min(appendixPages)]:
            pdfPages[min(appendixPages)] = re.split("Appendix",pdfPages[min(appendixPages)])[0]
        if "APPENDIX" in pdfPages[min(appendixPages)]:
            pdfPages[min(appendixPages)] = re.split("APPENDIX",pdfPages[min(appendixPages)])[0]
    
    # Delete the Acknowledgements as well (UK and US English spellings given).
    # Acknowledgements typically occur late in the text, but their page
    # numbers are ignored if they occur early to avoid main text deletion
    acknowPages = []
    for i in range(len(pdfPages)):
        if (re.search(r'\bAcknowledgements\b', pdfPages[i]) or
        re.search(r'\bACKNOWLEDGEMENTS\b', pdfPages[i]) or
        re.search(r'\bA C K N O W L E D G E M E N T S\b', pdfPages[i]) or
        re.search(r'\bAcknowledgments\b', pdfPages[i]) or
        re.search(r'\bACKNOWLEDGMENTS\b', pdfPages[i]) or
        re.search(r'\bA C K N O W L E D G M E N T S\b', pdfPages[i]) or
        re.search(r'\bAcknowledgement\b', pdfPages[i]) or
        re.search(r'\bACKNOWLEDGEMENT\b', pdfPages[i]) or
        re.search(r'\bA C K N O W L E D G E M E N T\b', pdfPages[i]) or
        re.search(r'\bAcknowledgment\b', pdfPages[i]) or
        re.search(r'\bACKNOWLEDGMENT\b', pdfPages[i]) or
        re.search(r'\bA C K N O W L E D G M E N T\b', pdfPages[i])):
            acknowPages.append(i)
            
    # The contents of the Acknowledgements themselves must also be deleted    
    if len(acknowPages) > 0:
        if "Acknowledgements" in pdfPages[max(acknowPages)]:
            pdfPages[max(acknowPages)] = re.split("Acknowledgements",pdfPages[max(acknowPages)])[0]
        if "ACKNOWLEDGEMENTS" in pdfPages[max(acknowPages)]:
            pdfPages[max(acknowPages)] = re.split("ACKNOWLEDGEMENTS",pdfPages[max(acknowPages)])[0]
        if "Acknowledgments" in pdfPages[max(acknowPages)]:
            pdfPages[max(acknowPages)] = re.split("Acknowledgments",pdfPages[max(acknowPages)])[0]
        if "ACKNOWLEDGMENTS" in pdfPages[max(acknowPages)]:
            pdfPages[max(acknowPages)] = re.split("ACKNOWLEDGMENTS",pdfPages[max(acknowPages)])[0]
        if "Acknowledgement" in pdfPages[max(acknowPages)]:
            pdfPages[max(acknowPages)] = re.split("Acknowledgement",pdfPages[max(acknowPages)])[0]
        if "ACKNOWLEDGEMENT" in pdfPages[max(acknowPages)]:
            pdfPages[max(acknowPages)] = re.split("ACKNOWLEDGEMENT",pdfPages[max(acknowPages)])[0]
        if "Acknowledgment" in pdfPages[max(acknowPages)]:
            pdfPages[max(acknowPages)] = re.split("Acknowledgment",pdfPages[max(acknowPages)])[0]
        if "ACKNOWLEDGMENT" in pdfPages[max(acknowPages)]:
            pdfPages[max(acknowPages)] = re.split("ACKNOWLEDGMENT",pdfPages[max(acknowPages)])[0]
            
    # Take the same approach to Author Contributions sections, deleting their
    # contents if found
    authorPages = []
    for i in range(len(pdfPages)):
        if (re.search(r'\bAuthor Contributions\b', pdfPages[i]) or
        re.search(r'\bAuthor contributions\b', pdfPages[i]) or
        re.search(r'\bAUTHOR CONTRIBUTIONS\b', pdfPages[i]) or
        re.search(r'\bAUTHOR INFORMATION\b', pdfPages[i]) or
        re.search(r'\bAuthor contribution statement\b', pdfPages[i]) or
        re.search(r'\bCRediT authorship contribution statement\b', pdfPages[i])):
            authorPages.append(i)
    if len(authorPages) > 0:
        if "Author Contributions" in pdfPages[max(authorPages)]:
            pdfPages[max(authorPages)] = re.split("Author Contributions",pdfPages[max(authorPages)])[0]
        if "Author contributions" in pdfPages[max(authorPages)]:
            pdfPages[max(authorPages)] = re.split("Author contributions",pdfPages[max(authorPages)])[0]
        if "AUTHOR CONTRIBUTIONS" in pdfPages[max(authorPages)]:
            pdfPages[max(authorPages)] = re.split("AUTHOR CONTRIBUTIONS",pdfPages[max(authorPages)])[0]
        if "AUTHOR INFORMATION" in pdfPages[max(authorPages)]:
            pdfPages[max(authorPages)] = re.split("AUTHOR INFORMATION",pdfPages[max(authorPages)])[0]
        if "Author contribution statement" in pdfPages[max(authorPages)]:
            pdfPages[max(authorPages)] = re.split("Author contribution statement",pdfPages[max(authorPages)])[0]
        if "CRediT authorship contribution statement" in pdfPages[max(authorPages)]:
            pdfPages[max(authorPages)] = re.split("CRediT authorship contribution statement",pdfPages[max(authorPages)])[0]

    # Same again if there exists a Data Availability Statement
    dataPages = []
    for i in range(len(pdfPages)):
        if (re.search(r'\bData availability statement\b', pdfPages[i]) or
        re.search(r'\bData Availability Statement\b', pdfPages[i]) or
        re.search(r'\bDATA AVAILABILITY STATEMENT\b', pdfPages[i])):
            dataPages.append(i)
    if len(dataPages) > 0:
        if "Data availability statement" in pdfPages[max(dataPages)]:
            pdfPages[max(dataPages)] = re.split("Data availability statement",pdfPages[max(dataPages)])[0]
        if "Data Availability Statement" in pdfPages[max(dataPages)]:
            pdfPages[max(dataPages)] = re.split("Data Availability Statement",pdfPages[max(dataPages)])[0]
        if "DATA AVAILABILITY STATEMENT" in pdfPages[max(dataPages)]:
            pdfPages[max(dataPages)] = re.split("DATA AVAILABILITY STATEMENT",pdfPages[max(dataPages)])[0]

    # Same again but if there exists a Declaration of Competing Interest
    declarePages = []
    for i in range(len(pdfPages)):
        if (re.search(r'\bDeclaration of conflicting interests\b', pdfPages[i]) or
        re.search(r'\bDeclaration of Conflicting Interests\b', pdfPages[i]) or
        re.search(r'\bDeclaration of conflicting interest\b', pdfPages[i]) or
        re.search(r'\bDeclaration of Competing Interest\b', pdfPages[i]) or 
        re.search(r'\bDeclaration of Competing interest\b', pdfPages[i]) or 
        re.search(r'\bDeclaration of competing interest\b', pdfPages[i]) or
        re.search(r'\bDeclarations\b', pdfPages[i]) or
        re.search(r'\bDisclosure statement\b', pdfPages[i]) or
        re.search(r'\bConflicts of Interest\b', pdfPages[i]) or
        re.search(r'\bCONFLICT OF INTEREST\b', pdfPages[i])):
            declarePages.append(i)
            
    if len(declarePages) > 0:
        if "Declaration of conflicting interests" in pdfPages[max(declarePages)]:
            pdfPages[max(declarePages)] = re.split("Declaration of conflicting interests",pdfPages[max(declarePages)])[0]
        if "Declaration of Conflicting Interests" in pdfPages[max(declarePages)]:
            pdfPages[max(declarePages)] = re.split("Declaration of Conflicting Interests",pdfPages[max(declarePages)])[0]
        if "Declaration of competing interest" in pdfPages[max(declarePages)]:
            pdfPages[max(declarePages)] = re.split("Declaration of competing interest",pdfPages[max(declarePages)])[0]
        if "Declaration of Competing Interest" in pdfPages[max(declarePages)]:
            pdfPages[max(declarePages)] = re.split("Declaration of Competing Interest",pdfPages[max(declarePages)])[0]
        if "Declaration of Competing interest" in pdfPages[max(declarePages)]:
            pdfPages[max(declarePages)] = re.split("Declaration of Competing interest",pdfPages[max(declarePages)])[0]
        if "Declarations" in pdfPages[max(declarePages)]:
            pdfPages[max(declarePages)] = re.split("Declarations",pdfPages[max(declarePages)])[0]
        if "Disclosure statement" in pdfPages[max(declarePages)]:
            pdfPages[max(declarePages)] = re.split("Disclosure statement",pdfPages[max(declarePages)])[0]
        if "Conflict of Interest" in pdfPages[max(declarePages)]:
            pdfPages[max(declarePages)] = re.split("Conflict of Interest",pdfPages[max(declarePages)])[0]
        if "CONFLICT OF INTEREST" in pdfPages[max(declarePages)]:
            pdfPages[max(declarePages)] = re.split("CONFLICT OF INTEREST",pdfPages[max(declarePages)])[0]

    # Join the pages together as one long string and then split on new line
    # (\n) characters
    extractedText = ", ".join(pdfPages).splitlines()

    return extractedText
    
############################ DELETE UNWANTED LINES ############################

# Lines of the extracted text that do not have a meaning pertinent to the main 
# text content must be deleted. Examples include table rows, stray 
# letters/numbers/characters on their own lines, empty lines, and errors in
# PDF to text extraction
def delUnwantedLines(extractedText):
    
    # Delete leading and trailing whitespace from each line first
    whiteStrip = (x.strip() for x in extractedText)
    extractedText = [line for line in whiteStrip if line]
    
    # Truncate all whitespace on each line to a single space
    extractedText = [" ".join(line.split()) for line in extractedText]
    
    # Delete comma and space at the start of some lines; these indicated 
    # detected new pages for tables and headers when the PDF to text 
    # conversion happened
    extractedText = [line.removeprefix(", ") for line in extractedText]

    # Use list comprehension to delete unwanted lines
    
    # 1) Remove lines only one character in length, likely to be page numbers,
    #    super/subscript characters, and table cell text
    extractedText = [line for line in extractedText if not len(line) == 1]
    
    # 2) Remove lines that consist solely of numbers, since these are also
    #    likely to be page numbers, super/subscript, and cell text
    def isFloat(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    extractedText = [line for line in extractedText if not isFloat(line)]
    
    # 3) Remove lines containing an "=" sign, since these are almost certainly
    #    equations and thus not essential to the main text's meaning
    extractedText = [line for line in extractedText if "=" not in line]
    
    # 4) Remove lines made up only of numbers, whitespace, and punctuation,
    #    which are likely the start of bullet pointed lists, table cells,
    #    and numbers broken up by commas
    def isSpace(s): 
        if ' ' in s: 
           return True
        else: 
           return False
    extractedText = [line for line in extractedText if not all(x.isdigit() or isSpace(x) or x in string.punctuation for x in line)]

    # 5) Remove any line that contains the following characters, since these
    #    are almost always author affiliations/details and headers/footers that
    #    were missed during the initial extraction
    extractedText = [line for line in extractedText if "doi:" not in 
                      line.lower() and "http" not in line.lower() and "www." 
                      not in line.lower() and "journal of" not in line.lower()
                      and "@" not in line and ".gov" not in line.lower()
                      and ", USA" not in line and "10.10" not in
                      line.lower() and "10.11" not in line.lower() and "©" not
                      in line.lower() and "Department of" not in line.lower() 
                      and "e-mail" not in line.lower()
                      and "fax" not in line.lower()
                      and "Tel:" not in line.lower()
                      and "all rights reserved" not in line.lower()
                      and "phone" not in line.lower()]

    # 6) Remove lines that are duplicates, since these are very likely table
    #    rows from the same/different tables, accidental duplications of the 
    #    main text from the PDF to text conversion, and can also remove any 
    #    remaining headers and footers
    extractedText = [i for n, i in enumerate(extractedText) if i not in extractedText[:n]]
    
    # To finish, PDF documents often use ligature characters, which should be
    # converted into standard Latin characters
    extractedText = [line.replace("ﬂ","fl") for line in extractedText]
    extractedText = [line.replace("ﬀ","ff") for line in extractedText]
    extractedText = [line.replace("ﬁ","fi") for line in extractedText]
    extractedText = [line.replace("ﬃ","ffi") for line in extractedText]
    extractedText = [line.replace("ﬄ","ffl") for line in extractedText]
    # Punctuation should also be standardized for later removal and inclusion
    extractedText = [line.replace("–","-") for line in extractedText]
    extractedText = [line.replace("‐","-") for line in extractedText]
    extractedText = [line.replace("‑","-") for line in extractedText]
    # Do the same thing with accented characters that appear frequently
    extractedText = [line.replace("ñ","n") for line in extractedText]
    extractedText = [line.replace("é","e") for line in extractedText]
    return extractedText

########################### DELETE TEXT INSIDE LINES ##########################

# Parts of text inside each line are now deleted too. This includes numbers,
# text inside parentheses, and any remaining unwanted text
def delInsideLines(extractedText):
    
    # All numbers in each line are deleted first
    extractedText = [re.sub(r'[0-9]+', '', line) for line in extractedText]
    
    # All text inside parentheses is deleted next
    extractedText = [re.sub('\(.*?\)','', line) for line in extractedText]
    # Parentheses often stretch across lines, so delete everything before a
    # closing parenthesis
    for i in range(len(extractedText)):
        if ")" in extractedText[i]:
            extractedText[i] = extractedText[i].split(")")[1]
        if "(" in extractedText[i]:
            extractedText[i] = extractedText[i].split("(")[0]
  
    # Hyphens at the end of lines almost always represent a single word written
    # across two lines. Each time this happens, the two lines are joined
    # together and the hyphen is deleted. Ignore floating hyphens.
    for line in extractedText:
        if line.endswith("-") and not line.endswith(" -"):
            index = extractedText.index(line)
            try:
               extractedText[index] = extractedText[index][:-1] + extractedText[index+1]
               extractedText.remove(extractedText[index+1])
            # Don't attempt if the final line of the main text ends with a hyphen
            except:
                break

    # Any text in each line that isn't a Latin character, period, hyphen, or 
    # whitespace is deleted
    extractedText = [re.sub(r'[^a-zA-Z .-]', '', line) for line in extractedText]
    
    # Delete any remaining lines that are a repeat of another line in the text;
    # this particularly targets repeating table rows once numbers and text in
    # parentheses have been removed
    extractedText = [line for line,elem in itertools.groupby(extractedText)]
    extractedText = '\n'.join(extractedText)    
    
    return extractedText

########################### TOKENIZE AND REMOVE ###############################

# By first defining new lines by sentences and then tokenizing the text, other
# undesired lines and words can be removed, such as stopwords, in-text 
# citations, people's names, and remaining unwanted sections.
def tokenizeAndRemove(extractedText,stopwordsFilePath):
    
    # Split text on periods instead of new line characters, so each list
    # element is now a sentence from the main text
    extractedText = extractedText.replace("\n"," ").split(".")
    
    # Remove any white space that was left behind by the within-line text
    # deletion in the previous function
    extractedText = [" ".join(line.split()) for line in extractedText]
    
    # Delete sentences that are four characters or shorter. These are usually
    # sentences that were either deleted completely by the previous functions, 
    # units, or table cell entries
    extractedText = [line for line in extractedText if not len(line) <= 4]
    
    # Delete sentences that start with a specific word(s) that represents an
    # unwanted section that still exists in the main text
    extractedText = [line for line in extractedText if not line.startswith("All authors have read and agreed")
                      and not line.startswith("All rights reserved")
                      and not line.startswith("ARTICLE HISTORY")
                      and not line.startswith("Author Contributions")
                      and not line.startswith("BioOne sees sustainable")
                      and not line.startswith("Citation")
                      and not line.startswith("Commercial inquiries")
                      and not line.startswith("Contents list available")
                      and not line.startswith("Correspondence to")
                      and not line.startswith("Corresponding author")
                      and not line.startswith("Data Availability Statement")
                      and not line.startswith("Declaration of Competing Interest")
                      and not line.startswith("Declaration of conflicting interest")
                      and not line.startswith("Funding")
                      and not line.startswith("Full Terms")
                      and not line.startswith("Informed Consent Statement")
                      and not line.startswith("Institutional Review Board")
                      and not line.startswith("Journal of")
                      and not line.startswith("Key Points")
                      and not line.startswith("KEY WORDS")
                      and not line.startswith("Key Words")
                      and not line.startswith("Key words")
                      and not line.startswith("Keywords")
                      and not line.startswith("No part of this periodical")
                      and not line.startswith("Open Access")
                      and not line.startswith("Page number")
                      and not line.startswith("Posted online")
                      and not line.startswith("Published in")
                      and not line.startswith("Published online")
                      and not line.startswith("Submit your article")
                      and not line.startswith("Supplemental Material")
                      and not line.startswith("SUPPLEMENTARY MATERIAL")
                      and not line.startswith("Supplementary Information")
                      and not line.startswith("Supporting information")
                      and not line.startswith("This manuscript was submitted on")
                      and not line.startswith("Your use of this PDF")]
 
    # Rejoin and then tokenize the text
    extractedText = ' '.join(extractedText) 
    tokens = WhitespaceTokenizer().tokenize(extractedText)
    
    # If a token ends with a hyphen, the next token is almost always the 
    # remainder of the same word. Join these tokens together
    for token in tokens:
        if token.endswith("-") and token.count("-") == 1 and len(token) > 1:
            index = tokens.index(token)
            try:
                tokens[index] = tokens[index][:-1] + tokens[index+1]
                tokens.remove(tokens[index+1])
            # Don't attempt if the final token ends with a hyphen
            except:
                break
    
    citedList = []
    # If the tokens "et" and "al" appear as subsequent tokens, append them
    # as well as the previous word (the name of the person being cited)
    for i in range(len(tokens)):
        if tokens[i] == "al" and tokens[i-1] == "et":
            citedList.append(tokens[i-2])
            citedList.append(tokens[i-1])
            citedList.append(tokens[i])
    # Delete these in-text citations from the list of tokens
    tokens = [token for token in tokens if token not in citedList]
    
    # A collection of stopwords are added to a list by reading its csv file first
    def readInCsv(csvFile):
            with open(csvFile, 'r', encoding = 'utf-8', errors = "ignore") as fp:
                reader = csv.reader(fp, delimiter = ',', quotechar = '"')
                dataRead = [row for row in reader]
            return dataRead
    def getStopwords():
        stopwords = readInCsv(stopwordsFilePath)
        stopwords = [word[0] for word in stopwords]
        return stopwords
    # Calling the above functions fills the list of stopwords
    stopwords = getStopwords()

    # Remove these stopwords from the tokenized text
    tokens = [token for token in tokens if token not in stopwords]
    
    # The remaining tokens must be lemmatized as the final pre-processing step
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatization is performed on nouns, verbs, adjectives, and
    # adverbs separately
    lemmaN = [lemmatizer.lemmatize(word, 'n') for word in tokens]
    lemmaNV = [lemmatizer.lemmatize(word, 'v') for word in lemmaN]
    lemmaNVA = [lemmatizer.lemmatize(word, 'a') for word in lemmaNV]
    lemmaNVAR = [lemmatizer.lemmatize(word, 'r') for word in lemmaNVA]
  
    text = " ".join(lemmaNVAR)    
    return text

########################## APPEND AND SAVE TEXT ###############################

# Once the text has been pre-processed, it must be added to the Preprocessed
# Text column in the Document Details database
def appendAndSave(savedDatabase,finalTexts):

    # If pre-processing all documents, blank list elements must be added to the 
    # finalTexts list, corresponding to absent texts in the database
    if yesNo == "N":
        for i in range(len(database)):    
            if str(database["Filename"][i]) == "nan":
                finalTexts.insert(i,"")
    # If pre-processing only new documents, add them to the end of the already
    # pre-processed texts
    else:
        finalTexts = database["Preprocessed Text"][:-len(finalTexts)].tolist() + finalTexts
    
    # Add the pre-processed text to the database and save the result
    database["Preprocessed Text"] = finalTexts
    database.to_excel(savedDatabase)
    
    return database

# =============================================================================
#                    LATENT DIRICHLET ALLOCATION FUNCTIONS
# =============================================================================

############################ OPEN AS DATAFRAME ################################

# If the user decides to work with the Document Details database without any 
# new pre-processing, the database is opened as is, as a pandas dataframe
def openDocumentDetails(filepath):
    # The database must be assigned as a global variable if accessed without
    # pre-processing first as well
    global database
    database = pd.read_excel(filepath, index_col = 0)
    return database

######################## SETTING UP THE TEXT SELECTION ########################

# The desired texts for topic modeling must be selected by the user
def textSelection(database):
    
    # This list will be filled with indices of database rows corresponding to 
    # specified user inputs below
    textIndices = []
    
    # First ask whether user wants all texts or a specific state/sub-basin/decade.
    def scopeOfTexts(values,message):
        while True:
            x = input(message)
            if x in values:
                scopeOfTexts.scope = x
                break
            else:
                print("Invalid value: options are " + str(values))
    scopeOfTexts(["All","State","Sub-Basin","Decade"],'''\nSpecify desired scope '''
                  '''of texts (All, State, Sub-Basin, Decade): \n''')
    
    # Make the selected texts a global variable for building word clouds later
    global textsOfInterest
    
    # A second user input specifies which specifc state, sub-basin, or decade is
    # of interest for running the topic model
    def textsOfInterest(values,message):
        while True:
            x = input(message)
            if x in values:
                textsOfInterest.texts = x
                break
            else:
                print("Invalid value: options are " + str(values))
    
    # Specific argument for the second user input, and collection of corresponding
    # indices from the database. First for when the input is "State"
    if scopeOfTexts.scope == "State":
        textsOfInterest(["Alabama","Arkansas","Colorado","Georgia","Illinois","Indiana",
                         "Iowa","Kansas","Kentucky","Louisiana","Maryland","Minnesota",
                         "Mississippi","Missouri","Montana","Nebraska","New Mexico",
                         "New York","North Carolina","North Dakota","Ohio","Oklahoma",
                         "Pennsylvania","South Dakota","Tennessee","Texas","Virginia",
                         "West Virginia","Wisconsin","Wyoming"], 
                         '''\nSpecify desired state (Alabama, Arkansas, Colorado, '''
                         '''Georgia, Illinois, Indiana, Iowa, Kansas, Kentucky, '''
                         '''Louisiana, Maryland, Minnesota, Mississippi, Missouri, '''
                         '''Montana, Nebraska, New Mexico, New York, North Carolina, '''
                         '''North Dakota, Ohio, Oklahoma, Pennsylvania, South Dakota, '''
                         '''Tennessee, Texas, Virginia, West Virginia, Wisconsin, '''
                         '''Wyoming): \n''')
    
        # Loop through the database and append target indices to the empty list
        for i in range(len(database)):
            if database["State(s)"][i] is not np.nan:
                if textsOfInterest.texts in database["State(s)"][i]:
                    textIndices.append(database.index[i])
    
    # Same again but if the input is "Sub-Basin". Must account for there also
    # being two states named "Missouri" and "Ohio" in the user input
    elif scopeOfTexts.scope == "Sub-Basin":
        textsOfInterest(["Arkansas-Red","Lower Mississippi","Missouri (Basin)",
                         "Ohio (Basin)","Upper Mississippi"],
                        '''\nSpecify desired sub-basin (Arkansas-Red, Lower Mississippi, '''
                        '''Missouri (Basin), Ohio (Basin), Upper Mississippi): \n''')
        
        for i in range(len(database)):
            if database["River/Sub-Basin(s)"][i] is not np.nan:
                if textsOfInterest.texts in database["River/Sub-Basin(s)"][i]:
                    textIndices.append(database.index[i])
            
    # If the input is "Decade", the database is indexed for all years that fall
    # within each decade
    elif scopeOfTexts.scope == "Decade":
        textsOfInterest(["1990s","2000s","2010s","2020s"],
                        '\nSpecify desired decade (1990s, 2000s, 2010s, 2020s): \n''')
        
        # The 1990s
        if textsOfInterest.texts == "1990s":
            textIndices = database.index[(database["Year"]==1990) 
                                              | (database["Year"]==1991)
                                              | (database["Year"]==1992) 
                                              | (database["Year"]==1993) 
                                              | (database["Year"]==1994) 
                                              | (database["Year"]==1995) 
                                              | (database["Year"]==1996) 
                                              | (database["Year"]==1997) 
                                              | (database["Year"]==1998) 
                                              | (database["Year"]==1999)].tolist()
    
        # The 2000s
        elif textsOfInterest.texts == "2000s":
            textIndices = database.index[(database["Year"]==2000) 
                                              | (database["Year"]==2001)
                                              | (database["Year"]==2002) 
                                              | (database["Year"]==2003) 
                                              | (database["Year"]==2004) 
                                              | (database["Year"]==2005) 
                                              | (database["Year"]==2006) 
                                              | (database["Year"]==2007) 
                                              | (database["Year"]==2008) 
                                              | (database["Year"]==2009)].tolist()
        # The 2010s
        elif textsOfInterest.texts == "2010s":
            textIndices = database.index[(database["Year"]==2010) 
                                              | (database["Year"]==2011)
                                              | (database["Year"]==2012) 
                                              | (database["Year"]==2013) 
                                              | (database["Year"]==2014) 
                                              | (database["Year"]==2015) 
                                              | (database["Year"]==2016) 
                                              | (database["Year"]==2017) 
                                              | (database["Year"]==2018) 
                                              | (database["Year"]==2019)].tolist()
            
        # The 2020s
        elif textsOfInterest.texts == "2020s":
            textIndices = database.index[(database["Year"]==2020) 
                                              | (database["Year"]==2021)
                                              | (database["Year"]==2022) 
                                              | (database["Year"]==2023)].tolist()
                    
    # If the user selects "All" then the indices for all texts are extracted
    else:
        textIndices = database.index[(database["Text ID"] != np.nan)].tolist()
        textsOfInterest.texts = "Basin-Wide"

    # Reduce database down to rows containing the desired text
    database = database.loc[textIndices]
        
    # Pre-processing can sometimes remove all main text, leaving np.nan in the
    # "Preprocessed Text" column. Remove these rows from the dataframe
    database = database[database["Preprocessed Text"].notna()]

    # Extract the desired text from the database
    global textsForTraining
    textsForTraining = database["Preprocessed Text"].tolist()

    # Will also need titles, citations, and URLs for constructing the
    # document-topic density table in a later function
    global titles, citations, urls
    titles = database["Document Title"].tolist()
    citations = database["Citations"].tolist()
    urls = database["URL"].tolist()
        
    # Need to be able to access the selected text
    return textsForTraining

############################## CREATE THE CORPUS ##############################

# All words that exist in the selected text must be combined into a single
# corpus, one big dataset comprised of ngrams extracted from the text
def createCorpus(selectedTexts,filepath):
    
    # Make n-gram size specified below global for final text output
    global ngramSize
    
    # The user is asked how large the extracted n-grams from the text can be
    # when creating the corpus (1 = unigrams only, 2 = unigrams and bigrams,
    # 3 = uni, bi, and trigrams, etc.)
    def ngramSize(values,message):
        while True:
            x = input(message)
            if x in values:
                ngramSize.choice = x
                break
            else:
                print("Invalid value: options are " + str(values))
        return ngramSize
    ngramSize(["1","2","3","4","5","6","7","8","9","10"],'''\nWhat maximum n-gram size would you like '''
                    '''to use for creating the training corpus? (Choose from 1 to 10): \n''')
    
    # This list will hold a list of lists of the training corpus, each element
    # containing n-grams derived from each selected text. This is made global
    # for later use
    global trainingCorpus
    trainingCorpus = []
         
    # Loop through the texts for training the model
    for i in range(len(textsForTraining)):
        singleText = [textsForTraining[i]]
        
        # Construct ngrams from the text (uni, bi, tri, quad) and append them
        # to the list of lists
        count_vect = CountVectorizer(ngram_range=(1,int(ngramSize.choice)))
        count_vect.fit_transform(singleText)
        ngrams = count_vect.get_feature_names_out().tolist()
        trainingCorpus.append(ngrams)

    # Words two characters or shorter in length must be removed from the
    # training corpus
    trainingCorpus = [[i for i in nested if len(i) > 2] for nested in trainingCorpus]
 
    # Unnest the list, these will be used to create a word cloud of the most
    # common n-grams in the training corpus
    textWordCloud = [val for sublist in trainingCorpus for val in sublist]
    
    # Create a dictionary of ngrams ordered by frequency
    wordCloudList = Counter(textWordCloud)

    # Make choice to remove commonest n-grams global for final text output
    global removeCommonNgrams
    
    # User input for whether the 100 commonest n-grams should be pre-emptively
    # removed from the training corpus before word cloud creation or LDA
    # algorithm training
    def removeCommonNgrams(values,message):
        while True:
            x = input(message)
            if x in values:
                removeCommonNgrams.yesNo = x
                break
            else:
                print("Invalid value: options are " + str(values))
        return removeCommonNgrams.yesNo
    removeCommonNgrams(["Y","N"],'''\nWould you like to pre-emptively remove '''
                  '''the 100 commonest n-grams from the training corpus? (Y/N): \n''')

    # Remove the n-grams if the user specified as such in the user input
    if removeCommonNgrams.yesNo == "Y":
        
        # The 100 commonest ngrams in all 2,158 PDFs. These words potentially
        # conceal spatiotemporally distinct research priorities
        commonestToRemove = ["river","mississippi","mississippi river","water",
                              "area","high","low","large","data","time","increase",
                              "analysis","result","system","range","year","indicate",
                              "great","occur","change","long","level","present",
                              "similar","determine","number","value","small",
                              "location","report","condition","compare","effect",
                              "period","first","represent","upper","different",
                              "measure","average","estimate","site","important",
                              "available","significant","term","flow",
                              "information","limit","reduce","process","state",
                              "difference","associate","potential","source", "point",
                              "remain","surface","observe","identify","sample",
                              "collect","factor","describe","develop","rate","vary",
                              "mean","natural","basin","affect","major","model",
                              "control","likely","scale","example","decrease",
                              "cause","distribution","reach","size","relatively",
                              "early","environmental","type","management","select",
                              "new","generally","field","survey","relative","require",
                              "specific","research","land","quality","individual"]
        
        # These loops remove these ngrams from both wordCloudList
        # and trainingCorpus
        for word in list(wordCloudList):
            if word in commonestToRemove:
                del wordCloudList[word]
        for sublist in trainingCorpus:
            for word in sublist:
                if word in commonestToRemove:
                    del trainingCorpus[trainingCorpus.index(sublist)][sublist.index(word)]
        
    # Construct a word cloud of the 100 most common n-grams
    wordCloud = WordCloud(background_color="white",colormap="plasma",collocations=False, contour_width=10,
                          mask=np.array(Image.open(filepath + "Word Cloud Templates/" + textsOfInterest.texts + ".jpg")),
                          max_words=100).generate_from_frequencies(wordCloudList)
    # Make common n-grams global for final text output
    global commonNgrams
    commonNgrams = wordCloudList.most_common(100)
    
    # Plot and display the word cloud
    fig = plt.figure()
    fig.text(x=0.5,y=0.08,s="Word Cloud: " + textsOfInterest.texts,fontsize=16,
              fontstyle="italic", horizontalalignment='center')
    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(filepath + "/Model Training Results/" + textsOfInterest.texts + "/Full Corpus.jpg",dpi=300)
    
    return trainingCorpus

########################### LDA ALGORITHM TRAINING ############################

# We enlist a Latent Dirichlet Allocation algorithm to probabilistically 
# identify the likeliest n-grams (topics) by topic (document).
# User input is included to allow users to specify this model's 
# (hyper)parameters and later perform sensitivity analysis on the model's output
def trainLDAAlgorithm(trainingCorpus, filepath):
    
    # Map all n-grams in the training corpus onto IDs
    ngramIDs = corpora.Dictionary(trainingCorpus)
    
    # Convert each n-gram into a number using the doc2bow function, creating a 
    # "bag of words"
    bagOfWords = [ngramIDs.doc2bow(text) for text in trainingCorpus]

    # Make choice to not apply TF-IDF below global for final text output
    global useTFIDF
    
    # User input for whether a TF-IDF (Term Frequency-Inverse Document
    # Frequency) correction should be used to weight n-grams by their frequency
    # of occurrence
    def useTFIDF(values,message):
        while True:
            x = input(message)
            if x in values:
                useTFIDF.yesNo = x
                break
            else:
                print("Invalid value: options are " + str(values))
        return useTFIDF.yesNo
    useTFIDF(["Y","N"],'''\nWould you like to use a TF-IDF algorithm '''
                  '''to weight n-grams by their frequency in the training corpus? (Y/N): \n''')

    # Apply the TF-IDF correction to the training corpus if the user wants it;
    # lower n-gram frequency means greater assigned weight
    global corpus
    if useTFIDF.yesNo == "Y":
        tfidf = models.TfidfModel(bagOfWords)
        corpus = tfidf[bagOfWords]
    else:
        corpus = bagOfWords

    # The four key parameters of the LDA algorithm (number of topics, number of
    # random words that seed each topic, and the alpha and eta hyperparameters)
    # must be calibrated first before training the model on the corpus. 
    
    # Start with number of topics. Trial different numbers to find what 
    # maximizes both the coherence and Jaccard similarity scores of the trained
    # LDA algorithm.   
    def calibrateNumTopics(topicRange, filepath):
        # The number of model runs depends on the number of topics tested, 
        # which is capped at 6 in order to avoid the unintentional creation
        # of "junk topics"      
        iterable = list(range(1,8))

        # Empty dictionaries to hold the results of training the LDA 
        # algorithm for each number of topics
        ldaModels = {}
        ldaTopics = {}
        
        print("\nCalibrating each potential number of topics:")
        for i in tqdm(iterable):

            # First train the algorithm itself
            ldaModels[i] = models.LdaModel(corpus=corpus,
                                      id2word=ngramIDs,
                                      num_topics=i,                                     
                                      eval_every=None)
            # Extract topics from the LDA algorithm along with the 20 most 
            # likely occurring words in each topic
            shownTopics = ldaModels[i].show_topics(num_topics=i,num_words = 20,
                                                    formatted=False)
            ldaTopics[i] = [[word[0] for word in topic[1]] for topic in shownTopics]

        # This function calculates a Jaccard similarity score for each pair
        # of topics, assessing how similar the words in each topic are
        def jaccardSimilarity(topic1, topic2):
            # Sum of words that occur in both topic lists, divided by sum
            # of unique words in both lists
            intersection = set(topic1).intersection(set(topic2))
            union = set(topic1).union(set(topic2))
            return float(len(intersection))/float(len(union))  
        
        # Stability of topics is assessed through pairwise calculation of
        # Jaccard similarity for all possible pairs of topics
        ldaStability = {}
        for i in range(0, len(iterable)-1):
            jaccardSims = []
            # Pick a topic
            for t1, topic1 in enumerate(ldaTopics[iterable[i]]):
                sims = []
                # Compare that topic pairwise to each other topic
                for t2, topic2 in enumerate(ldaTopics[iterable[i+1]]):
                    sims.append(jaccardSimilarity(topic1, topic2))    
                jaccardSims.append(sims)    
            # Append the computed Jaccard similarity scores to the dictionary
            ldaStability[iterable[i]] = jaccardSims
            
        # Compute the mean Jaccard similarity for each number of topics, then
        # normalize them
        meanJaccards = [np.array(ldaStability[i]).mean() for i in iterable[:-1]] 
        normalJaccards = (meanJaccards - np.mean(meanJaccards))/np.std(meanJaccards)
        
        # Topic coherence is computed for all LDA models and numbers of topics,
        # and again normalize them
        coherences = [models.CoherenceModel(model=ldaModels[i], texts=trainingCorpus, 
                      dictionary=ngramIDs, coherence='u_mass').get_coherence()
                      for i in iterable[:-1]]
        normalCoherences = (coherences - np.mean(coherences))/np.std(coherences)
                
        # Compute the calibrated number of topics, that which maximizes coherence
        # and minimizes Jaccard similarity
        coherenceMinusJaccard = [normalCoherences - normalJaccards for normalCoherences,normalJaccards in zip(normalCoherences,normalJaccards)]
        maxDifference = max(coherenceMinusJaccard)
        # Iterate through all differences in case two or more are the same
        maxIndexes = [i for i, j in enumerate(coherenceMinusJaccard) if j == maxDifference]
        idealTopicIndex = maxIndexes[0]
        idealNumTopics = iterable[idealTopicIndex]
                
        # Create a plot of normalized Jaccard similarity scores against
        # coherence scores, showing the calibrated number of topics to use
        plt.figure(figsize=(15,10))
        ax = sns.lineplot(x=iterable[:-1], y=normalJaccards, label='Mean Jaccard Similarity', color="orange")
        ax = sns.lineplot(x=iterable[:-1], y=normalCoherences, label='Topic Coherence', color="blue")
        ax.axvline(x=idealNumTopics, label='Calibrated Number of Topics', color='black', ls = "--")
        ax.axvspan(xmin=idealNumTopics - 1, xmax=idealNumTopics + 1, alpha=0.5, facecolor='grey')    
        ax.set_xlim([0.5, 6.5])
        ax.axes.set_title('Normalized Metrics: Coherence & Similarity', fontsize=30)
        ax.set_ylabel('Standard Score', fontsize=20)
        ax.set_xlabel('Number of Topics', fontsize=20)
        ax.locator_params(axis='x',nbins=20)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.legend(fontsize=18,loc='upper center',bbox_to_anchor=(0.5, -0.08),fancybox=True,ncol=3)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.savefig(filepath + "/Model Training Results/" + textsOfInterest.texts + "/Calibrated Number of Topics.png",dpi=300)
        
        return idealNumTopics        

    # The calibrated number of topics for the LDA algorithm will also be needed
    # for other functions
    global numberOfTopics
    numberOfTopics = calibrateNumTopics(trainingCorpus, filepath)

    # The coherence metric alone is used to calibrate values for the random
    # seed size, and the alpha and eta hyperparameters. random_state ranges
    # from 0 to the length of the unnested training corpus (in multiples of 
    # 1000). Based on gensim documentation, Griffiths and Steyvers (2004), and
    # Steyvers et al. (2007), default values of alpha and eta of 50/numTopics
    # and 0.1, respectively, users can pick these or calibrate their own.
    def calibrateSeedAlphaEta(parameter):
        
        # Make choice to use default alpha and eta below global for final text output
        global useDefaultAlpha
        global useDefaultEta
        
        # User input for using default alpha and eta values if so desired
        if parameter == "alpha":
            def useDefaultAlpha(values,message):
                while True:
                    x = input(message)
                    if x in values:
                        useDefaultAlpha.yesNo = x
                        break
                    else:
                        print("Invalid value: options are " + str(values))
                return useDefaultAlpha.yesNo
            useDefaultAlpha(["Y","N"],'''\nWould you like to use the default value for ''' 
                          '''alpha (50/numTopics) to train the LDA algorithm? (Y/N): \n''')
            
            # Alpha takes 50/numberOfTopics as a default, or try 1 to 50 in
            # increments of 1
            if useDefaultAlpha.yesNo == "Y":
                alphaValue = 50/numberOfTopics
                return alphaValue
            else:
                valueList = list(range(1,51,1))
        # Same as above but for the value of eta
        elif parameter == "eta":
            def useDefaultEta(values,message):
                while True:
                    x = input(message)
                    if x in values:
                        useDefaultEta.yesNo = x
                        break
                    else:
                        print("Invalid value: options are " + str(values))
                return useDefaultEta.yesNo
            useDefaultEta(["Y","N"],'''\nWould you like to use the default value for ''' 
                          '''eta (0.1) to train the LDA algorithm? (Y/N): \n''')
            # Eta takes 0.1 as a default, or try 0 to 1 in increments of 0.02
            if useDefaultEta.yesNo == "Y":
                etaValue = 0.1
                return etaValue
            else:
                valueList = [x * 0.02 for x in range(0, 51)]
                
        # Provide a list of values for when this function is used to calibrate
        # the seed size
        if (parameter == "seed size"):
            valueList = list(range(0,len(list(chain.from_iterable(trainingCorpus))),int(len(list(chain.from_iterable(trainingCorpus)))/100)))

        # If a list of values exists, then the parameter is calibrated
        if valueList:
            # Empty dictionary to hold each trained LDA algorithm
            ldaModels = {}
            
            print("\nCalibrating each potential " + parameter + ":")
            for i in tqdm(valueList):

                # First train the model itself
                if parameter == "seed size":
                    ldaModels[i] = models.LdaModel(corpus=corpus, 
                                                    id2word=ngramIDs,
                                                    num_topics=numberOfTopics,
                                                    eval_every=None,
                                                    random_state=i)
                elif parameter == "alpha":
                    ldaModels[i] = models.LdaModel(corpus=corpus, 
                                                    id2word=ngramIDs,
                                                    num_topics=numberOfTopics,
                                                    eval_every=None,
                                                    alpha=i)
                elif parameter == "eta":
                    ldaModels[i] = models.LdaModel(corpus=corpus, 
                                                    id2word=ngramIDs,
                                                    num_topics=numberOfTopics,
                                                    eval_every=None,
                                                    eta=i)
            
            # Calculate topic coherence for all LDA models and normalize the
            # obtained scores
            coherences = [models.CoherenceModel(model=ldaModels[i], texts=trainingCorpus, 
                          dictionary=ngramIDs, coherence='u_mass').get_coherence()\
                          for i in valueList[:-1]]
            normalCoherences = (coherences - np.mean(coherences))/np.std(coherences)
            normalCoherences = normalCoherences.tolist()

            # Find index for which coherence maximizes and use to index the
            # list of tested parameter values
            idxMaxCoherence = normalCoherences.index(max(normalCoherences))
            calibValue = valueList[idxMaxCoherence]

            def plotParam(paramName,span,filepath,xData,calibValue):
            
                # Create the same plots as for number of topics but this time
                # for change in coherence for different seed sizes, alphas, and etas
                plt.figure(figsize=(15,10))
                ax = sns.lineplot(x=xData[:-1], y=normalCoherences, label='Topic Coherence', color="blue")
                ax.axvline(x=calibValue, label='Calibrated ' + paramName, color='black', ls = "--")
                ax.axvspan(xmin=calibValue - span, xmax=calibValue + span, alpha=0.5, facecolor='grey')    
                ax.set_xlim([0, xData[-1]])
                ax.axes.set_title('Normalized Coherence: ' + paramName, fontsize=30)
                ax.set_ylabel('Standard Score', fontsize=20)
                ax.set_xlabel(paramName, fontsize=20)
                ax.locator_params(axis='x',nbins=10)
                ax.legend(fontsize=18,loc='upper center',bbox_to_anchor=(0.5, -0.08),fancybox=True,ncol=2)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.savefig(filepath + "/Model Training Results/" + textsOfInterest.texts+ "/Calibrated " + paramName + ".png",dpi=300)

            
            # Crete the plots and return the calibrated parameters. For seed
            # size, the calibrated value and x-axis are plotted as percentages
            # of the number of n-grams in the corpus
            if parameter == "seed size":
                
                # Make the percentage global for use in the final text output
                global calibToPlot
                seedPercent = list(range(0,101))
                calibToPlot = seedPercent[idxMaxCoherence]
                
                plotParam("Percent of Corpus Seeded",1,filepath,seedPercent,calibToPlot)
                seedSize = calibValue
                return seedSize
            
            elif parameter == "alpha":
                plotParam("Alpha",1,filepath,valueList,calibValue)
                alphaValue = calibValue
                return alphaValue
            
            elif parameter == "eta":
                plotParam("Eta",0.02,filepath,valueList,calibValue)
                etaValue = calibValue
                return etaValue
            
        else:
            print("User chose default value for", parameter)

    # Make these calibrated values global for final text output
    global seedSize, alphaValue, etaValue
    seedSize = calibrateSeedAlphaEta("seed size")
    alphaValue = calibrateSeedAlphaEta("alpha")
    etaValue = calibrateSeedAlphaEta("eta")
    
    # The four (hyper)parameters can now be used to train a calibrated version 
    # of the LDA algorithm. The function below runs the model with 50 passes  
    # over the training corpus to yield likeliest n-grams associated with each  
    # topic, along with a coherence score.
    def calibratedLDAAlgorithm(numberOfTopics,seedSize,alphaValue,etaValue):
        
        # Train the LDA algorithm with the calibrated parameters
        ldaModel = models.LdaModel(corpus=corpus,id2word=ngramIDs,
                                    num_topics=numberOfTopics,
                                    random_state=seedSize,
                                    alpha=alphaValue,
                                    eta=etaValue,
                                    passes=50,
                                    eval_every=None)
                
        # Make coherence global for final text output
        global coherence
        # Calculate the coherence score of the fitted model
        coherence = models.CoherenceModel(model=ldaModel, texts=trainingCorpus,
                                          dictionary=ngramIDs, coherence='u_mass')
        
        # Make the trained LDA algorithm available outside the function
        return ldaModel
    
    trainedModel = calibratedLDAAlgorithm(numberOfTopics,seedSize,alphaValue,etaValue)
    return trainedModel

#################### EVALUATE THE TRAINED LDA ALGORITHM #######################

# The trained model can now be evaluated by constructing word clouds for
# each topic and computing document-topics densities that summarize the most
# appropriate document assignment
def evaluateTrainedModel(trainedModel,filepath):
    
    print("\nPlease wait while the trained model is evaluated...")
    
    # Firstly, construct the word clouds; the largest words are those that occur
    # in each topic the most frequently
    def wordCloudPerTopic(filepath):
        
        # The while loop runs until numberOfTopics is reached
        topic = 0
        while topic < numberOfTopics:
            # Create a dictionary of the paired n-grams (top 100) and 
            # frequencies for the topic
            topicWordFreq = dict(trainedModel.show_topic(topic, topn=100)) 
            topic += 1

            # Generate the word cloud
            wordCloud = WordCloud(background_color="white",colormap="plasma",collocations=False, contour_width=2.5,
                                  mask=np.array(Image.open(filepath + "Word Cloud Templates/" + textsOfInterest.texts + ".jpg")),
                                  max_words=100).generate_from_frequencies(topicWordFreq)
            fig = plt.figure()
            fig.text(x=0.5,y=0.08,s=textsOfInterest.texts + " Topic " + str(topic),
                      fontsize=16,fontstyle="italic", horizontalalignment='center')
            plt.imshow(wordCloud, interpolation='bilinear')
            plt.axis("off")
            plt.savefig(filepath + "/Model Training Results/" + textsOfInterest.texts+ "/Corpus of Topic " + str(topic) + ".png",dpi=300)

    wordCloudPerTopic(filepath)
    
    # Secondly, compute document-topic densities for each document in the
    # training corpus, to determine the likelihood of the topics' association
    # with each one
    def documentTopicDensity(filepath):
            
        # Empty list to hold the computed densities
        densityList = []
        assignedTopics = []
        
        # Calculate document-topic densities for each text and add to the
        # empty list
        docNumber = 0
        for doc in corpus:
            docTopics = trainedModel.get_document_topics(corpus[docNumber],minimum_probability=0)
            # Add the document-topic densities to one list
            densityList.append(docTopics)
            # Add the likeliest topic for each document to the other
            assignedTopics.append(max(docTopics,key=itemgetter(1))[0]+1)
            docNumber += 1

        # The nested for loop regroups the calculated document-topic densities
        # by number of topics, and produces corresponding column names
        probsPerDoc = []
        colNames = []
        stem = "Topic"
        for i in range(numberOfTopics):
            # Create the column name
            colName = stem + " " + str(i+1)
            colNames.append(colName)
            probs = []
            for j in range(len(densityList)):
                # Extract the document-topic density
                probs.append(densityList[j][i][1])
            probsPerDoc.append(probs)

        # Create a Pandas dataframe that will hold document details and the
        # document-topic density results
        df = pd.DataFrame()
        df["Document Title"] = titles
        df["Citation"] = citations

        # Create dictionary from the two lists and iteratively add to the 
        # pandas dataframe
        dictionary = dict(zip(colNames,probsPerDoc))
        for colName, colData in dictionary.items():
            df[colName] = colData   
            # Reduce probabilities to 3 decimal places
            df[colName] = df[colName].astype(str).str[:5]
        
        # Add the likeliest topic and document URLs as final dataframe columns
        df["Likeliest Topic"] = assignedTopics
        df["URL"] = urls
        
        # Sort by the likeliest topics and reset the index
        df.sort_values("Likeliest Topic",inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Save as a csv file
        df.to_csv(filepath + "/Model Training Results/" + textsOfInterest.texts + "/Document-Topic Densities.csv")

    documentTopicDensity(filepath)
    
    # Finally, word webs are created that illustrate the pairwise occurrence of 
    # the commonest n-grams together in the same documents. A spreadsheet that
    # quantifies the contents of each word web is also created.
    def wordWebs(filepath):
        
        # Run while loop until numberOfTopics is reached
        topic = 0
        while topic < numberOfTopics:
            # Dictionary is created only for the 25 most common n-grams
            topicWordFreq = dict(trainedModel.show_topic(topic, topn=25)) 
            topic += 1
 
            # All keys and values in the dictionary are added to separate lists 
            ngramKeys = list(topicWordFreq.keys())
            ngramValues = list(topicWordFreq.values())
            
            # Values are rescaled from 15 to 50 for fontsize control in the word web
            oldRange = (max(ngramValues)-min(ngramValues))
            newMax = 50
            newMin = 15
            # Add the rescaled values to a new list
            valuesRescale = []
            for i in range(len(ngramValues)):
                value = ((((ngramValues[i]-min(ngramValues))*(newMax-newMin))/oldRange) + newMin)
                valuesRescale.append(value)
           
            # This list contains all possible pairs of n-grams
            pairedNgrams = list(combinations(ngramKeys,2))
            # This list will hold the frequency of each n-gram pair
            pairFrequencies = []
            
            # Run each pair of n-grams back through the training corpus. The
            # frequency of each pair within the same documents is recorded
            for i in range(len(pairedNgrams)):
                pairCount = 0 
                for j in range(len(trainingCorpus)):
                    if pairedNgrams[i][0] in trainingCorpus[j] and pairedNgrams[i][1] in trainingCorpus[j]:
                        pairCount = pairCount + 1
                pairFrequencies.append(pairCount)
            
            # Pairwise frequencies are also rescaled for color control, this
            # time from 0.1 to 1
            oldRange = (max(pairFrequencies)-min(pairFrequencies))
            newMax = 1
            newMin = 0.1
            # Add the rescaled pairwise frequencies to a new list
            pairRescale = []
            for i in range(len(pairFrequencies)):
                pair = ((((pairFrequencies[i]-min(pairFrequencies))*(newMax-newMin))/oldRange) + newMin)
                pairRescale.append(pair)
            # Frequencies of 0.1 are recast back to zero to reflect lack of pairwise occurrence
            pairRescale = [0.0 if i == 0.1 else i for i in pairRescale]  
            # Any frequencies that ended up larger than 1 (though this should
            # not happen) are recast to 1
            pairRescale = [1.0 if i > 1 else i for i in pairRescale] 
            
            # A dictionary is created using the pairs of n-grams and the 
            # rescaled pairwise frequencies
            dictNgramPairs = {}
            for key in pairedNgrams:
                for pair in pairRescale:
                    dictNgramPairs[key] = pair
                    pairRescale.remove(pair)
                    break
            
            # Build a dataframe that allows n-grams to be indexed by their values
            df = pd.DataFrame()
            df["N-Grams"] = ngramKeys
            df["Values"] = ngramValues
            df["Values_Rescaled"] = valuesRescale
            
            # These two lists will contain the number of times each of the 25
            # most common n-grams appeared paired with another in the training
            # corpus, and the number of documents in which each n-gram
            # appeared individually
            pairOccurrences = []
            soloOccurrences = []
            
            for i in range(len(ngramKeys)):
                pairFrequency = 0
                soloFrequency = 0
                
                # Fills the paired occurrences list
                for j in range(len(pairedNgrams)):
                    if ngramKeys[i] in pairedNgrams[j]:
                        pairFrequency = pairFrequency + pairFrequencies[j]
                pairOccurrences.append(pairFrequency)
                
                # Fills the solo occurrences list
                for j in range(len(trainingCorpus)):
                    if ngramKeys[i] in trainingCorpus[j]:
                        soloFrequency = soloFrequency + 1
                soloOccurrences.append(soloFrequency)
           
            # Add these two lists to the dataframe defined above, expressed as 
            # raw values and as percentages (to 1 decimal place)
            df["Pair_Freq"] = pairOccurrences
            pairPercent = df["Pair_Freq"]/(len(trainingCorpus)*25)*100
            df["Pair_%"] = pairPercent.round(1)
            df["Solo_Freq"] = soloOccurrences
            soloPercent = df["Solo_Freq"]/len(trainingCorpus)*100
            df["Solo_%"] = soloPercent.round(1)
            # Save the dataframe
            df.to_csv(filepath + "/Model Training Results/" + textsOfInterest.texts + "/Spreadsheet for Word Web for Topic " + str(topic) + ".csv")
            
            # Add this dictionary as the attribute of an empty graph, where
            # rescaled pairwise frequencies represent the color, width, and
            # opacity of each line
            G = nx.Graph()
            G.add_edges_from(pairedNgrams)
            nx.set_edge_attributes(G,dictNgramPairs,"attributes")

            # Apply the attributes to each edge of the graph
            edges = G.edges()
            attributes = [G[u][v]['attributes'] for u,v in edges]
            
            # The dictionaries below will hold the widths, colors, and opacities
            dictEdgeColor = {}
            dictEdgeAlpha = {}
            dictEdgeWidth = {}
            for key in pairedNgrams:
                for attribute in attributes:
                    dictEdgeColor[key] = (0,attribute,0)
                    dictEdgeAlpha[key] = attribute
                    dictEdgeWidth[key] = attribute
                    attributes.remove(attribute)
                    break
            
            # Sort the original dictionary so that the order of plotted edges
            # is set, such that those with the greatest number of pairs are
            # plotted on the top
            dictEdgeOrder = dict(Counter(dictNgramPairs).most_common())
                        
            # The nodes of the graph are each n-gram, held about in a circular
            # layout that minimizes edge crossover. Sort the dataframe by the
            # Pair_Freq column first
            # Sort the dataframe for use when setting the nodes of the graph
            df = df.sort_values("Pair_Freq")
            nodes = get_circular_layout(pairedNgrams, node_order = df["N-Grams"].tolist(),reduce_edge_crossings=False)
            
            # Set the edge layout, which controls how long/curved the edges
            # are between n-gram pairs
            layout = get_bundled_edge_paths(pairedNgrams, nodes, compatibility_threshold = 0.5, straighten_by=0.5)
            
            # # Set the plot size
            plt.figure(figsize=(15,15))

            # Create the word web, using the spring layout and edge widths, and also
            # specifying that edges should be bundled for simpler interpretation
            Graph(pairedNgrams,
                  node_color="black",
                  node_layout=nodes,
                  node_size=1,
                  edge_width=dictEdgeWidth,   
                  edge_color=dictEdgeColor,
                  edge_alpha=dictEdgeAlpha,
                  edge_zorder=dictEdgeOrder,
                  edge_layout=layout)
            
            # Add each n-gram as a text label on top of the word web, its size 
            # proportional to its individual frequency of occurrence 
            for node, (x, y) in nodes.items(): 
                plt.text(x-0.028, y-0.028, node, color="black", weight="bold", fontsize=df["Values_Rescaled"][df.index[df["N-Grams"]==str(node)]], ha='center', va='center')
                plt.text(x-0.025, y-0.025, node, color="red", weight="bold", fontsize=df["Values_Rescaled"][df.index[df["N-Grams"]==str(node)]], ha='center', va='center')

            # Add a title and save the word web
            plt.title("Word Web for " + textsOfInterest.texts + " Topic " + str(topic),
                      fontsize=30, fontstyle="italic", weight = "bold", horizontalalignment='center')
            plt.savefig(filepath + "/Model Training Results/" + textsOfInterest.texts+ "/Word Web for Topic " + str(topic) + ".png",dpi=300)

            
    wordWebs(filepath)

############################## TEXT FILE WRITE ################################

# All of the end-user decisions that went into training the model, along 
# with other outputs of interest, are written to a text file for later reference
def writeTextFile(filePath, yesNo):
    
    # Write and open a new text file
    with open(filePath + "/Model Training Results/" + textsOfInterest.texts + "/End-User Decisions and Other Outputs.txt", 'w') as file:
        # Write content to the file
        file.write("Outputs and Assumptions:\n")
        file.write("\nWere the PDFs pre-processed again before training the model?:" + yesNo + "\n")
        file.write("\nN-gram size selected by the user:" + ngramSize.choice + "\n")
        file.write("\nDid the user select to pre-emptively remove the 100 commonest n-grams?:" + removeCommonNgrams.yesNo + "\n")
        file.write("\nThe 100 most common n-grams in the entire corpus:" + str(commonNgrams) + "\n")
        file.write("\nDid the user choose to apply a TF-IDF algorithm to the training corpus?:" + useTFIDF.yesNo + "\n")
        file.write("\nDid the user choose the default alpha (50/numTopics) to train the model?:" + useDefaultAlpha.yesNo + "\n")
        file.write("\nDid the user choose the default eta (0.1) to train the model?:" + useDefaultEta.yesNo + "\n")
        file.write("\nCalibrated Number of Topics:" + str(numberOfTopics) + "\n")
        file.write("\nCalibrated Seed Size:" + str(seedSize) + "(" + str(calibToPlot) + "%)" + "\n")
        if useDefaultAlpha.yesNo == "Y":
            file.write("\nAlpha Value:" + str(50/numberOfTopics) + "\n")
        else:
            file.write("\nCalibrated Alpha:" + str(alphaValue) + "\n")
        if useDefaultEta.yesNo == "Y":
            file.write("\nEta Value:" + "0.1\n")
        else:
            file.write("\nCalibrated Eta:" + str(etaValue) + "\n")
        file.write("\nCoherence Score of the trained model:" + str(coherence.get_coherence()) + "\n")
        
############################ MOVE TO A SUB-FOLDER ############################

# All of the calibration charts, words clouds, the document-topic density
# spreadsheet, and the written text file need to be moved to a sub-folder of
# their own. Doing so allows for comparison between model runs when conducting
# sensitivity analysis
def moveToSubFolder(filePath, yesNo):
    
    # Path to folder containing the model training results
    folderPath = filePath + "/Model Training Results/" + textsOfInterest.texts    

    # Define the sub-folder's name, made up of six end-user choices:
    # Redoing pre-processing, n-gram size, removal of 100 commonest n-grams,
    # use of TFIDF, use of default alpha, and use of default eta
    subFolderName = "Redo" +yesNo+ "_Ngram" +ngramSize.choice+ "_Remove" +removeCommonNgrams.yesNo+ "_TFIDF" +useTFIDF.yesNo+ "_Alpha" +useDefaultAlpha.yesNo+ "_Eta" +useDefaultEta.yesNo
    
    # Complete path to the sub-folder
    subFolderPath = filePath + "/Model Training Results/" + textsOfInterest.texts + "/" + subFolderName
    # Delete the sub-folder first if it already exists
    if os.path.exists(subFolderPath):
        shutil.rmtree(subFolderPath)
    # Copy all training results to the new sub-folder    
    shutil.copytree(folderPath, subFolderPath, dirs_exist_ok=True, ignore=shutil.ignore_patterns('Redo*'))
    
    # Delete all files outside of the sub-folder
    fileList = os.listdir(folderPath)
    fileList = [x for x in fileList if not x.startswith("Redo")]
    [os.remove(folderPath + "/" + x) for x in fileList]        
