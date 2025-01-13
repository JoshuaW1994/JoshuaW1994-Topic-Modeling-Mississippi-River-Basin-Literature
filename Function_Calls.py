# Script Name: Function_Calls
# Author: Joshua (Jay) Wimhurst
# Date Created: 10/17/2023
# Date Last Edited: 1/13/2025

################################ DESCRIPTION ##################################
# Functions in the Preprocessing_and_Topic_Modeling_Functions script are called
# here, first extracting desired main texts iteratively, saving them to the
# database and then performing topic modeling on the desired texts
# NOTE: When adding new texts to Document Details.xlsx, they must be fully
# documented before attempting to pre-process them!
###############################################################################

# Necessary packages
import fitz
from natsort import natsorted

# Update filepath on Line 19 to match where all downloaded files/folders are located
filepath = #insert filepath here
stopwordsFilePath = filepath + "Stopwords.csv"       

# =============================================================================
#                         PRE-PROCESSING FUNCTIONS
# =============================================================================

# Ask user whether to pre-process text first or go straight to topic modeling
from Preprocessing_and_Topic_Modeling_Functions import preprocessText
yesNo = preprocessText.yesNo

# If the user said yes to pre-processing, then run the functions below
if yesNo == "Y":

    # Create the list of PDFs to pre-process, calling on the filepath to the 
    # Document Details database as an argument
    from Preprocessing_and_Topic_Modeling_Functions import pdfFileList
    fileList = pdfFileList(filepath + "Document Details.xlsx")
    
    # List to be filled and added to the "Preprocessed Text" column in the
    # Document Details database
    finalTexts = []
    
    # The following functions from the pre-processing script are called 
    # iteratively to fill the empty list above with the main text from each PDF
    for file in natsorted(fileList):
        
        # Convert the PDF document into text, using the filepath to each PDF
        # document as an argument
        from Preprocessing_and_Topic_Modeling_Functions import pdfToText
        print(file + "\n")
        text = pdfToText(fitz.open(filepath + "PDFs/" + file))

        # Removal of any lines of PDF text that do not contribute to the main text
        from Preprocessing_and_Topic_Modeling_Functions import delUnwantedLines
        text = delUnwantedLines(text)
        
        # Removal of alphanumeric characters within lines that also do not
        # contribute to the main text
        from Preprocessing_and_Topic_Modeling_Functions import delInsideLines
        text = delInsideLines(text)
        
        # Tokenize the text so that any remaining undesired words and characters
        # can be removed more precisely
        from Preprocessing_and_Topic_Modeling_Functions import tokenizeAndRemove
        text = tokenizeAndRemove(text,stopwordsFilePath)
        
        # Add the pre-processed text to the empty list (finalTexts)
        finalTexts.append(text)
    
    # Once the loop ends, pre-processed texts are added and saved to the
    # Document Details database
    from Preprocessing_and_Topic_Modeling_Functions import appendAndSave
    database = appendAndSave(filepath + "Document Details.xlsx",finalTexts)
    
    # The database is finally opened as a pandas dataframe in preparation
    # for the LDA algorithm training
    from Preprocessing_and_Topic_Modeling_Functions import openDocumentDetails
    database = openDocumentDetails(filepath + "Document Details.xlsx")

# =============================================================================
#                   LATENT DIRICHLET ALLOCATION FUNCTIONS
# =============================================================================

# If the user said no to pre-processing, only the openDocumentDetails
# function from above is used to open the database as a pandas dataframe
else:
    from Preprocessing_and_Topic_Modeling_Functions import openDocumentDetails
    database = openDocumentDetails(filepath + "Document Details.xlsx")

# Extract the desired texts from the Document Details database based on
# user input criteria (state/sub-basin/decade)
from Preprocessing_and_Topic_Modeling_Functions import textSelection
textsForTraining = textSelection(database)

# Create the corpus that will be used to train the LDA algorithm, also
# specifying the n-gram size with user input
from Preprocessing_and_Topic_Modeling_Functions import createCorpus
trainingCorpus = createCorpus(textsForTraining,filepath)

# Train the Latent Dirichlet Allocation (LDA) algorithm and provide user 
# inputs for performing later sensitivity analysis on model output
from Preprocessing_and_Topic_Modeling_Functions import trainLDAAlgorithm
trainedModel = trainLDAAlgorithm(trainingCorpus,filepath)

# Use the trained LDA algorithm to create word clouds that show the frequency 
# of n-grams within topics, assess document-topic densities, and create word
# webs showing the pairwise occurrence of the commonest n-grams within documents
from Preprocessing_and_Topic_Modeling_Functions import evaluateTrainedModel
evaluateTrainedModel(trainedModel,filepath)

# Write all end-user decisions and other model outputs not presented in 
# map/chart form to a separate text file
from Preprocessing_and_Topic_Modeling_Functions import writeTextFile
writeTextFile(filepath, yesNo)

# Copy all the model outputs into a sub-folder of their own, with the
# sub-folder's name reflecting the decisions made by the user
from Preprocessing_and_Topic_Modeling_Functions import moveToSubFolder
moveToSubFolder(filepath, yesNo)

