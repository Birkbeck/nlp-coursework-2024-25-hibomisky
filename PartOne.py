#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
import spacy
from pathlib import Path

import pandas as pd
import os
import pickle
import math

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000 #cos the novels are long



def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    novels_data = []  # empty list to store novel data

    for file_path in path.glob("*.txt"):  # only care about .txt files
        if file_path.suffix == ".txt":
            filename_parts = file_path.stem.split("_")  # remove the .txt part of the filename
            #split filename with - like the way the filenames are

            year = int(filename_parts[-1])  # year is the number at the end of the filename

            author = filename_parts[-2]  # author is the second to last part
            # this is to get the author name, which is the second to last part of the filename
            #title is allowed to have - 
            title = "-".join(filename_parts[:-2])  # title is whatever is left in the filename
            # this is to join the rest of the filename parts with - to get the title
            
            # open and read the text file in the novel folder
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            # make a dictionary of the novel data to add to the empty list
            novels_data.append({
                "title": title,
                "author": author,
                "year": year,
                "text": text
            })
    # this is the end of the loop + make a pandas dataframe       
    df = pd.DataFrame(novels_data)
    df.sort_values(by="year", inplace=True)  # sort the dataframe by year
    df.reset_index(drop=True, inplace=True)  # reset the index of the dataframe/delete old index

    return df

def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    # check if the word is in the dictionary
    word = word.lower() #make it lowercase
    
    if word in d:
        try: #this is for error handling for words not in the dictionary
            return len([p for p in d[word][0] if p[-1].isdigit()])  # count the number of syllables in the word
        except (KeyError, IndexError): #copilot suggested to add IndexError to handle cases where the word is not in the dictionary
            pass
    vowels = "aeiouy"  # define the vowels
    syllable_count = 0

    #if the words starts with a vowel, add one to the syllable count
    if word and word[0] in vowels:
        syllable_count += 1

    #this is to doublecheck that the word is not empty and it has more than one letter
    for i in range(1, len(word)):
        #if the current letter is a vowel & the previous letter is not a vowel+ add one to the syllable count
        if word[i].lower() in vowels and word[i -1] not in vowels:
            syllable_count += 1 #this is to count the number of syllables in the word
        
        if syllable_count == 0:
            syllable_count = 1 #this is for words with no vowels (like mythS)
   
    return syllable_count  # return the syllable count for the word
    
def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    #use NLTK word tokenizer to tokenize the text
    
    sentences = nltk.sent_tokenize(text)  # tokenize the text into sentences
    total_sentences = len(sentences)  # total number of sentences in the text
    
    #now use the NLTK tokeniser to tokenize the text into words
    words = [word for word in nltk.word_tokenize(text) if word.isalpha()]  # only keep words that are letters
    total_words = len(words)  # total number of words in the text

    #count the number of syllables in each word using the count_syl function
    total_syllables = 0  # total number of syllables in the text
    for word in words:
        total_syllables += count_syl(word, d) #loop each word + use count_syl 

    if total_words == 0 or total_sentences == 0:
        return 0
    #this is to check if the text is empty/has no sentences

    #standard Flesch-Kincaid Grade Level formula
    #calcuate word complexity by using the total number of words, total number of sentences, and total number of syllables
    fk_score = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
    return fk_score  # return the Flesch-Kincaid Grade Level of the text


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""

    #save pickle file or make a new one
    if not store_path.exists():
        store_path.mkdir(parents=True) # create the directory if it doesn't exist (copilot)

    parsed_docs = []  # empty list to store all the parsed docs

    for text in nlp.pipe(df['text'].tolist(), disable = ["ner"]):
    # end of loop, add the parsed docs to the dataframe
        parsed_docs.append(doc) #doc now has the parsed text

    df['parsed'] = parsed_docs  # add the parsed docs to the dataframe
    #say what the output file path is
    output_file_path = store_path / out_name  # this is where to find the output_file
    #now savethe dataframe to a pickle file
    with open(output_file_path, "wb") as f: #this opens file 
        pickle.dump(df, f) #should see a df in the pickle file

        return df  # return the dataframe with the parsed + saved docs



def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""

    text = text.lower()  # convert text to lowercase letters cos coursework says ignore the case

    tokens = nltk.word_tokenize(text)  # tokenize the text

    #no words = TTR = 0
    words = [token for token in tokens if token.isalpha()]  # this is to keep only words (ignore punctuation and numbers)
    if not words:
        return 0
    
    total_tokens = len(tokens)  # this is the number of tokens in the text
    
    total_types = len(set(words))  # this is the number of unique words in the text

    return total_types / total_tokens  # return the type-token ratio (TTR) of the text


        #need to define words and total_sentences



def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"]) #this is for token type ratio for the rows
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()  # load the cmudict dictionary for syllable counting

    for _, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 3) 
        #this is to round off fk score to 3.d.p
    return results


def subjects_by_verb_pmi(doc, target_verb): #wrong function change!!!
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    
    subject_verb_cooccurrences = Counter()
    for token in doc:
        if token.lemma_ == verb_lemma:
            #for every token in the doc, check if token = verb
            for child in token.children:
                if child.dep_ == "nsubj":
                    #if the child is a subject of the verb, you got to add it to the list of subjects
                    subject_verb_cooccurrences[child.lemma_.lower()] += 1
                    #this is to count the number of times the subject is with the verb
    
    all_subjects_counts = Counter(token.lemma_.lower() for token in doc if token.dep_ == "nsubj")
    #so you can ger total number of subjects in the doc
    total_subjects = sum(all_subjects_counts.values())  
    # this is the total number of subjects in the doc

    X = sum(subject_verb_cooccurrences.values())  # total co-occurrences of the verb with subjects
    if X == 0 or if total_subjects == 0:
        return []


def object_count(doc): #for question 1f
    #to find the most common objecys in parsed spacy doc
    objects = [token.lemma_.lower() for token in doc if token.dep_ == "doc_obj"]

    return Counter(objects).most_common(10)  # this is top 10 common objects in the doc


def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    for token in doc:
        if token.lemma_ == verb_lemma:
        #for every token in the doc, check if token = verb
            for child in token.children:
                if child.dep_ == "nsubj": #then its a subject of verb
                #if the child is a subject of the verb, add it to the list of subjects
                    subjects.append(child.lemma_.lower())
    
    return Counter(subjects).most_common(10)  # this will return the top 10 common subjects for the verb
    
    
    pass



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    adjectives = [token.lemma.lower() for token in doc if token.pos_ == "ADJ"]

    return Counter(adjectives).most_common(10)  # return the top 10 common adjectives





if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    #path = Path.cwd() / "p1-texts" / "novels"
    #print(path)
    #df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    #print(df.head())
    #nltk.download("cmudict")
    #parse(df)
    #print(df.head())
    #print(get_ttrs(df))
    #print(get_fks(df))
    #df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # print(adjective_counts(df))
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """

