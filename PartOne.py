#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
import spacy
from pathlib import Path

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

import pandas as pd
import os

def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    novels_data = []  # empty list to store novel data

    for file_path in path.glob("*.txt"):  # only care about .txt files
        if file_path.suffix == ".txt":
            filename_parts = file_path.stem.split("_")  # remove the .txt part of the filename

            year = int(filename_parts[-1])  # year is the number at the end of the filename

            author = filename_parts[-2]  # author is the second to last part of the filename

            title = "-".join(filename_parts[1:])  # title is whatever is left in the filename

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
    # end of loop, make a pandas dataframe       
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
    if word in d:
        try:
            return len([p in p in d[word][0] if p[-1].isdigit()])  # count the number of syllables in the word
        except KeyError:
            pass
    vowels = "aeiouy"  # define the vowels
    syllable_count = 0

    #if the words starts with a vowel, add one to the syllable count
    if word[0].lower() in vowels:
        syllable_count += 1

    #this is to doublecheck that the word is not empty and it has more than one letter
    for i in range(1, len(word)):
        #if the current letter is a vowel & the previous letter is not a vowel+ add one to the syllable count
        if word[i].lower() in vowels and word[i - 1].lower() not in vowels:
            syllable_count += 1
        if word.endswith("e"):
            syllable_count -= 1
        
        if syllable_count == 0:
            syllable_count = 1
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
    words = [word for word in nltk.word_tokenize(text) if word.isalpha()]  # only keep words that are letters
    total_words = len(words)  # total number of words in the text

    total_syllables = 0  # total number of syllables in the text
    for word in words:
        total_syllables += count_syl(word, d) #loop each word + use count_syl 

    if total_words == 0 or total_sentences == 0:
        return 0.0
    #this is to check if the text is empty/has no sentences

    #standard Flesch-Kincaid Grade Level formula
    #calcuate word complexity by using the total number of words, total number of sentences, and total number of syllables
    fk_score = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
    return fk_score  # return the Flesch-Kincaid Grade Level of the text


    pass


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""

    #save pickle file or make a new one


    pass


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""

    text = text.lower()  # convert text to lowercase

    tokens = nltk.word_tokenize(text)  # tokenize the text

    if not words:
        return 0.0
    
    total_tokens = len(tokens)  # this is the number of tokens in the text
    
    total_types = len(set(words))  # this is the number of unique words in the text

    return total_types / total_tokens  # return the type-token ratio (TTR) of the text




    pass


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    pass



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

