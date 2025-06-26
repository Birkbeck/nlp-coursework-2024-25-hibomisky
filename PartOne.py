#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
import spacy
from pathlib import Path

import pandas as pd
import os
import pickle
import math

from collections import Counter #added to count objects, subjects, and adjectives

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000 #cos the novels are long


def read_novels(path=Path.cwd() / "texts" / "novels"):

    novels_data = []  # empty list to store novel data

    for file_path in path.glob("*.txt"):  # only care about .txt files
        if file_path.suffix == ".txt":
            try:
                filename_parts = file_path.stem.split('-')  # remove the .txt part of the filename
                #split filename with - like the way the filenames are
                    #corrected "_" to '-' syntax issues

                if len(filename_parts) == 3:
                    title, author, year = filename_parts  # if there are 3 parts, then title, author, year
                    #added this to make it clearer that the filename is split into 3 parts
                    year = int(year.strip())  # convert year to an integer and remove any whitespace
                else:
                    print("This file does not have the correct format:", file_path)

                # open and read the text file in the novel folder
                with open(file_path, "r", encoding="utf-8") as f: #copilot suggested to use utf-8 encoding
                    # this is to read the file in utf-8 encoding which is standard for text files apaprently
                    text = f.read()
                    # make a dictionary of the novel data to add to the empty list
                novels_data.append({
                    "title": title,
                    "author": author,
                    "year": year,
                    "text": text
                })
            # this is the end of the loop + make a pandas dataframe  
            except Exception as e:
                print(f"Error reading file {file_path}: {e}") #suggested by copilot
                # this is to catch any errors that occur while reading the file and print the error message     
    df = pd.DataFrame(novels_data)

    if df.empty:
        print("No novels found in the specified directory.")
        return df


    df["year"] = pd.to_numeric(df["year"], errors="coerce")  # convert year to numeric (copilot suggested this)
    df.sort_values(by="year", inplace=True)  # sort the dataframe by year
    df.reset_index(drop=True, inplace=True)  # reset the index of the dataframe/delete old index

    return df

def count_syl(word, d):

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
        
    if len(word) > 1 and word.endswith("e"):
        #for words that ends in e, +  word longer than 1 letter (like cake for example)
        syllable_count -= 1  #remove silent e from syllable count
        
    if syllable_count == 0:
        syllable_count = 1 #this is for words with no vowels (like mythS)
        #word must have at least 1 syllable
   
    return syllable_count  # return the syllable count for the word
    
def fk_level(text, d):

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
        return 0.0 #corrected to be like float return type that is stanfard for Flesch-Kincaid Grade Level
    #this is to check if the text is empty/has no sentences

    #standard Flesch-Kincaid Grade Level formula
    #calcuate word complexity by using the total number of words, total number of sentences, and total number of syllables
    fk_score = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
    return fk_score  # return the Flesch-Kincaid Grade Level of the text

def flesch_kincaid(df): #similar to nltk_ttr pattern
    results = {}
    cmudict = nltk.corpus.cmudict.dict() 
    for _, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 3)
    return results #added extra to return dictionary as per questions


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):

    #save pickle file or make a new one
    if not store_path.exists():
        store_path.mkdir(parents=True) # create the directory if it doesn't exist (copilot)

    parsed_docs = []  # empty list to store all the parsed docs

    for doc in nlp.pipe(df['text'].tolist(), batch_size=50, disable = ["ner"]): #error not text but doc here
    # end of loop, add the parsed docs to the dataframe
        parsed_docs.append(doc) #doc now has the parsed text

    df['parsed'] = parsed_docs  # add the parsed docs to the dataframe
    
    #say what the output file path is
    output_file_path = store_path / out_name  # this is where to find the output_file
    
    #now savethe dataframe to a pickle file
    with open(output_file_path, "wb") as f: #this opens file 
        pickle.dump(df, f) #should see a df in the pickle file

    return df  # return the dataframe with the parsed + saved docs
    #corrected indentation 


def nltk_ttr(text):
    
    text = text.lower()
    tokens = nltk.word_tokenize(text)  # tokenize the text as per hint
    words = [token for token in tokens if token.isalpha()]  # this is to keep only words (ignore punctuation and numbers)
    return len(set(words))/len(words) if words else 0.0


def get_ttrs(df): #helper function

    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = round(nltk_ttr(row["text"]),4) #this is for token type ratio for the rows
    return results


def get_fks(df):

    results = {}
    cmudict = nltk.corpus.cmudict.dict()  # load the cmudict dictionary for syllable counting

    for _, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 3) 
        #this is to round off fk score to 3.d.p
    return results

def object_count(doc):

    objects = []
    for token in doc:
        if token.dep in ["dobj", "pobj", "iobj", "obj"]:
            #this is to check for all the object tags

            objects.append(token.lemma_.lower())
    #to find the most common objecys in parsed spacy doc
    return Counter(objects).most_common(10)  # this is top 10 common objects in the doc

def common_objects(doc):
    return Counter(
        token.lemma_.lower() for token in doc #lemma better
        if token.dep_ in {"dobj", "pobj", "iobj"} #all object types
    ).most_common(10) #trying this logic to see if it works

def subjects_by_verb_count(doc, verb): #moved fuction up for my own thinking cos it makes more sense

    subjects = []  # this is to store the subjects of the verb
    for token in doc:
        if token.lemma_ == verb:
            if "hear" in token.lemma_.lower() and token.pos_ == "VERB":
                #to fix the hear issue in main
        #for every token in the doc, check if token = verb
                for child in token.children:
                    if child.dep_ in ["nsubj","nsubjpass"]: #then its a subject of verb
                    #if the child is a subject of the verb, add it to the list of subjects
                        subjects.append(child.lemma_.lower())
        
    return Counter(subjects).most_common(10)  # this will return the top 10 common subjects for the verb

def subjects_by_verb_pmi(doc, target_verb): 

    #first count total times subjects appears with target verb in doc
    cooccrances_count = Counter()
    all_subject_counts = Counter()
    total_verb_subject_count = 0 #fixing counters


    for token in doc:
        if token.dep_ in ["nsubj", "nsubjpass"]: #active+passive
        #if token.lemma_ == target_verb:
            #for every token in the doc, check if token = verb
            #for child in token.children:
            all_subject_counts[token.lemma_.lower()] +=1

        if token.lemma_.lower() == target_verb and token.pos_ == "VERB":
            total_verb_subject_count += 1
            
            
    #second: get counts for all subject for any verb in the doc
    for token in doc:
        if token.lemma_.lower() == target_verb and token.pos_ == "VERB":
            for child in token.children: #this is to find subjects in direct children - copilot suggestion
                if child.dep_ in ["nsubj", "nsubjpass"]: #similar to above
                    subject_lemma = child.lemma_.lower()
                    cooccrances_count[subject_lemma] +=1
            

    #now get total counts (changed variable name to stop confusion)
    total_subjects_in_the_doc = sum(all_subject_counts.values())

    if total_verb_subject_count == 0 or total_subjects_in_the_doc == 0:
        return [] #total counts
    
    #now calcuate pmi scores for each subject
    pmi_scores = {}

    for subject, count_cooccurrence in cooccrances_count.items(): #getting confused ahhh/double check this

        count_subject_total = all_subject_counts.get(subject, 0)
        #added this to get the count of the subject in the doc correctly as previously not defined
        #this is to calculate the pmi score for each subject

        if count_subject_total == 0:
            continue

        #pmi formular
        pmi = math.log2((count_cooccurrence * total_subjects_in_the_doc) /
                        (count_subject_total * total_verb_subject_count))
        pmi_scores[subject] = pmi  

    
    return sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    #this is to sort the pmi scores in descending order and return the top 10
 

def adjective_counts(doc): 
    
    df = doc

    all_adjectives = [] #different logic to try to stop issues in main function
    #this is to store all the adjectives in the doc

    for single_doc in doc['parsed']: #loop for each parsed doc in the DataFrame (df changed to doc)
            for token in single_doc:
                if token.pos_ == "ADJ":  # check if the token is an adjective
                    all_adjectives.append(token.lemma_.lower())
    
    return Counter(all_adjectives).most_common(10)  # return the top 10 common adjectives


if __name__ == "__main__":


    path = Path.cwd() / "novels"
    #this is to show the path to the novels folder
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above (done)
    print(df.head())
    nltk.download("cmudict")
    parse(df)
    print(df.head())
    print(get_ttrs(df))
    print(get_fks(df))
    df = pd.read_pickle(Path.cwd() / "pickles" /"parsed.pickle")
    print(adjective_counts(df))
    print(flesch_kincaid(df)) #added after extra function as per questions
    
     
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(f"{row['title']} - common objects:" )
        print(common_objects(row["parsed"]))
       #print(df["parsed"].iloc[0]) #checking to see if its a spacy doc and not empty in terminal because i keep having bugs