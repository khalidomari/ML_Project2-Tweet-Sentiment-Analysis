import re
from collections import Counter
from itertools import groupby
from wordsegment import load, segment

def check_dict(word, dictionary):
    if word in dictionary:
        return True
    return False

def correct_char_repetition(word):
    word = word.lower()
    occurance = [(k, sum(1 for i in g)) for k,g in groupby(word)]
    if len(occurance)==1: 
        return word
    if max([j for (_,j) in occurance]) > 2:
        corrected_word = ''
        for (i,j) in occurance:
            if j>2:
                corrected_word += 2*i
            else:
                corrected_word += i*j
        return corrected_word
    else:
        return word

def words(text): return re.findall(r'\w+', text.lower())

dictionary = Counter(words(open('../data/english_words.txt').read()))

def P(word, N=sum(dictionary.values())): 
    "Probability of `word`."
    return dictionary[word] / N

def separate(words):
    return segment(words)

def correction(word): 
    #load()
    "Most probable spelling correction for word."
    
    #delete repeated letters
    word = correct_char_repetition(word)

    #if degits return it
    if word.isdigit():
        return word

    if word in dictionary:
        return word
    else:
        segments = separate(word)
        if len(segments) > 1:
            return ' '.join(word for word in segments)
        else:
            return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in dictionary)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))