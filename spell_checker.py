import nltk
import random
import math
import re
from collections import Counter
from nltk.metrics.distance import edit_distance
import spelling_confusion_matrices
from nltk.corpus import gutenberg, names


class Spell_Checker:
    """The class implements a context sensitive spell checker. The corrections
        are done in the Noisy Channel framework, based on a language model and
        an error distribution model.
    """

    def __init__(self, lm=None):
        """Initializing a spell checker object with a language model as an
        instance  variable.

        Args:
            lm: a language model object. Defaults to None
        """
        self.lm = lm
        self.error_tables = {}

    def add_language_model(self, lm):
        """Adds the specified language model as an instance variable.
            (Replaces an older LM dictionary if set)

            Args:
                lm: a Spell_Checker.Language_Model object
        """
        self.lm = lm

    def add_error_tables(self, error_tables):
        """ Adds the specified dictionary of error tables as an instance variable.
            (Replaces an older value dictionary if set)

            Args:
            error_tables (dict): a dictionary of error tables in the format
            of the provided confusion matrices:
            https://www.dropbox.com/s/ic40soda29emt4a/spelling_confusion_matrices.py?dl=0
        """

        prob = {'insertion':{}, 'deletion':{}, 'substitution':{}, 'transposition':{}}

        for k in error_tables:
            # pass on the char in each section and make new matrix
            for key in error_tables[k]:
                if key[0] not in prob[k]:
                    prob[k][key[0]] = {}
                prob[k][key[0]][key[1]] = error_tables[k][key]
            # matrix count to probability
            for key in error_tables[k]:
                total = sum(Counter(prob[k][key[0]]).values())
                prob[k][key[0]][key[1]] /= (total if total > 0 else 1)

        self.error_tables = prob


    def evaluate(self, text):
        """Returns the log-likelihood of the specified text given the language
            model in use. Smoothing should be applied on texts containing OOV words

           Args:
               text (str): Text to evaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """
        return self.lm.evaluate(text)

    def spell_check(self, text, alpha):
        """ Returns the most probable fix for the specified text. Use a simple
        noisy channel model if the number of tokens in the specified text is
        smaller than the length (n) of the language model.

        Args:
            text (str): the text to spell check.
            alpha (float): the probability of keeping a lexical word as is.

        Return:
            A modified string (or a copy of the original if no corrections are made.)
        """
        alpha = 1-alpha # the log is lower is better so time alpha should be lower is better
        txt = text.split()
        n = self.lm.get_model_window_size()
        suggest = self.lm.get_suggested_dictionary()
        dic = self.lm.get_model_dictionary()
        error_tables = self.error_tables
        correct_words = self.lm.get_words()
        # nltk.download('names')  # todo uncomment if not have this on your computer
        names_lst = names.words()

        # check if have word with error
        grams = "# " * (n - 1) + text
        i = -1
        sug = []
        err = False  # if found error in word
        for gram in nltk.ngrams(grams.split(), n):
            i += 1
            ngram = " ".join(gram)

            if txt[i] in names_lst:  # if found a name
                continue

            if ngram not in dic and txt[i] not in correct_words:  # found an oov word
                if n > len(txt) or not suggest.get(" ".join(gram[:n - 1]), False):  # if cant fix by context
                    word = txt[i]
                    temp = [w for w in correct_words if
                            word[0] == w[0] and edit_distance(word, w) == 1]

                    for w in temp:

                        w1 = "#" + word
                        w2 = "#" + w
                        lst1 = list(w1)
                        len_lst1 = len(lst1)
                        lst2 = list(w2)

                        prob = 1
                        j = 0

                        # pass over all letter and check the probability of the change
                        while j < min(len(w1), len(w2)):
                            if lst1[j] == lst2[j]:  # char not changed
                                j += 1
                                continue

                            if j + 1 < len(w1):
                                if lst1[j + 1] == lst2[j]:
                                    if j + 1 < len(w2) and lst1[j] == lst2[j + 1]:  # [trans]
                                        prob *= error_tables["transposition"][w1[j]][w1[j + 1]]
                                        j += 1
                                    else:  # [ins]
                                        prob *= error_tables["insertion"][w1[j - 1]][w1[j]]
                                        len_lst1 -= 1
                                elif j + 1 < len(w2) and lst1[j] == lst2[j + 1]:  # [del]
                                    prob *= error_tables["deletion"][w1[j - 1]][w2[j]]
                                    len_lst1 += 1
                                else:  # [sub]
                                    prob *= error_tables["substitution"][w1[j]][w2[j]]
                            elif j + 1 < len(w2):  # [del]
                                prob *= error_tables["deletion"][w1[j - 1]][w2[j]]
                                len_lst1 += 1
                            else:  # [sub]
                                prob *= error_tables["substitution"][w1[j]][w2[j]]
                            j += 1

                        if len_lst1 < len(w2):  # [del]
                                if j != len(w2):
                                    prob *= error_tables["deletion"][w1[j - 1]][w2[j]]
                        elif len_lst1 > len(w2):  # [ins]
                            if j == len(w1):
                                prob *= error_tables["insertion"][w1[j - 2]][w1[j-1]]
                            else:
                                prob *= error_tables["insertion"][w1[j - 1]][w1[j]]

                        bug = txt.copy()
                        bug[i] = w
                        t = " ".join(bug)

                        sug.append((t, self.evaluate(t)*prob)) if prob != 0 else 0
                else:

                    gram = " ".join(gram[:n - 1])
                    # check by context
                    after = txt[i+1] if  len(txt) > i+1 else "#"
                    # check the before + word
                    for w in suggest[gram]:
                        bug = txt.copy()
                        bug[i] = w
                        t = " ".join(bug)
                        sug.append((t, self.evaluate(t)))
                    # check the word + after
                    for key, val in suggest.items():
                        for v in val:
                            if v == after:
                                bug = txt.copy()
                                bug[i] = key
                                t = " ".join(bug)
                                sug.append((t, self.evaluate(t)))
                                break

                err = True
                break

        if not err: # not found word with error

            # add original with alpha
            bug = txt.copy()
            t = " ".join(bug)
            sug.append((t, self.evaluate(t) * alpha))
            i = -1

            # add other fixes with 1-alpha
            for gram in nltk.ngrams(grams.split(), n):
                i += 1

                if txt[i] in names_lst:  # if found a name
                    continue

                if n > len(txt) or not suggest.get(" ".join(gram[:n - 1]), False):  # if cant fix by context
                    word = txt[i]
                    temp = [w for w in correct_words if
                            word[0] == w[0] and edit_distance(word, w) == 1]

                    for w in temp:

                        w1 = "#" + word
                        w2 = "#" + w
                        lst1 = list(w1)
                        len_lst1 = len(lst1)
                        lst2 = list(w2)

                        prob = 1
                        j = 0

                        # pass over all letter and check the probability of the change
                        while j < min(len(w1), len(w2)):
                            if lst1[j] == lst2[j]:  # char not changed
                                j += 1
                                continue

                            if j + 1 < len(w1):
                                if lst1[j + 1] == lst2[j]:
                                    if j + 1 < len(w2) and lst1[j] == lst2[j + 1]:  # [trans]
                                        prob *= error_tables["transposition"][w1[j]][w1[j + 1]]
                                        j += 1
                                    else:  # [ins]
                                        prob *= error_tables["insertion"][w1[j - 1]][w1[j]]
                                        len_lst1 -= 1
                                elif j + 1 < len(w2) and lst1[j] == lst2[j + 1]:  # [del]
                                    prob *= error_tables["deletion"][w1[j - 1]][w2[j]]
                                    len_lst1 += 1
                                else:  # [sub]
                                    prob *= error_tables["substitution"][w1[j]][w2[j]]
                            elif j + 1 < len(w2):  # [del]
                                prob *= error_tables["deletion"][w1[j - 1]][w2[j]]
                                len_lst1 += 1
                            else:  # [sub]
                                prob *= error_tables["substitution"][w1[j]][w2[j]]
                            j += 1

                        if len_lst1 < len(w2):  # [del]
                            if j != len(w2):
                                prob *= error_tables["deletion"][w1[j - 1]][w2[j]]
                        elif len_lst1 > len(w2):  # [ins]
                            if j == len(w1):
                                prob *= error_tables["insertion"][w1[j - 2]][w1[j - 1]]
                            else:
                                prob *= error_tables["insertion"][w1[j - 1]][w1[j]]

                        bug = txt.copy()
                        bug[i] = w
                        t = " ".join(bug)

                        sug.append((t, self.evaluate(t) * prob * (1 - alpha))) if prob != 0 else 0
                else:

                    gram = " ".join(gram[:n - 1])
                    # check by context
                    after = txt[i + 1] if len(txt) > i + 1 else "#"
                    # check the before + word
                    for w in suggest[gram]:
                        if edit_distance(txt[i], w) != 1:
                            continue
                        bug = txt.copy()
                        bug[i] = w
                        t = " ".join(bug)
                        sug.append((t, self.evaluate(t) * (1 - alpha)))
                    # check the word + after
                    for key, val in suggest.items():
                        for v in val:
                            if v == after:
                                if edit_distance(txt[i], key) != 1:
                                    break
                                bug = txt.copy()
                                bug[i] = key
                                t = " ".join(bug)
                                sug.append((t, self.evaluate(t) * (1 - alpha)))
                                break

        if not sug:  # not found any fix
            return text
        return max(sug,key=lambda item:item[1])[0]

    #####################################################################
    #                   Inner class                                     #
    #####################################################################

    class Language_Model:
        """The class implements a Markov Language Model that learns a model from a given text.
            It supports language generation and the evaluation of a given string.
            The class can be applied on both word level and character level.
        """

        def __init__(self, n=3, chars=False):
            """Initializing a language model object.
            Args:
                n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
                chars (bool): True iff the model consists of ngrams of characters rather then word tokens.
                              Defaults to False
            """
            self.n = n
            self.chars = chars
            # "#" in dictionary represent empty word
            self.model_dict = None  # a dictionary of the form {ngram:count}, holding counts of all ngrams in the specified text.
            self.suggested = None  # a dictionary of the form {ngram-1:{suggested word:count}}
            self.total_size = 0
            self.unique_size = 0
            self.words = []

        def build_model(self, text):  # should be called build_model
            """populates the instance variable model_dict.

                Args:
                    text (str): the text to construct the model from.
            """
            chars = self.chars
            n = self.n

            # create word list
            # clean the text for error fix
            temp = str(text)
            temp = re.sub(r'[\[|\]|.|!|?|#|,]+', '', temp)

            for word in temp.split():
                if word not in self.words:
                    self.words.append(word)

            dic = {}
            suggested = {}

            if not chars:
                self.total_size = len(text.split())
                self.unique_size = len(set(text.split()))
                # split text to n-gram and add counter in dict
                text = "# " * (n - 1) + text + " #" * (n - 1)
                for gram in nltk.ngrams(text.split(), n):
                    grm = " ".join(gram)
                    if dic.get(grm, False):
                        dic[grm] += 1
                    else:
                        dic[grm] = 1

                    context = " ".join(gram[:n - 1])
                    suggest = gram[n - 1]
                    if not suggested.get(context, False):
                        suggested[context] = {}
                    if suggested[context].get(suggest, False):
                        suggested[context][suggest] += 1
                    else:
                        suggested[context][suggest] = 1
            else:
                self.total_size = sum(Counter(text).values())
                self.unique_size = len(Counter(text))
                # split text to words and each word to char n-gram and add counter in dict
                for word in text.split():
                    word = "#" * (n - 1) + word + "#" * (n - 1)
                    for gram in nltk.ngrams(word, n):
                        grm = "".join(gram)
                        if dic.get(grm, False):
                            dic[grm] += 1
                        else:
                            dic[grm] = 1

                        context = "".join(gram[:n - 1])
                        suggest = gram[n - 1]
                        if not suggested.get(context, False):
                            suggested[context] = {}
                        if suggested[context].get(suggest, False):
                            suggested[context][suggest] += 1
                        else:
                            suggested[context][suggest] = 1

            self.model_dict = dic
            self.suggested = suggested

        def get_model_dictionary(self):
            """Returns the dictionary class object
            """
            return self.model_dict

        def get_suggested_dictionary(self):
            """Returns the dictionary class object
            """
            return self.suggested

        def get_model_window_size(self):
            """Returning the size of the context window (the n in "n-gram")
            """
            return self.n

        def get_words(self):
            """Returning the unique words list in the text for error fixing
            """
            return self.words

        def generate(self, context=None, n=20):
            """Returns a string of the specified length, generated by applying the language model
            to the specified seed context. If no context is specified the context should be sampled
            from the models' contexts distribution. Generation should stop before the n'th word if the
            contexts are exhausted. If the length of the specified context exceeds (or equal to)
            the specified n, the method should return a prefix of length n of the specified context.

                Args:
                    context (str): a seed context to start the generated string from. Defaults to None
                    n (int): the length of the string to be generated.

                Return:
                    String. The generated text.

            """
            # convenion defs
            win = n
            n = self.n

            if context is None:
                context = ""

            if self.chars:
                word = context
                if len(word) >= win:  # if already exceeds len
                    return word

                # add empty starting to start generate
                word = "#" * (n - 1) + word

                while len(word) < n + win - 1:
                    predict = "".join(tuple(word[len(word) - n + 1:]))  # the last n-1 word
                    # search in dict a candidate and add to the sentence
                    dic = self.suggested[predict]

                    # have chance we will choose less popular word
                    if n == 1:
                        probability = 0.3
                    else:
                        probability = 0.7

                    for key, value in sorted(dic.items(), key=lambda item: item[1], reverse=True):
                        candidate = key
                        if random.random() <= probability:
                            break

                    # contexts is exhausted
                    if candidate == "#":
                        break

                    word += candidate

                # return without the starting empty words
                return word[n - 1:]

            else:
                words = context.split()
                if len(words) >= win:  # if already exceeds len
                    return ' '.join(words[:win])

                # add empty words to start generate
                context = "# " * (n - 1) + context
                words = context.split()

                while len(words) < n + win - 1:
                    predict = " ".join(tuple(words[len(words) - n + 1:]))  # the last n-1 words
                    # search in dict a candidate and add to the sentence
                    dic = self.suggested[predict]

                    # have chance we will choose less popular word
                    if n == 1:
                        probability = 0.3
                    else:
                        probability = 0.7

                    for key, value in sorted(dic.items(), key=lambda item: item[1], reverse=True):
                        candidate = key
                        if random.random() <= probability:
                            break

                    # contexts is exhausted
                    if candidate == "#":
                        break

                    words.append(candidate)

                # return without the starting empty words
                return ' '.join(words[n - 1:])

        def evaluate(self, text):
            """Returns the log-likelihood of the specified text to be a product of the model.
               Laplace smoothing should be applied if necessary.

               Args:
                   text (str): Text to evaluate.

               Returns:
                   Float. The float should reflect the (log) probability.
            """
            n = self.n
            dic = self.model_dict
            suggested = self.suggested

            total = 0
            # pass over the text and sum the log
            if self.chars:  # chars = true
                sequence = text.split()
                if n == 1:
                    for word in sequence:
                        for ch in word:
                            if not dic.get(ch, False):
                                probability = self.smooth(ch)  # laplace
                            else:
                                probability = dic[ch] / self.total_size

                            total += math.log(probability)
                else:
                    for word in sequence:
                        word = "#" * (n - 1) + word
                        for gram in nltk.ngrams(word, n):
                            if not suggested.get(' '.join(gram[:len(gram) - 1]), False) or not dic.get(' '.join(gram), False):
                                probability = self.smooth(' '.join(gram))  # laplace
                            else:
                                probability = dic[' '.join(gram)] / sum(suggested[' '.join(gram[:len(gram) - 1])].values())

                            total += math.log(probability)
            else: # chars = false
                if n == 1:
                    sequence = text.split()
                    for word in sequence:
                        if not dic.get(word, False):
                            probability = self.smooth(word)  # laplace
                        else:
                            probability = dic[word] / self.total_size

                        total += math.log(probability)
                else:
                    sequence = "# " * (n - 1) + text
                    for gram in nltk.ngrams(sequence.split(), n):
                        if not suggested.get(' '.join(gram[:len(gram) - 1]), False) or not dic.get(' '.join(gram), False):
                            probability = self.smooth(' '.join(gram))  # laplace
                        else:
                            probability = dic[' '.join(gram)] / sum(suggested[' '.join(gram[:len(gram) - 1])].values())

                        total += math.log(probability)

            return total

        def smooth(self, ngram):
            """Returns the smoothed (Laplace) probability of the specified ngram.

                Args:
                    ngram (str): the ngram to have it's probability smoothed

                Returns:
                    float. The smoothed probability.
            """
            n = self.n
            dic = self.model_dict
            suggested = self.suggested

            ct = 0  # mone arg ci / c(wn-1,wn)

            # in case of unigram
            if n == 1:
                if dic.get(ngram, False):
                    ct = dic[ngram]
                return (ct + 1) / (self.unique_size + self.total_size)

            # in case of anygram
            cd = 0  # mehana arg c(wn-1)
            gram = tuple(ngram.split())

            if self.chars:
                if suggested.get(''.join(gram[:len(gram) - 1]), False):
                    cd = sum(suggested[''.join(gram[:len(gram) - 1])].values())
                if dic.get(''.join(gram), False):
                    ct = dic[''.join(gram)]
            else:
                if suggested.get(' '.join(gram[:len(gram) - 1]), False):
                    cd = sum(suggested[' '.join(gram[:len(gram) - 1])].values())
                if dic.get(' '.join(gram), False):
                    ct = dic[' '.join(gram)]

            return (ct + 1) / (cd + self.unique_size)


def normalize_text(text):
    """Returns a normalized version of the specified string.
      You can add default parameters as you like (they should have default values!)
      You should explain your decisions in the header of the function.

      I wanted to make a Language Model that support context between two sentences so
      the model is not "clean" the punctuation for the "learning".
      I used model of window size 3 because it give me the best accuracy of test set of
      short sentences of daily life.
      I used the the Tragedy of Hamlet by William Shakespeare for the learning of the
      Language Model because it was short enough to normalize text fast and was
      legacy enough to make the model unique.
      Alpha high enough to not annoying the user with fixes.
      I used gutenberg, names from nltk.corpus for load the Hamlet and Names corpus.
      Name corpus used for more advance fixes because names not found in all text but
      need to not fixed.


      Args:
        text (str): the text to normalize

      Returns:
        string. the normalized text.
    """
    lm = Spell_Checker.Language_Model(n=3, chars=False)
    # nltk.download('gutenberg')  # todo uncomment if not have this on your computer
    corpus = gutenberg.raw("shakespeare-hamlet.txt")
    lm.build_model(corpus)
    spell_checker = Spell_Checker()
    spell_checker.add_language_model(lm)
    ready_error_table = spelling_confusion_matrices.error_tables
    spell_checker.add_error_tables(ready_error_table)
    alpha = 0.85
    return spell_checker.spell_check(text, alpha)
