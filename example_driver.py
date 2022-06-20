import spell_checker
import spelling_confusion_matrices

text = 'I will not eat them in the rain. I will not eat them on a train. ' \
       'Not in the dark! Not in a tree! Not in a car! You let me be!' \
       'I do not like them in a box. I do not like them with a fox.' \
       'I will not eat them in a house. I do not like them with a mouse. ' \
       'I do not like them here or there. I do not like them anywhere!' \
       'I do not like green eggs and ham! I do not like them, Sam-I-am.'

# feel free the use the entire story:
# https://www.clear.rice.edu/comp200/resources/texts/Green%20Eggs%20and%20Ham.txt

if __name__ == '__main__':
    # Building the language model
    lm = spell_checker.Spell_Checker.Language_Model(n=2, chars=False)
    lm.build_model(text)

    # Checking the model's dictionary
    d = lm.get_model_dictionary()
    assert d['not like'] >= 1

    # Generate sentence
    sentence = lm.generate(context='will not', n=10)
    print(sentence)

    # Evaluating sentences
    likely_text = 'I do not like'
    unlikely_text = 'I really like'
    assert lm.evaluate(likely_text) > lm.evaluate(unlikely_text)

    # Prepering Spell_Checker instance
    spell_checker = spell_checker.Spell_Checker()
    spell_checker.add_language_model(lm)
    ready_error_table = spelling_confusion_matrices.error_tables
    spell_checker.add_error_tables(ready_error_table)

    alpha = 0.7
    # test sentence
    err_sentence = 'heere or there'
    right_sentence = 'here or there'
    correction = spell_checker.spell_check(err_sentence, alpha)
    assert correction == right_sentence

    # test word 1
    correction = spell_checker.spell_check('heere', alpha)
    assert correction == 'here'

    # test word 2
    correction = spell_checker.spell_check('tre', alpha)
    assert correction == 'tree'
