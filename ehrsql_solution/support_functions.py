from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json
import sqlite3

punctuation_set = set(string.punctuation + '«»—')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

null_starting_words = json.load(open("statics/starting_null_words.json", 'r'))


def lemmatize(word):
    return lemmatizer.lemmatize(word)


def exclude_stop_words(question):
    word_tokens = word_tokenize(question)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    # with no lower case conversion

    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    result_sentence = " ".join(filtered_sentence)
    return result_sentence


def clean_punctuation(input_text: str) -> str:
    space = " "

    # replace punctuation with spaces
    cleaned_string = "".join([c if c not in punctuation_set else space for c in input_text])
    # remove duplicate whitespaces
    cleaned_string = space.join(cleaned_string.split())
    return cleaned_string


def lemmatize_sentence(question):
    question_tokens = question.split()
    result = []
    for word in question_tokens:
        lemma = lemmatize(word)
        result.append(lemma)
    result_sentence = " ".join(result)
    return result_sentence


def preprocess_question(question):
    # удалить все знаки препинания
    # удалить пробелы
    processed_question = clean_punctuation(question)
    processed_question = exclude_stop_words(processed_question)
    processed_question = lemmatize_sentence(processed_question)
    processed_question = processed_question.lower()
    return processed_question

def clean_word_punctuation(input_text: str) -> str:
    # replace punctuation with spaces
    cleaned_string = "".join([c if c not in punctuation_set else "" for c in input_text])
    return cleaned_string

def process_word(word):
    pr_word = str(word)
    pr_word = clean_word_punctuation(pr_word)
    pr_word = lemmatize(pr_word)
    pr_word = pr_word.lower()
    return pr_word


def get_starting_words_status(question):
    processed_question = preprocess_question(question)

    status = False
    for word in null_starting_words:
        if processed_question.startswith(word):
            status = True
            break

    return status


def get_processed_question_length(question):
    processed_question = preprocess_question(question)
    length = len(processed_question.split(" "))
    return length


def execute_query(sql, DB_PATH):
    con = sqlite3.connect(DB_PATH)
    con.text_factory = lambda b: b.decode(errors="ignore")
    cur = con.cursor()
    result = None
    try:
        result = cur.execute(sql).fetchall()
        con.close()
    except sqlite3.OperationalError as e:
        # print(e)
        pass
    return result


def check_sql_result(sql_result):
    result = True
    if sql_result is None:
        result = False

    elif len(sql_result) > 0 and sql_result[0][0] == None:
        result = False

    elif len(sql_result) == 0:
        result = False

    elif len(sql_result) > 0 and sql_result[0][0] == 0:
        result = False

    elif sql_result == 0:
        result = False

    return result


def write_json(result_dict, run_name):
    json.dump(result_dict, open(run_name, 'w'), ensure_ascii=False, indent=4)
    print('File written!')


def make_prediction_with_meta_model(id_, t5_preds, meta_model_preds, db_path):
    """
    We get the predictions from meta_model_prediction.
    If not null - we pass sql through T5-model and verify model result.
    If success - we use generated SQL, else pass null.
    :param id_:
    :param t5_preds:
    :param meta_model_preds:
    :param db_path:
    :return: str
    """
    sql = t5_preds[id_]['sql']
    meta_model_decision = meta_model_preds[id_]
    result = 'null'
    if meta_model_decision != 'null':
        sql_result = execute_query(sql, db_path)
        sql_check = check_sql_result(sql_result)
        if sql_check:
            result = sql
    return result


def get_all_consequtive_words_from_str(word_list):
    substring_list = []
    for i in range(len(word_list)):
        for j in range(i + 1, len(word_list) + 1):
            word_substring = word_list[i:j]
            substring_list.append(word_substring)
    substring_set = set([" ".join(a) for a in substring_list])
    return substring_set


def find_match(query, db_elements):
    processed_query = preprocess_question(query)
    substr_list = get_all_consequtive_words_from_str(processed_query.split())
    intersect = substr_list.intersection(db_elements)
    return intersect
