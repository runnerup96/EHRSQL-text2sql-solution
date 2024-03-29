from sql_metadata import Parser
import re

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit',
                   'intersect', 'union', 'except', 'distinct', 'with', 'over',
                   'partition', 'replace', 'substr', 'row_number', 'cast')
JOIN_KEYWORDS = ('join', 'on', 'as', 'inner', 'left', 'right')

WHERE_OPS = ('not', 'between', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none',)
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg', 'length')

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc', 'by')
TYPES_OPS = ('int')

SQL_SYNTAX_KEYWORDS = CLAUSE_KEYWORDS + JOIN_KEYWORDS + WHERE_OPS + UNIT_OPS + \
                      AGG_OPS + COND_OPS + SQL_OPS + ORDER_OPS


def remove_table_alias(s):
    tables_aliases = Parser(s).tables_aliases
    new_tables_aliases = {}
    for i in range(1, 11):
        if "t{}".format(i) in tables_aliases.keys():
            new_tables_aliases["t{}".format(i)] = tables_aliases["t{}".format(i)]

    tables_aliases = new_tables_aliases
    for k, v in tables_aliases.items():
        s = s.replace("as " + k + " ", "")
        s = s.replace(k, v)

    return s


def white_space_fix(s):
    parsed_s = Parser(s)
    s = " ".join([token.value for token in parsed_s.tokens])
    return s

def lower_sql_keywords(s):
    query_tokens = s.split()
    query_out = []
    for token in query_tokens:
        if token.lower() in SQL_SYNTAX_KEYWORDS:
            lower_token = token.lower()
            query_out.append(lower_token)
        else:
            query_out.append(token)
    query_out_str = " ".join(query_out)
    return query_out_str

def lower_sql(s):
    in_quotation = False
    out_s = ""
    for char in s:
        if in_quotation:
            out_s += char
        else:
            out_s += char.lower()

        if char == "'":
            if in_quotation:
                in_quotation = False
            else:
                in_quotation = True

    return out_s




# remove ";"
def remove_semicolon(s):
    if s.endswith(";"):
        s = s[:-1]
    return s


# double quotation -> single quotation
def double2single(s):
    return s.replace("\"", "'")


def add_asc(s):
    pattern = re.compile(
        r'order by (?:\w+ \( \S+ \)|\w+\.\w+|\w+)(?: (?:\+|\-|\<|\<\=|\>|\>\=) (?:\w+ \( \S+ \)|\w+\.\w+|\w+))*')
    if "order by" in s and "asc" not in s and "desc" not in s:
        for p_str in pattern.findall(s):
            s = s.replace(p_str, p_str + " asc")

    return s

def fix_internal_fails(s):
    s = s.replace('! =', '!=')
    return s



def query_normalization_pipeline(sql):
    """
    adapted from https://github.com/RUCKBReasoning/RESDSQL
    :param sql:
    :return:
    """
    formatted_sql = remove_semicolon(sql)
    formatted_sql = double2single(formatted_sql)
    formatted_sql = white_space_fix(formatted_sql)
    formatted_sql = lower_sql(formatted_sql)
    formatted_sql = add_asc(formatted_sql)
    formatted_sql = remove_table_alias(formatted_sql)
    formatted_sql = fix_internal_fails(formatted_sql)
    formatted_sql = formatted_sql.replace('\t', "")
    formatted_sql = formatted_sql.replace('\n', "")

    return formatted_sql


def normalize_sql_query(query_str):
    query_str = query_str.strip()
    norm_sql = query_normalization_pipeline(query_str)
    return norm_sql


def prepare_model_input(db_id, question, schema_str):
    source = f"{db_id} : {question} {schema_str}"
    return source

def prepare_model_target(db_id, query):
    target = f"{db_id} | {query}"
    return target


def process_input_question(question_str):
    question_str = question_str.replace("\u2018", "'").replace("\u2019", "'") \
        .replace("\u201c", "'").replace("\u201d", "'").replace('\t', "").strip().replace('\n', '')
    return question_str


def query_post_processor(query):
    query = query.split('|')[-1]
    query = query.strip()
    query = query.replace("='", "= '").replace("!=", " !=")
    query = query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')
    query = query.replace("<pad>", "").replace("</s>", "")
    return query



# # лишнее
# def add_separation_spaces(s):
#     out_s = ""
#
#     for char_idx in range(1, len(s) - 1):
#         if s[char_idx] in ['(', ')']:
#             bracket = s[char_idx]
#             if s[char_idx - 1] == ' ' and s[char_idx + 1] == ' ':
#                 continue
#             elif s[char_idx - 1] == ' ' and s[char_idx + 1] != ' ':
#                 out_s += f'{bracket} '
#             elif s[char_idx - 1] != ' ' and s[char_idx + 1] == ' ':
#                 out_s += f' {bracket}'
#             elif s[char_idx - 1] != ' ' and s[char_idx + 1] != ' ':
#                 out_s += f' {bracket} '
#         else:
#             out_s += s[char_idx]
#     return out_s
#
# # лишнее
# # convert everything except text between single quotation marks to lower case and schema items
# def remove_spaces_from_values(s):
#     in_quotation = False
#     out_s = ""
#     for char in s:
#         if in_quotation:
#             out_s += char
#             if char == " ":
#                 out_s += '&&'
#
#         if char == "'":
#             if in_quotation:
#                 in_quotation = False
#             else:
#                 in_quotation = True
#
#     return out_s
