import preprocess.question_query_processing_utils as processing_utils
import collections

def get_schema_string(table_json):
    """Returns the schema serialized as a string."""
    table_id_to_column_names = collections.defaultdict(list)
    for table_id, name in table_json["column_names_original"]:
        table_id_to_column_names[table_id].append(name.lower())
    tables = table_json["table_names_original"]

    table_strings = []
    for table_id, table_name in enumerate(tables):
        column_names = table_id_to_column_names[table_id]
        table_string = "| %s : %s" % (table_name.lower(), " , ".join(column_names))
        table_strings.append(table_string)
    result_string = "".join(table_strings).lower().replace('\t', "").strip()
    return result_string

def write_tsv(examples, filename, expected_num_columns=2):
    """Write examples to tsv file."""
    with open(filename, "w") as tsv_file:
        for example in examples:
            if len(example) != expected_num_columns:
                raise ValueError("Example '%s' has %s columns." %
                                 (example, len(example)))
            example = "\t".join(example)
            line = "%s\n" % example
            tsv_file.write(line)
    print("Wrote %s examples to %s." % (len(examples), filename))