import base64
import binascii

import nltk


def handle_long_text(text, max_seq_len=450):
    if len(text.split()) > max_seq_len:
        text_list = long_str_to_list_str(text)
        all_text = ''
        new_list = []
        for sent in text_list:
            all_text_tmp = all_text + ' ' + sent
            if len(all_text_tmp.split()) > max_seq_len:
                new_list.append(all_text)
                all_text = sent
            else:
                all_text = all_text_tmp
        return new_list
    else:
        return [text]


def long_str_to_list_str(text, max_sent_len=500):
    def get_split(text1, max_len=500):
        l_total = []
        if len(text1.split()) // 150 > 0:
            n = len(text1.split()) // 150
        else:
            n = 1
        for w in range(n):
            if w == 0:
                l_parcial = text1.split()[:max_len]
                l_total.append(" ".join(l_parcial))
            else:
                l_parcial = text1.split()[w * 150:w * 150 + max_len]
                l_total.append(" ".join(l_parcial))
        return l_total

    sentences = nltk.sent_tokenize(text)
    sub_sentences = [get_split(sent, max_len=max_sent_len) if len(sent.split()) > max_sent_len else [sent] for sent in
                     sentences]
    sub_sentences = [item for items in sub_sentences for item in items]

    return sub_sentences


def base64_to_string(b):
    """
    Function to decode base64
    :param b: input text
    :return: decoded text
    """

    try:
        encoded_text = base64.b64decode(b)
    except binascii.Error:
        return 500, ''

    try:
        decoded_text = encoded_text.decode('utf-8')
        return 200, decoded_text
    except UnicodeDecodeError:
        return 500, ''


def collapse(ner_result):
    """
    Function to collaps BERT output, i.e. de-tokenizer
    :param ner_result: the output of BERT NER model
    :return: list of entities
    """
    # List with the result
    collapsed_result = []

    # Buffer for tokens belonging to the most recent entity
    current_entity_tokens = []
    current_entity = None

    current_start = None
    current_end = None
    current_score = 0.0

    # Iterate over the tagged tokens
    for idx, (token, tag, st, en, score) in enumerate(ner_result):

        if tag == "O":
            continue

        # If an entity span starts ...
        if tag.startswith("B-"):
            # ... if we have a previous entity in the buffer, store it in the result list
            if current_entity is not None:
                collapsed_result.append(
                    (" ".join(current_entity_tokens).replace(' ##', ''), current_entity, current_start, current_end,
                     current_score / len(current_entity_tokens)))

            current_entity = tag[2:]
            current_start = st
            current_end = en
            current_score = score
            # The new entity has so far only one token
            current_entity_tokens = [token]

        elif current_entity is None:
            if tag.startswith('I-'):
                current_entity_tokens = [token]
                current_entity = tag[2:]
                current_start = en - 1
                current_end = en
                current_score += score

        # If the entity continues ...
        elif current_entity is not None:
            if tag == "I-" + current_entity:
                # Just add the token buffer
                current_entity_tokens.append(token)
                current_end = en
                current_score += score

            elif tag != "I-" + current_entity:
                # Just add the token buffer
                current_entity = tag[2:]
                current_entity_tokens = [token]
                current_end = en
                current_score = score
        else:
            raise ValueError("Invalid tag order.")

    # The last entity is still in the buffer, so add it to the result
    # ... but only if there were some entity at all
    if current_entity is not None:
        collapsed_result.append(
            (" ".join(current_entity_tokens).replace(' ##', ''), current_entity, current_start, current_end,
             current_score / len(current_entity_tokens)))
    return collapsed_result
