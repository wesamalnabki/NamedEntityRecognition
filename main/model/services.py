import os
import pathlib
import time

import nltk
import torch
import torch.nn.functional as nnf
from torch import cuda
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForTokenClassification

from .utils import collapse, base64_to_string, handle_long_text

MAX_LEN = 512

MODEL_NAME_PATH_ES = os.path.join(pathlib.Path(__file__).parent.resolve(), '../data/ES_NER_11Cat_beto_es_cased')
MODEL_NAME_PATH_EN = os.path.join(pathlib.Path(__file__).parent.resolve(), '../data/EN_TOR_NER_8Cat_bertmulti_cased')
nltk.data.path.append(os.path.join(pathlib.Path(__file__).parent.resolve(), '../data/nltk_data'))


class NamedEntityRecognition:
    """
    A class for Named Entity Recognition in Spanish and English (TOR) network

    The following classes are supported for English:
    -	Products
    -	Organization
    -	Quantities
    -	Person
    -	Time
    -	Events
    -	Money
    -	Location
    -	Language

    The following classes are supported for Spanish:
    - PRODUCTS
    - ORGANIZATION
    - GPE-LOCATION
    - QUANTITIES
    - PERSON
    - FACILITIES
    - EVENTS
    - MONEY
    - LOCATION
    - LANGUAGE
    - GROUP
    """

    def __init__(self):
        # detect processing device
        self.device = 'cuda' if cuda.is_available() else 'cpu'

        self.labels_to_ids_es = {'B-PERSON': 0,
                                 'O': 1,
                                 'B-GPE-LOCATION': 2,
                                 'I-PERSON': 3,
                                 'B-ORGANIZATION': 4,
                                 'I-ORGANIZATION': 5,
                                 'B-MONEY': 6,
                                 'I-GPE-LOCATION': 7,
                                 'B-LANGUAGE': 8,
                                 'B-PRODUCTS': 9,
                                 'I-PRODUCTS': 10,
                                 'B-FACILITIES': 11,
                                 'I-FACILITIES': 12,
                                 'B-EVENTS': 13,
                                 'I-EVENTS': 14,
                                 'B-GROUP': 15,
                                 'I-GROUP': 16,
                                 'I-MONEY': 17,
                                 'B-QUANTITIES': 18,
                                 'I-QUANTITIES': 19,
                                 'B-LOCATION': 20,
                                 'I-LOCATION': 21,
                                 'I-LANGUAGE': 22}

        self.ids_to_labels_es = {v: k for k, v in self.labels_to_ids_es.items()}

        self.labels_to_ids_en = {'O': 0,
                                 'B-DRG': 1,
                                 'I-DRG': 2,
                                 'B-CUR': 3,
                                 'B-ORG': 4,
                                 'I-ORG': 5,
                                 'B-DAT': 6,
                                 'I-DAT': 7,
                                 'B-LOC': 8,
                                 'B-PER': 9,
                                 'I-PER': 10,
                                 'I-LOC': 11,
                                 'B-MISC': 12,
                                 'I-MISC': 13,
                                 'B-WEP': 14,
                                 'I-WEP': 15}

        self.ids_to_labels_en = {v: k for k, v in self.labels_to_ids_en.items()}

    def load_model_ner_es(self):
        """
        A function to load NER model for Spanish
        :return: None
        """
        # Load NER Spanish model
        self.tokenizer_es = BertTokenizerFast.from_pretrained(MODEL_NAME_PATH_ES, do_lower_case=False)
        self.model_es = BertForTokenClassification.from_pretrained(MODEL_NAME_PATH_ES,
                                                                   num_labels=len(self.ids_to_labels_es))
        self.model_es.to(self.device)

    @staticmethod
    def _decode_input(input_text, input_encoding):
        """
        Functin to decode input
        :param input_text: string input text
        :param input_encoding: encoding used to generate the text. It must be either BASE64 or UTF-8
        :return:
        """
        if input_encoding == "BASE64":
            conversion_code, text = base64_to_string(input_text)

            if conversion_code == 200:
                return text, conversion_code

            if conversion_code == 500:
                return "error while decoding base64 text", conversion_code

        elif input_encoding == "UTF-8":
            input_text = input_text.encode('utf-8').decode('utf-8')
            return input_text, 200
        else:
            return "unknown decoding format. It must be UTF-8 or BASE64", 500

        return input_text, 200

    def recognize_named_entities_es(self, text, text_encoding='UTF-8'):
        """
        A function to recognize named entities in Spanish text
        :param text: Spanish text
        :param text_encoding: text encoding which can be either "BASE64" or "UTF-8"
        :return: if the model detected the entities correctly in text, it returns a dictionary with three keys
            - response: a dictionary where the keys refer to the entity types and the values refer to the detected
             entities
            - spent_time: a float refers to the processing time
            - result_collapsed: a list of the detected entities. Each item in the list has the text of the entity, start
            index, end index, and a confidence score.

            if the request was not processed successfully, the function return a dictionary with two keys:
            - error: a message describing the error
            - spent_time: a float refers to the processing time
        """

        t = time.time()

        text = text.encode().decode('utf-8-sig')

        text, decoding_code = self._decode_input(input_text=text, input_encoding=text_encoding)

        if decoding_code != 200:
            return {
                "error": text,
                "spent_time": time.time() - t
            }

        if not text.strip():
            return {
                'error': "missing input text",
                'spent_time': time.time() - t
            }

        offset = 0
        collapsed = []
        unique_entity_names = {k[2:]: [] for k in self.labels_to_ids_es.keys() if k[2:]}

        text_list = handle_long_text(text)

        for sub_text in tqdm(text_list):

            sent_tokens = nltk.word_tokenize(sub_text)
            inputs = self.tokenizer_es(sent_tokens,
                                       is_split_into_words=True,
                                       return_offsets_mapping=True,
                                       padding='max_length',
                                       truncation=True,
                                       max_length=MAX_LEN,
                                       return_tensors="pt")

            # move to gpu
            ids = inputs["input_ids"].to(self.device, dtype=torch.long)
            mask = inputs["attention_mask"].to(self.device, dtype=torch.long)

            # forward pass
            with torch.no_grad():
                outputs = self.model_es(ids, attention_mask=mask)
            logits = outputs[0]

            active_logits = logits.view(-1, self.model_es.num_labels)  # shape (batch_size * seq_len, num_labels)
            prob = nnf.softmax(active_logits, dim=1)
            top_p, top_class = prob.topk(1, dim=1)

            flattened_predictions = torch.argmax(active_logits, dim=1)

            tokens = self.tokenizer_es.convert_ids_to_tokens(ids.squeeze().tolist())
            token_predictions = [self.ids_to_labels_es[i] for i in flattened_predictions.cpu().numpy()]
            wp_preds = list(zip(tokens, token_predictions, [round(float(x), 2) for x in top_p.cpu().detach().numpy()]))

            prediction = []
            for (tk, token_pred, tag_conf), mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
                # only predictions on first word pieces are important
                if mapping[0] == 0 and mapping[1] != 0:
                    prediction.append((token_pred, tag_conf))
                else:
                    continue

            result = [[word, tag, idx, idx + 1, conf] for idx, (word, (tag, conf)) in
                      enumerate(zip(sent_tokens, prediction))]

            result_collapsed = collapse(result)

            for elm in result_collapsed:
                tag_name = elm[1]
                if tag_name in unique_entity_names:
                    entity = elm[0]
                    unique_entity_names[tag_name].append(entity)

            collapsed.append([[x, y, z + offset, w + offset, h] for (x, y, z, w, h) in result_collapsed])
            offset += len(sent_tokens)

        spent_time = time.time() - t
        return {
            'response': unique_entity_names,
            'result_collapsed': [element for tup in collapsed for element in tup],
            'spent_time': spent_time
        }

    def unload_model_ner_es(self):
        """
        Function to remove the Spanish NER model from the memory
        :return: None
        """
        del self.tokenizer_es
        del self.model_es
        torch.cuda.empty_cache()

    def load_model_ner_tor(self):
        """
        Function to load TOR NER model to memory
        :return:
        """
        # Load NER English model
        self.tokenizer_en = BertTokenizerFast.from_pretrained(MODEL_NAME_PATH_EN, do_lower_case=False)
        self.model_en = BertForTokenClassification.from_pretrained(MODEL_NAME_PATH_EN,
                                                                   num_labels=len(self.ids_to_labels_en))
        self.model_en.to(self.device)

    def recognize_named_entities_tor(self, text, text_encoding='UTF-8'):
        """
        A function to recognize named entities in English text
        :param text: English text
        :param text_encoding: text encoding which can be either "BASE64" or "UTF-8"
        :return: if the model detected the entities correctly in text, it returns a dictionary with three keys
            - response: a dictionary where the keys refer to the entity types and the values refer to the detected entities
            - spent_time: a float refers to the processing time
            - result_collapsed: a list of the detected entities. Each item in the list has the text of the entity, start
            index, end index, and a confidence score.

            if the request was not processed successfully, the function return a dictionary with two keys:
            - error: a message describing the error
            - spent_time: a float refers to the processing time
        """

        t = time.time()

        text, decoding_code = self._decode_input(input_text=text, input_encoding=text_encoding)

        if decoding_code != 200:
            return {
                "error": text,
                "spent_time": time.time() - t
            }

        if not text.strip():
            return {
                'error': "missing input text",
                'spent_time': time.time() - t
            }

        offset = 0
        collapsed = []
        unique_entity_names = {k[2:]: [] for k in self.labels_to_ids_en.keys() if k[2:]}

        text_list = handle_long_text(text)
        for sub_text in tqdm(text_list):
            sent_tokens = nltk.word_tokenize(sub_text)
            inputs = self.tokenizer_en(sent_tokens,
                                       is_split_into_words=True,
                                       return_offsets_mapping=True,
                                       padding='max_length',
                                       truncation=True,
                                       max_length=MAX_LEN,
                                       return_tensors="pt")

            # move to gpu
            ids = inputs["input_ids"].to(self.device, dtype=torch.long)
            mask = inputs["attention_mask"].to(self.device, dtype=torch.long)

            # forward pass
            with torch.no_grad():
                outputs = self.model_en(ids, attention_mask=mask)
            logits = outputs[0]

            active_logits = logits.view(-1, self.model_en.num_labels)  # shape (batch_size * seq_len, num_labels)
            prob = nnf.softmax(active_logits, dim=1)
            top_p, top_class = prob.topk(1, dim=1)

            flattened_predictions = torch.argmax(active_logits, dim=1)

            tokens = self.tokenizer_en.convert_ids_to_tokens(ids.squeeze().tolist())
            token_predictions = [self.ids_to_labels_en[i] for i in flattened_predictions.cpu().numpy()]
            wp_preds = list(zip(tokens, token_predictions, [round(float(x), 2) for x in top_p.cpu().detach().numpy()]))

            prediction = []
            for (tk, token_pred, tag_conf), mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
                # only predictions on first word pieces are important
                if mapping[0] == 0 and mapping[1] != 0:
                    prediction.append((token_pred, tag_conf))
                else:
                    continue

            result = [[word, tag, idx, idx + 1, conf] for idx, (word, (tag, conf)) in
                      enumerate(zip(sent_tokens, prediction))]

            result_collapsed = collapse(result)

            for elm in result_collapsed:
                tag_name = elm[1]
                if tag_name in unique_entity_names:
                    entity = elm[0]
                    unique_entity_names[tag_name].append(entity)

            collapsed.append([[x, y, z + offset, w + offset, h] for (x, y, z, w, h) in result_collapsed])
            offset += len(sent_tokens)

        spent_time = time.time() - t
        return {
            'response': unique_entity_names,
            'result_collapsed': [element for tup in collapsed for element in tup],
            'spent_time': spent_time
        }

    def unload_model_ner_tor(self):
        """
        Function to unload NER TOR English model from memory
        :return: None
        """
        del self.tokenizer_en
        del self.model_en
        torch.cuda.empty_cache()
