#!/usr/bin/python3
# -*- coding: utf-8 -*-

import inspect
import json
import re
import unittest

from main.model.services import NamedEntityRecognition

__author__ = 'Wesam Alnabki'


class TestPDE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ner = NamedEntityRecognition()

        with open('test/data/data.json', 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)

        cls.data = data
        cls.test_method_regex = r'test_\w{3}_(.*)_\d{3}'

    def setUp(self):
        pass

    def get_input(self, test_id):

        test_index = int(test_id[-3:]) - 1
        tested_method = re.match(self.test_method_regex, test_id).group(1)

        test_input = self.data['test'][tested_method][test_index]

        if test_input['id'] == test_id:
            text = test_input['arguments'].get('text')
            text_encoding = test_input['arguments'].get('text_encoding')
            return text, text_encoding
        else:
            raise (ValueError("Wrong test ID, check the json file"))

    def test_NER_recognize_named_entities_ES_001(self):
        """
        NER_recognize_named_entities_ES_001
        """

        test_id = inspect.stack()[0][3]
        text, text_encoding = self.get_input(test_id)

        self.ner.load_model_ner_es()
        detected_ne = self.ner.recognize_named_entities_es(text, text_encoding)
        self.ner.unload_model_ner_es()

        self.assertEqual(len(detected_ne['response'].keys()), 11)
        self.assertEqual(len(detected_ne), 3)

    def test_NER_recognize_named_entities_ES_002(self):
        """
        NER_recognize_named_entities_ES_002
        """

        test_id = inspect.stack()[0][3]
        text, text_encoding = self.get_input(test_id)

        self.ner.load_model_ner_es()
        detected_ne = self.ner.recognize_named_entities_es(text, text_encoding)
        self.ner.unload_model_ner_es()

        self.assertEqual(len(detected_ne['response'].keys()), 11)
        self.assertEqual(len(detected_ne), 3)

    def test_NER_recognize_named_entities_ES_003(self):
        """
        NER_recognize_named_entities_ES_003
        """

        test_id = inspect.stack()[0][3]
        text, text_encoding = self.get_input(test_id)

        self.ner.load_model_ner_es()
        detected_ne = self.ner.recognize_named_entities_es(text, text_encoding)
        self.ner.unload_model_ner_es()

        self.assertEqual(detected_ne.get('response'), None)
        self.assertEqual(len(detected_ne), 2)

    def test_NER_recognize_named_entities_ES_004(self):
        """
        NER_recognize_named_entities_ES_004
        """

        test_id = inspect.stack()[0][3]
        text, text_encoding = self.get_input(test_id)

        self.ner.load_model_ner_es()
        detected_ne = self.ner.recognize_named_entities_es(text, text_encoding)
        self.ner.unload_model_ner_es()

        self.assertEqual(detected_ne.get('response'), None)
        self.assertEqual(len(detected_ne), 2)

    def test_NER_recognize_named_entities_ES_005(self):
        """
        NER_recognize_named_entities_ES_005
        """

        test_id = inspect.stack()[0][3]
        text, text_encoding = self.get_input(test_id)

        self.ner.load_model_ner_es()
        detected_ne = self.ner.recognize_named_entities_es(text, text_encoding)
        self.ner.unload_model_ner_es()

        self.assertEqual(detected_ne.get('response'), None)
        self.assertEqual(len(detected_ne), 2)

    def test_NER_recognize_named_entities_ES_006(self):
        """
        NER_recognize_named_entities_ES_006
        """

        test_id = inspect.stack()[0][3]
        text, text_encoding = self.get_input(test_id)

        self.ner.load_model_ner_es()
        detected_ne = self.ner.recognize_named_entities_es(text, text_encoding)
        self.ner.unload_model_ner_es()

        self.assertEqual(len(detected_ne['response'].keys()), 11)
        self.assertEqual(len(detected_ne), 3)

    def test_NER_recognize_named_entities_TOR_001(self):
        """
        NER_recognize_named_entities_TOR_001
        """

        test_id = inspect.stack()[0][3]
        text, text_encoding = self.get_input(test_id)

        self.ner.load_model_ner_tor()
        detected_ne = self.ner.recognize_named_entities_tor(text, text_encoding)
        self.ner.unload_model_ner_tor()

        self.assertEqual(len(detected_ne['response'].keys()), 8)
        self.assertEqual(len(detected_ne), 3)

    def test_NER_recognize_named_entities_TOR_002(self):
        """
        NER_recognize_named_entities_TOR_002
        """

        test_id = inspect.stack()[0][3]
        text, text_encoding = self.get_input(test_id)

        self.ner.load_model_ner_tor()
        detected_ne = self.ner.recognize_named_entities_tor(text, text_encoding)
        self.ner.unload_model_ner_tor()

        self.assertEqual(len(detected_ne['response'].keys()), 8)
        self.assertEqual(len(detected_ne), 3)

    def test_NER_recognize_named_entities_TOR_003(self):
        """
        NER_recognize_named_entities_TOR_003
        """

        test_id = inspect.stack()[0][3]
        text, text_encoding = self.get_input(test_id)

        self.ner.load_model_ner_tor()
        detected_ne = self.ner.recognize_named_entities_tor(text, text_encoding)
        self.ner.unload_model_ner_tor()

        self.assertEqual(detected_ne.get('response'), None)
        self.assertEqual(len(detected_ne), 2)

    def test_NER_recognize_named_entities_TOR_004(self):
        """
        NER_recognize_named_entities_TOR_004
        """

        test_id = inspect.stack()[0][3]
        text, text_encoding = self.get_input(test_id)

        self.ner.load_model_ner_tor()
        detected_ne = self.ner.recognize_named_entities_tor(text, text_encoding)
        self.ner.unload_model_ner_tor()

        self.assertEqual(detected_ne.get('response'), None)
        self.assertEqual(len(detected_ne), 2)

    def test_NER_recognize_named_entities_TOR_005(self):
        """
        NER_recognize_named_entities_TOR_005
        """

        test_id = inspect.stack()[0][3]
        text, text_encoding = self.get_input(test_id)

        self.ner.load_model_ner_tor()
        detected_ne = self.ner.recognize_named_entities_tor(text, text_encoding)
        self.ner.unload_model_ner_tor()

        self.assertEqual(detected_ne.get('response'), None)
        self.assertEqual(len(detected_ne), 2)

    def test_NER_recognize_named_entities_TOR_006(self):
        """
        NER_recognize_named_entities_TOR_006
        """

        test_id = inspect.stack()[0][3]
        text, text_encoding = self.get_input(test_id)

        self.ner.load_model_ner_tor()
        detected_ne = self.ner.recognize_named_entities_tor(text, text_encoding)
        self.ner.unload_model_ner_tor()

        self.assertEqual(len(detected_ne['response'].keys()), 8)
        self.assertEqual(len(detected_ne), 3)


# Executing the tests in the above test case class
if __name__ == "__main__":
    unittest.main()
