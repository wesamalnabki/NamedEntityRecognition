from flask import Flask, request

from model.services import NamedEntityRecognition

app = Flask(__name__)

named_entity_recognition_service = NamedEntityRecognition()


@app.route('/ner', methods=['POST'])
def spanish_ner():
    data = request.json
    encoding = data.get('encoding', 'UTF-8')
    text = data.get('text', '')
    lang = data.get('lang', '')

    if lang == 'en':
        return named_entity_recognition_service.recognize_named_entities_en(text=text, text_encoding=encoding)
    elif lang == 'es':
        return named_entity_recognition_service.recognize_named_entities_es(text=text, text_encoding=encoding)
    else:
        return {
            'error': "Please select the language either Spanish 'es' or English 'en'"
        }


if __name__ == '__main__':
    app.run()
