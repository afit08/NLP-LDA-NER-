from flask import Flask, request, jsonify
import stanza
from deep_translator import GoogleTranslator
from langdetect import detect_langs
from transformers import pipeline
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty

# Initialize Flask app
app = Flask(__name__)

# Initialize NLP models
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
summary_model = pipeline("summarization", model="Falconsai/text_summarization", device=0)

# Download Stanza resources
stanza.download('en')

# Queues for communication with the Stanza thread
task_queue = Queue()
result_queue = Queue()

# Stanza worker thread function
def stanza_worker():
    stanza_pipeline = stanza.Pipeline(lang='en', processors='tokenize,ner', use_gpu=True)
    while True:
        try:
            text = task_queue.get(timeout=1)
            doc = stanza_pipeline(text)
            result_queue.put(doc)
            task_queue.task_done()
        except Empty:
            continue
        except Exception as e:
            result_queue.put({"error": str(e)})
            task_queue.task_done()

# Start the Stanza worker thread
thread = Thread(target=stanza_worker, daemon=True)
thread.start()

# Named entity recognition function
def named_entities(translated_text):
    try:
        task_queue.put(translated_text)
        doc = result_queue.get()
        if isinstance(doc, dict) and "error" in doc:
            raise Exception(doc["error"])
        entities = []
        for ent in doc.entities:
            translated_ent = GoogleTranslator(source='auto', target='id').translate(ent.text)
            entity_type = ent.type if ent.type not in ["CARDINAL", "ORDINAL"] else "NUMBER"
            if ent.type in ["GPE", "LOCATION"]:
                entity_type = "LOCATION"
            entities.append({"text": translated_ent, "type": entity_type})
        return entities
    except Exception as e:
        return [{"error": str(e)}]

@app.route('/nlp', methods=['POST'])
def nlp():
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({'error': 'text is required'}), 400
    
    try:
        translatedENG = GoogleTranslator(source='auto', target='en').translate(text)
        lang_detect = detect_langs(text)
        resultLanguage = [{"lang": lang.lang, "score": lang.prob} for lang in lang_detect]

        emotion_labels = ['Angry', 'Sadness', 'Joy', 'Fear']
        hateSpeech_labels = ['Good Speech', 'Hate Speech', 'Neutral']
        newsTopic_labels = ["Politics", "Economy", "Technology", "Health", "Environment", "Science", "Entertainment", "Sports", "Education", "Global Issues"]

        with ThreadPoolExecutor() as executor:
            emotion_future = executor.submit(classifier, translatedENG, emotion_labels, multi_label=True)
            hateSpeech_future = executor.submit(classifier, translatedENG, hateSpeech_labels, multi_label=True)
            newsTopic_future = executor.submit(classifier, translatedENG, newsTopic_labels, multi_label=True)

            emotion = emotion_future.result()
            hateSpeech = hateSpeech_future.result()
            newsTopic = newsTopic_future.result()

        resultEmotion = {"emotion": emotion['labels'][0], "score": emotion['scores'][0]}
        resultHateSpeech = {"label": hateSpeech['labels'][0], "score": hateSpeech['scores'][0]}
        resultNewsTopic = {"topic": newsTopic['labels'][0], "score": newsTopic['scores'][0]}

        named_entities_result = named_entities(translatedENG)

        return jsonify({
            "language": resultLanguage,
            "hate_speech": resultHateSpeech,
            "news_topic": resultNewsTopic,
            "emotion": resultEmotion,
            'named_entities': named_entities_result,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/summary', methods=['POST'])
def summaryz():
    try:
        data = request.json
        text = data.get('text')
        
        if not text:
            return jsonify({'error': 'text is required'}), 400
        
        result = summary_model(text)
        
        return jsonify({
            "message": "Summary Conversation",
            "status": 200,
            "data": result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6000)
