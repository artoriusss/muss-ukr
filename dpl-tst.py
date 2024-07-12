import deepl
import time 
import random
from deep_translator import GoogleTranslator
import os

auth_key = os.get_env('DPL_API_KEY')
dpl = deepl.Translator(auth_key)

uk_to_en_gt = GoogleTranslator(source='uk', target='en')
en_to_uk_gt = GoogleTranslator(source='en', target='uk')

def safe_translate(translator, text, source_lang, target_lang):
    print(f"Translating from {source_lang} to {target_lang}: {text}")
    result = translator.translate_text(text, source_lang=source_lang, target_lang=target_lang).text
    time.sleep(0.5 + random.random())
    return result

def gt(translator, text):
    result = translator.translate(text)
    time.sleep(0.5 + random.random())
    return result

if __name__ == '__main__':
    example_sentence = "Це тестове речення."
    
    translated_to_en = gt(uk_to_en_gt, example_sentence)
    print(f"Translated to English: {translated_to_en}")
    
    translated_back_to_uk = gt(en_to_uk_gt, translated_to_en)
    print(f"Translated back to Ukrainian: {translated_back_to_uk}")
