import argparse
import deepl
import time 
import random

from muss.simplify import ALLOWED_MODEL_NAMES, simplify_sentences
from muss.utils.helpers import read_lines

auth_key = '956fd93f-7ff3-4cc9-8ecd-a1c091cf25cd:fx'  # Replace with your DeepL API key
translator = deepl.Translator(auth_key)

def safe_translate(translator, text, source_lang, target_lang):
    try:
        return translator.translate_text(text, source_lang=source_lang, target_lang=target_lang).text, None  # Return the translated text and no error
    except Exception as e:
        return text, e  # Return the original text and the error

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines if line.strip()]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simplify a file line by line.')
    parser.add_argument('--filepath', type=str, default="uk_input.txt", help='File containing the source sentences, one sentence per line.')
    parser.add_argument('--model-name', type=str, default=ALLOWED_MODEL_NAMES[0], choices=ALLOWED_MODEL_NAMES, help=f'Model name to generate from. Models selected with the highest validation SARI score.')
    args = parser.parse_args()
    
    # Read input file
    content = read_file(args.filepath)
    
    # Translate from Ukrainian to English
    source_sentences = []
    with open("en_input.txt", "w+", encoding='utf-8') as f_en:
        for sent in content:
            translated, error = safe_translate(translator, sent, source_lang="UK", target_lang="EN-US")
            if error:
                print(f"Error translating from Ukrainian to English: {error}")
            else:
                f_en.write(translated + '\n')
            source_sentences.append(translated)
    print("Translation to English ready!")

    # Simplify sentences
    pred_sentences = simplify_sentences(source_sentences, model_name=args.model_name)
    
    # Save simplified sentences
    with open("simplified_sentences.txt", "w+", encoding='utf-8') as f_simplified:
        for simplified in pred_sentences:
            f_simplified.write(simplified + '\n')
    print("Simplified sentences saved!")

    # Translate back from English to Ukrainian
    with open("output_dpl.txt", "w+", encoding='utf-8') as f_out:
        for original, simplified in zip(source_sentences, pred_sentences):
            if not simplified.strip():
                answer = original
            else:
                answer, error = safe_translate(translator, simplified, source_lang="EN", target_lang="UK")
                if error:
                    print(f"Error translating from English to Ukrainian: {error}")
                    answer = simplified
            print(f"Original: {original}\nSimplified: {simplified}\nTranslated back: {answer}\n")
            f_out.write(answer + '\n')
    print("Translation back to Ukrainian ready!")