from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import math
import torch
import wikipedia
from newspaper import Article, ArticleException
from GoogleNews import GoogleNews
from pyvis.network import Network
import IPython

from KB import KB


def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
    return model, tokenizer

def extract_relations_from_model_output(text):
    relations = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        relations.append({
            'head': subject.strip(),
            'type': relation.strip(),
            'tail': object_.strip()
        })
    return relations



# build a knowledge base from text
def from_small_text_to_kb(text, model, tokenizer, verbose=False):
    kb = KB()

    # Tokenizer text
    model_inputs = tokenizer(text, max_length=512, padding=True, truncation=True,
                            return_tensors='pt')
    if verbose:
        print(f"Num tokens: {len(model_inputs['input_ids'][0])}")

    # Generate
    gen_kwargs = {
        "max_length": 216,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": 3
    }
    generated_tokens = model.generate(
        **model_inputs,
        **gen_kwargs,
    )
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    # create kb
    for sentence_pred in decoded_preds:
        relations = extract_relations_from_model_output(sentence_pred)
        for r in relations:
            kb.add_relation(r)

    return kb

# test the `from_small_text_to_kb` function

text = "Napoleon Bonaparte (born Napoleone di Buonaparte; 15 August 1769 – 5 " \
"May 1821), and later known by his regnal name Napoleon I, was a French military " \
"and political leader who rose to prominence during the French Revolution and led " \
"several successful campaigns during the Revolutionary Wars. He was the de facto " \
"leader of the French Republic as First Consul from 1799 to 1804. As Napoleon I, " \
"he was Emperor of the French from 1804 until 1814 and again in 1815. Napoleon's " \
"political and cultural legacy has endured, and he has been one of the most " \
"celebrated and controversial leaders in world history."

model, tokenizer = load_model()
kb = from_small_text_to_kb(text, model, tokenizer, verbose=True)
kb.print()