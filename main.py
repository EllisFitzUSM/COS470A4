from ranx import Run, Qrels, evaluate
from bs4 import BeautifulSoup as bs
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration, AutoConfig, AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
import argparse as ap
import string
import random
import torch
import json
import os
import re

# Inspiration from Arvin Zhuang Pair-wise

os.environ['TRANSFORMERS_CACHE'] = '/mnt/netstore1_home/'
os.environ['HF_HOME'] = '/mnt/netstore1_home/ellis.fitzgerald/HF'

model_name = 'meta-llama/Llama-3.1-8B-Instruct'

prompt_1 = '''
Given a query '{query}', which of the following two documents is more relevant to the query?

Document 1: '{doc_1}'

Document 2: '{doc_2}'

Output Document 1 or Document 2:
'''

answers = {}
tokenizer = None
device = None
llm_model = None

def main():
    global answers, topics, tokenizer, device, llm_model
    argp = ap.ArgumentParser()
    argp.add_argument('answers', type=str, help='Path to Answer file.')
    argp.add_argument('qrel_1', type=str, help='Path to Qrel file. Corresponds to first Topic and Result file.')
    argp.add_argument('topic_1', type=str, help='Path to Topic file. Corresponds to Qrel and Result file.')
    argp.add_argument('result_1', type=str, help='Path to Result file. Corresponds with Topic and Qrel file.')
    argp.add_argument('-t','--topics', type=str, help='Path to n Topic files corresponding with Results.', nargs='+')
    argp.add_argument('-r', '--results', type=str, help='Path to Results files corresponding with Topics.', nargs='+')
    args = argp.parse_args()
    os.system('huggingface-cli login')

    answers = convert_answers(args.answers)
    qrel_1 = Qrels.from_file(args.qrel_1, kind='trec')
    topic_1 = convert_topics(args.topic_1)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.use_default_system_prompt = False
    tokenizer.pad_token = '[PAD]'
    tokenizer.padding_side = 'left'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        cache_dir=None
    )

    result_1 = Run.from_file(args.result_1, kind='trec')

    topic_1_reranked_zero_norole = result_rerank(prompt_1, topic_1, result_1, answers)
    write_result_dict(topic_1_reranked_zero_norole, 1, 1)

    # topic_2_reranked_fewshot_role =

def siftdown(prompt, query, doc_heap, start_index, end_index):
    root = start_index
    while True:
        left_child = 2 * root + 1
        right_child = 2 * root + 2
        swap = root
        if left_child < end_index and not compare(prompt, query, doc_heap[root], doc_heap[left_child]):
            swap = left_child
        if right_child < end_index and not compare(prompt, query, doc_heap[swap], doc_heap[right_child]):
            swap = right_child
        if swap == root:
            break
        doc_heap[root], doc_heap[swap] = doc_heap[swap], doc_heap[root]
        root = swap

def result_rerank(prompt, topic, result, answers):
    for qid, doc_ranking in result.items():
        query = topic[qid]
        doc_heap = list[doc_ranking.keys()]
        doc_count = len(doc_heap)

        # Build the heap with sift-down from the last non-leaf node down to the root
        for i in range((doc_count - 2) // 2, -1, -1):
            siftdown(prompt, query, doc_heap, i, doc_count)

        # Extract elements from the heap to get them in sorted order
        sorted_docs = []
        for end_index in range(doc_count - 1, 0, -1):
            # Swap the root (most relevant document) with the last item in the heap
            doc_heap[0], doc_heap[end_index] = doc_heap[end_index], doc_heap[0]
            # Append the most relevant document to the sorted list
            sorted_docs.append(doc_heap[end_index])
            # Restore the heap property for the reduced heap
            siftdown(doc_heap, 0, end_index)

        # Append the last remaining document to the sorted list
        sorted_docs.append(doc_heap[0])
        sorted_docs.reverse()  # Reverse to have descending order of relevance

        # Update result with the sorted documents based on relevance
        result[qid] = {doc: answers[doc] for doc in sorted_docs}

    return result


def compare(prompt_format, query, doc_1, doc_2):
    global tokenizer, device, llm_model
    swapped = False
    # Zhuang's method uses both doc_1 and doc_2 pairings. Instead, I intend to just do one or the other.
    if bool(random.getrandbits(1)):
        doc_1, doc_2, swapped = doc_2, doc_1, True

    prompt_formatted = prompt_format.format(query=query, doc_1=doc_1, doc_2=doc_2)
    chat_template = [{'role': 'user', 'content': prompt_formatted}]
    prompt_final = tokenizer.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)
    prompt_final += ' Document: '

    input_ids = tokenizer(prompt_final, return_tensors='pt').input_ids.to(device)
    # Zhuang's method uses 0.0 temperature. Instead, I used a similar value to what you use in your examples.
    # As you mentioned in lecture, putting the temperature to 0.0 does not actually make it deterministic.
    output_ids = llm_model.generate(input_ids,
                                    do_sample=False,
                                    temperature=0.6,
                                    top_p=None,
                                    max_new_tokens=1)

    output_response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    if swapped:
        if output_response == '1' or output_response == 'Document 1':
            return False
        else:
            return True
    else:
        if output_response == '1' or output_response == 'Document 1':
            return True
        else:
            return False

# llama_model = AutoModelForCasualLM.from_pretrained(model_name)
    # pipeline = transformers.pipeline(
    #     'text-generation',
    #     model=model_name,
    #     torch_dtype=torch.bfloat16,
    #     device=device
    # )

    # pipeline.model.generation_config.pad_token_id = pipeline.tokenizer.pad_token_id
    #
    # result = Run.from_file(args.results[0], kind='trec')
    #
    # system_message = make_system_prompt('You are a tour guide who can pick the most relevant answer out of two when given a traveling question.')
    # prompt_template = system_message + create_few_shot_example(topic_1['6766'], answers['6768'], answers['176214'])
    #
    # prompt = pipeline.tokenizer.apply_chat_template(prompt_template, tokenize=False, add_generation_prompt=True)
    # terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    # outputs = pipeline(prompt,
    #                    max_new_tokens=10000,
    #                    eos_token_id=terminators,
    #                    do_sample=True,
    #                    temperature=0.6, top_p=0.9,
    #                    pad_token_id=pipeline.tokenizer.eos_token_id)
    # result = outputs[0]['generated_text'][len(prompt):].strip()



    # for topic_index, topic_file in enumerate(args.topics):
    #     topics = convert_topics(topic_file)
    #     result = Run.from_file(args.results[topic_index], kind='trec')
    #     result_dict = result.to_dict()
    #
    #     prompt1_rerank_dict = {}
    #     prompt2_rerank_dict = {}
    #
    #     for topic_id, doc_ranking in result_dict.items():
    #         topic = topics[topic_id]
    #         prompt1_rerank_dict[topic_id] = prompt_rerank(prompt_1, topic, topic_id, doc_ranking)
    #         prompt2_rerank_dict[topic_id] = prompt_rerank(prompt_2, topic, topic_id, doc_ranking)
    #
    #     write_result_dict(prompt1_rerank_dict, topic_index, 1)
    #     write_result_dict(prompt2_rerank_dict, topic_index, 2)
    #
    # pass



def write_result_dict(rerank_dict, topic_index, prompt_type):
    with open(f'prompt{prompt_type}_{topic_index}.tsv', 'w', encoding='utf-8', newline='') as csvfile:
        for qid, doc_id_scores  in rerank_dict.items():
            for rank, (doc_id, score_tensor) in enumerate(doc_id_scores.items()):
                score = score_tensor.item()
                csvfile.write('\t'.join([qid, 'Q0', doc_id, str(rank), str(score), f'prompt{prompt_type}']) + '\n')
        csvfile.close()
    pass

def convert_answers(answers_file):
    answers_list = json.load(open(answers_file, 'r', encoding='utf-8'))
    mapped_answers = {}
    for answer in answers_list:
        Id = answer['Id']
        Text = clean_string(answer['Text'])
        mapped_answers[Id] = Text
    return mapped_answers

def convert_topics(topics_file):
    topics_list = json.load(open(topics_file, 'r', encoding='utf-8'))
    mapped_topics = {}
    for topic in topics_list:
        Id = topic['Id']
        Title = topic['Title']
        Body = topic['Body']
        Tags = topic['Tags']
        mapped_topics[Id] = clean_string(' '.join([Title, Body, Tags]))
    return mapped_topics

def clean_string(text_string):
    res_str = bs(text_string, "html.parser").get_text(separator=' ')
    res_str = re.sub(r'http(s)?://\S+', ' ', res_str)
    res_str = re.sub(r'[^\x00-\x7F]+', '', res_str)
    res_str = res_str.translate({ord(p): ' ' if p in r'\/.!?-_' else None for p in string.punctuation})
    res_str = ' '.join(res_str.split())
    return res_str

# def make_prompt(query, doc_1, doc_2):
#     return f'''Given a query {query}, which of the following two documents is more relevant to the query?
#     Document 1: {doc_1}; Document 2: {doc_2};
#     Output Document 1 or Document 2:'''
#
# # Randomize which document is the 'correct' one so there is not a bias for Document 1 over Document 2
# def create_few_shot_example(query, correct_doc, incorrect_doc):
#     if bool(random.getrandbits(1)):
#         doc_1, doc_2, answer = incorrect_doc, correct_doc, 'Document 2'
#     else:
#         doc_1, doc_2, answer = correct_doc, correct_doc, 'Document 1'
#     prompt = make_prompt(query, doc_1, doc_2)
#     return [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}]
#
# def make_system_prompt(string):
#     return [{'role': 'system', 'content': string}]
#
# def prompt_rerank(prompt, topic, topic_id, doc_ranking):
#     pass


if __name__ == '__main__':
    main()