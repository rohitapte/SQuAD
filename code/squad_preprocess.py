import os
import json
from tqdm import tqdm
from nltk import word_tokenize
import numpy as np
from sentence_operations import split_by_whitespace

def total_exs(dataset):
    """
    Returns the total number of (context, question, answer) triples,
    given the data read from the SQuAD json file.
    """
    total = 0
    for article in dataset['data']:
        for para in article['paragraphs']:
            total += len(para['qas'])
    return total

def tokenize_sequence(sequence):
    tokens=[token.replace("``", '"').replace("''", '"').lower() for token in word_tokenize(sequence)]
    #tokens= [token.replace("``", '"').replace("''", '"').lower() for token in split_by_whitespace(sequence)]
    return tokens

def get_char_word_loc_mapping(context, context_tokens):
    mapping={}
    token_index=0
    word_crawl=''
    for char_id,char in enumerate(context):
        if char not in [' ','\n']:
            context_token=context_tokens[token_index]
            word_crawl+=char
            if word_crawl==context_token:
                start_pos=char_id-len(word_crawl)+1
                for i in range(start_pos,char_id+1):
                    mapping[i]=(word_crawl,token_index)
                token_index+=1
                word_crawl=''
    if token_index!=len(context_tokens):
        for char in word_crawl:
            print(char,ord(char))
        return None
    else:
        return mapping

def extract_json_to_files(input_dir,output_dir):
    """takes training and dev jsons from SQuAD website. and extracts into 4 files
        paragraph
        question
        answer
        answer span
    the file contains the structure
                    json_data
                    /     \
            title(str)    data(list of dicts)
                         /       \
                    title   paragraph
                            /       \
                        context     qas (list)
                                  /  |  --------\
                                id  question    answers (list of dict)
                                                /      \
                                        answer_start    text
    """
    files={}
    files['train']='train-v1.1.json'
    files['dev']='dev-v1.1.json'

    for file in files:
        filename=os.path.join(input_dir,files[file])
        with open(filename,'r',encoding='utf-8') as data_file:
            examples = []
            dataset=json.load(data_file)
            count_total=total_exs(dataset)
            count_mapping_problem=0
            count_token_problem=0
            count_ansspan_problem=0
            count_examples=0
            for article_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(file)):
                article_paragraph=dataset['data'][article_id]['paragraphs']
                for paragraph_id in range(len(article_paragraph)):
                    context=article_paragraph[paragraph_id]['context']
                    context=context.replace("''",'"').replace("``",'"')
                    context = context.replace('\u3000', ' ').replace('\u202f',' ').replace('\u2009', ' ')#.replace("'","'")
                    context=context.replace('\-',' ')
                    context_tokens=tokenize_sequence(context)
                    context=context.lower()
                    qas=article_paragraph[paragraph_id]['qas']
                    charloc2wordloc=get_char_word_loc_mapping(context, context_tokens)
                    if charloc2wordloc is None:
                        count_mapping_problem+=len(qas)
                        continue
                    for qa in qas:
                        question=qa['question'].lower()
                        question_tokens=tokenize_sequence(question)

                        ans_text=qa['answers'][0]['text'].lower()
                        ans_text=ans_text.replace('\u3000', ' ').replace('\u202f', ' ').replace('\u2009', ' ')
                        ans_start_loc=qa['answers'][0]['answer_start']
                        if qa['id'] in ['5706baed2eaba6190074aca5','57269c73708984140094cbb5','57269c73708984140094cbb7','572a11661d04691400779721','572a11661d04691400779722','572a11661d04691400779723','572a11661d04691400779724','572a11661d04691400779725','572a2cfc1d0469140077981b','572a3a453f37b319004787e9','572a84d3f75d5e190021fb3c']:
                            ans_start_loc+=1
                        if qa['id'] in ['572a5df77a1753140016aedf','572a5df77a1753140016aee0','572a84d3f75d5e190021fb38','572a84d3f75d5e190021fb39','572a84d3f75d5e190021fb3a','572a84d3f75d5e190021fb3b','572a85df111d821400f38bad','572a85df111d821400f38bae','572a85df111d821400f38baf','572a85df111d821400f38bb0']:
                            ans_start_loc+=2
                        if qa['id'] in ['572a5df77a1753140016aee1','572a5df77a1753140016aee2']:
                            ans_start_loc+=3
                        if qa['id'] in ['57286bf84b864d19001649d6','57286bf84b864d19001649d5']:
                            ans_start_loc-=1
                        if qa['id'] in ['5726bee5f1498d1400e8e9f3','5726bee5f1498d1400e8e9f4']:
                            ans_start_loc-=2
                        ans_end_loc=ans_start_loc+len(ans_text)

                        if context[ans_start_loc:ans_end_loc]!=ans_text:
                            count_ansspan_problem+=1
                            continue
                        ans_start_wordloc = charloc2wordloc[ans_start_loc][1]  # answer start word loc
                        ans_end_wordloc = charloc2wordloc[ans_end_loc-1][1]  # answer end word loc
                        assert ans_start_wordloc <= ans_end_wordloc

                        ans_tokens = context_tokens[ans_start_wordloc:ans_end_wordloc + 1]
                        if "".join(ans_tokens) != "".join(ans_text.split()):
                            count_token_problem += 1
                            #print(ans_text)
                            #print(ans_tokens)
                            continue  # skip this question/answer pair
                        examples.append((' '.join(context_tokens),' '.join(question_tokens),' '.join(ans_tokens),' '.join([str(ans_start_wordloc),str(ans_end_wordloc)])))
            print("Number of (context, question, answer) triples discarded due to char -> token mapping problems: ", count_mapping_problem)
            print("Number of (context, question, answer) triples discarded because character-based answer span is unaligned with tokenization: ",count_token_problem)
            print("Number of (context, question, answer) triples discarded due character span alignment problems (usually Unicode problems): ",count_ansspan_problem)
            print("Processed %i examples of total %i\n" % (len(examples), len(examples)+count_mapping_problem+count_token_problem+count_ansspan_problem))
        indices = list(range(len(examples)))
        np.random.shuffle(indices)
        with open(os.path.join(output_dir,file+'.context'),'w',encoding='utf-8') as context_file, \
             open(os.path.join(output_dir,file+'.question'),'w',encoding='utf-8') as question_file, \
             open(os.path.join(output_dir,file+'.answer'),'w',encoding='utf-8') as answer_file, \
             open(os.path.join(output_dir,file+'.span'),'w',encoding='utf-8') as span_file:
            for i in indices:
                (context,question,answer,span)=examples[i]
                context_file.write(context+'\n')
                question_file.write(question+'\n')
                answer_file.write(answer+'\n')
                span_file.write(span+'\n')

extract_json_to_files('../data/','../data/')