#from nlp_functions.word_and_character_vectors import PAD_ID,UNK_ID
#from nlp_functions.sentence_operations import get_ids_and_vectors
import os
import random
import numpy as np
from nlp_functions.word_and_character_vectors import PAD_ID,UNK_ID,get_glove,get_char
from nlp_functions.sentence_operations import split_by_whitespace,tokens_to_word_ids,tokens_to_char_ids,pad_words,convert_ids_to_word_vectors,pad_characters,convert_ids_to_char_vectors

def padded(token_batch, batch_pad=0):
    """
    Inputs:
      token_batch: List (length batch size) of lists of ints.
      batch_pad: Int. Length to pad to. If 0, pad to maximum length sequence in token_batch.
    Returns:
      List (length batch_size) of padded of lists of ints.
        All are same length - batch_pad if batch_pad!=0, otherwise the maximum length in token_batch
    """
    if batch_pad==0:
        maxlen=max([len(token) for token in token_batch])
    else:
        maxlen=batch_pad
    retval=[]
    for token in token_batch:
        newtoken=token+[PAD_ID]*(maxlen-len(token))
        retval.append(newtoken)
    return retval

class Batch(object):
    def __init__(self,context_tokens, context_mask, context_words_glove, context_words_char, question_tokens,question_mask, question_words_glove, question_words_char, ans_span,uuids=None):
        self.context_tokens=context_tokens
        self.context_mask=context_mask
        self.context_words_glove=context_words_glove
        self.context_words_char=context_words_char
        self.question_tokens=question_tokens
        self.question_mask=question_mask
        self.question_words_glove=question_words_glove
        self.question_words_char=question_words_char
        self.ans_span=ans_span
        self.uuids=uuids
        self.batch_size = len(self.context_mask)

class SQuadDataObject(object):
    def __init__(self,glove_word2id,glove_id2word,glove_embed_matrix,word2vec_word2id,word2vec_id2word,word2vec_embed_matrix,fasttext_word2id,fasttext_id2word,fasttext_embed_matrix,char2id,id2char,char_embed_matrix,data_path,batch_size,context_length,question_length,context_word_length,question_word_length):
        self.glove_word2id=glove_word2id
        self.glove_id2word=glove_id2word
        self.glove_embed_matrix=glove_embed_matrix
        self.word2vec_word2id=word2vec_word2id
        self.word2vec_id2word=word2vec_id2word
        self.word2vec_embed_matrix=word2vec_embed_matrix
        self.fasttext_word2id=fasttext_word2id
        self.fasttext_id2word=fasttext_id2word
        self.fasttext_embed_matrix=fasttext_embed_matrix
        self.char2id=char2id
        self.id2char=id2char
        self.char_embed_matrix=char_embed_matrix
        self.train_data=[]
        self.dev_data=[]
        context_file,question_file,ans_file=open(os.path.join(data_path,'train.context'),'r',encoding='utf-8'),open(os.path.join(data_path,'train.question'),'r',encoding='utf-8'),open(os.path.join(data_path,'train.span'),'r',encoding='utf-8')
        context_line,question_line,ans_line=context_file.readline(),question_file.readline(),ans_file.readline()
        while context_line and question_line and ans_line:
            self.train_data.append((context_line,question_line,ans_line))
            context_line,question_line,ans_line=context_file.readline(),question_file.readline(),ans_file.readline()
        context_file,question_file,ans_file=open(os.path.join(data_path,'dev.context'),'r',encoding='utf-8'),open(os.path.join(data_path,'dev.question'),'r',encoding='utf-8'),open(os.path.join(data_path,'dev.span'),'r',encoding='utf-8')
        context_line,question_line,ans_line=context_file.readline(),question_file.readline(),ans_file.readline()
        while context_line and question_line and ans_line:
            self.dev_data.append((context_line,question_line,ans_line))
            context_line,question_line,ans_line=context_file.readline(),question_file.readline(),ans_file.readline()
        print("Size of (context, question, answer) triples in training set: ",len(self.train_data))
        print("Size of (context, question, answer) triples in dev set: ", len(self.dev_data))
        self.batch_size=batch_size
        self.context_length=context_length
        self.question_length=question_length
        self.context_word_length=context_word_length
        self.question_word_length=question_word_length

    def process_data(self,context,question,ans,discard_long):
        context_tokens = split_by_whitespace(context)
        question_tokens = split_by_whitespace(question)
        if len(context_tokens) > self.context_length:
            if discard_long:
                return None, None, None, None, None, None,None, None,None
            else:
                context_tokens = context_tokens[:self.context_length]
        if len(question_tokens) > self.question_length:
            if discard_long:
                return None, None, None, None, None, None,None, None,None
            else:
                question_tokens = question_tokens[:self.question_length]
        ans_line = [int(s) for s in ans.split()]
        assert len(ans_line) == 2
        if ans_line[1] < ans_line[0]:
            print("Found an ill-formed gold span: start=%i end=%i" % (ans_line[0], ans_line[1]))
            return None,None,None,None,None,None,None, None,None

        glove_ids_context=pad_words(tokens_to_word_ids(context_tokens,self.glove_word2id),self.context_length)
        glove_vectors_context=convert_ids_to_word_vectors(glove_ids_context,self.glove_embed_matrix)
        char_ids_context=pad_characters(tokens_to_char_ids(context_tokens,self.char2id),self.context_length,self.context_word_length)
        char_vectors_context=convert_ids_to_char_vectors(char_ids_context,self.char_embed_matrix)

        glove_ids_question=pad_words(tokens_to_word_ids(question_tokens,self.glove_word2id),self.question_length)
        glove_vectors_question=convert_ids_to_word_vectors(glove_ids_question,self.glove_embed_matrix)
        char_ids_question = pad_characters(tokens_to_char_ids(question_tokens, self.char2id), self.question_length,self.question_word_length)
        char_vectors_question = convert_ids_to_char_vectors(char_ids_question, self.char_embed_matrix)
        return context_tokens,glove_ids_context,glove_vectors_context,char_vectors_context,question_tokens,glove_ids_question,glove_vectors_question,char_vectors_question,ans_line

    def generate_one_training_epoch(self,discard_long=False):
        num_batches=int(len(self.train_data))//self.batch_size
        if self.batch_size*num_batches<len(self.train_data): num_batches+=1
        random.shuffle(self.train_data)

        for i in range(num_batches):
            context_tokens=[]
            context_words_for_mask=[]
            context_words_glove=[]
            context_words_char=[]
            question_tokens=[]
            question_words_for_mask=[]
            question_words_glove=[]
            question_words_char=[]
            ans_span=[]

            for (context,question,ans) in self.train_data[i*self.batch_size:(i+1)*self.batch_size]:
                context_token, glove_ids_context, glove_vectors_context, char_vectors_context, question_token, glove_ids_question, glove_vectors_question, char_vectors_question, ans_line=self.process_data(context,question,ans,discard_long)
                if glove_ids_context is not None:
                    context_words_for_mask.append(glove_ids_context)
                    context_words_glove.append(glove_vectors_context)
                    context_words_char.append(char_vectors_context)
                    question_words_for_mask.append(glove_ids_question)
                    question_words_glove.append(glove_vectors_question)
                    question_words_char.append(char_vectors_question)
                    ans_span.append(ans_line)
            context_tokens.append(context_token)
            context_words_for_mask=np.array(context_words_for_mask)
            context_words_glove=np.array(context_words_glove)
            context_words_char=np.array(context_words_char)
            question_tokens.append(question_token)
            question_words_for_mask=np.array(question_words_for_mask)
            question_words_glove=np.array(question_words_glove)
            question_words_char=np.array(question_words_char)
            ans_span=np.array(ans_span)
            context_mask=(context_words_for_mask!=PAD_ID).astype(np.int32)
            question_mask=(question_words_for_mask!=PAD_ID).astype(np.int32)
            batch = Batch(context_tokens,context_mask, context_words_glove, context_words_char,question_tokens, question_mask, question_words_glove,question_words_char, ans_span)
            yield batch

    def generate_one_dev_epoch(self,discard_long=False):
        num_batches=int(len(self.dev_data))//self.batch_size
        if self.batch_size*num_batches<len(self.dev_data): num_batches+=1
        random.shuffle(self.dev_data)

        for i in range(num_batches):
            context_tokens = []
            context_words_for_mask = []
            context_words_glove = []
            context_words_char = []
            question_tokens = []
            question_words_for_mask = []
            question_words_glove = []
            question_words_char = []
            ans_span = []

            for (context,question,ans) in self.dev_data[i*self.batch_size:(i+1)*self.batch_size]:
                context_token, glove_ids_context, glove_vectors_context, char_vectors_context, question_token, glove_ids_question, glove_vectors_question, char_vectors_question, ans_line=self.process_data(context,question,ans,discard_long)
                if glove_ids_context is not None:
                    context_words_for_mask.append(glove_ids_context)
                    context_words_glove.append(glove_vectors_context)
                    context_words_char.append(char_vectors_context)
                    question_words_for_mask.append(glove_ids_question)
                    question_words_glove.append(glove_vectors_question)
                    question_words_char.append(char_vectors_question)
                    ans_span.append(ans_line)
            context_tokens.append(context_token)
            context_words_for_mask = np.array(context_words_for_mask)
            context_words_glove = np.array(context_words_glove)
            context_words_char = np.array(context_words_char)
            question_tokens.append(question_token)
            question_words_for_mask = np.array(question_words_for_mask)
            question_words_glove = np.array(question_words_glove)
            question_words_char = np.array(question_words_char)
            ans_span = np.array(ans_span)
            context_mask = (context_words_for_mask != PAD_ID).astype(np.int32)
            question_mask = (question_words_for_mask != PAD_ID).astype(np.int32)
            batch = Batch(context_tokens, context_mask, context_words_glove, context_words_char, question_tokens,question_mask, question_words_glove, question_words_char, ans_span)
            yield batch


"""glove_embed_matrix,glove_word2id,glove_id2word=get_glove('../../ml_data_files/')
word2vec_embed_matrix,word2vec_word2id,word2vec_id2word=None,None,None
fasttext_embed_matrix,fasttext_word2id,fasttext_id2word=None,None,None
char_embed_matrix,char2id,id2char=get_char('../../ml_data_files/',128,5)

data_path='../data/'
batch_size=500
context_length=300
question_length=30
context_word_length=37
question_word_length=30
discard_long=False
zz=SQuadDataObject(glove_word2id,glove_id2word,glove_embed_matrix,word2vec_word2id,word2vec_id2word,word2vec_embed_matrix,fasttext_word2id,fasttext_id2word,fasttext_embed_matrix,char2id,id2char,char_embed_matrix,data_path,batch_size,context_length,question_length,context_word_length,question_word_length)
for batch in zz.generate_one_training_epoch(discard_long=True):
    print(batch.context_mask.shape)
    print(batch.context_words_glove.shape)
    print(batch.context_words_char.shape)
    print(batch.question_mask.shape)
    print(batch.question_words_glove.shape)
    print(batch.question_words_char.shape)
    print(batch.ans_span.shape)
    print("---------------------")
"""