import pandas as pd
from nltk import sent_tokenize


class LLM_Utility:

    def __init__(self):
        self.questions = self.read_questions()
        self.instruction = 'Instruction: Read what the Clinician wrote about the patient in the Context and ' \
                           'answer the Question by choosing from the provided Choices.'
        self.token_length = 350
        self.min_token_length = 4

    @staticmethod
    def read_questions():
        temp = pd.read_csv('Questions.csv')
        return dict(zip(temp['Category'], temp['Question']))

    def format_query(self, sent, category):
        context = 'Context: The Clinician wrote: "' + sent + '"'
        question = 'Question: In the Clinician\'s opinion, "' + self.questions[category].lower() + '"'
        choices = 'Choices: yes; no; not relevant'
        answer = 'Answer: '
        templated_query = self.instruction + '\n' + context + '\n' + question + '\n' + choices + '\n' + answer
        return templated_query

    def divide_sentences(self, tokens, t_length):
        sents = []
        try:
            for i in range(0, t_length, self.token_length):
                if i + self.token_length < t_length:
                    sents.append(' '.join(tokens[i:i + self.token_length]))
                else:
                    sents.append(' '.join(tokens[i:]))
        except:
            print('Error in dividing tokens ', tokens)
        return sents

    def get_answer(self, template, tokenizer, model):
        input_ids = tokenizer(template, return_tensors="pt").input_ids
        outputs = model.generate(input_ids)
        predicted_ans = tokenizer.decode(outputs[0]).strip()
        # print(predicted_ans)
        result = None
        if 'yes' in predicted_ans:
            result = 1
        elif 'not relevant' in predicted_ans:
            result = 0
        elif 'no' in predicted_ans:
            result = 0
        else:
            print('Error in predicted ans ', predicted_ans)
        return result

    def preprocess_sentences(self, sent):
        ''' Spliting sentences using nltk and double space.
        '''
        sents = []
        sent_text = sent_tokenize(sent)  # this gives us a list of sentences
        # now loop over each sentence and tokenize it separately
        for sentence in sent_text:
            for sent in sentence.split('  '):
                sent = sent.strip()
                if sent == '':
                    continue
                sents.append(sent)
        return sents

    def find_category(self, list_sent, category, tokenizer, model):
        answers = []
        for long_sent in list_sent:
            sents = []
            for long_sent in self.preprocess_sentences(long_sent):
                try:
                    long_sent = long_sent.strip()
                    tokens = long_sent.split()
                    t_length = len(tokens)
                    if t_length < self.min_token_length:
                        continue
                    if t_length > self.token_length:
                        sents.extend(self.divide_sentences(tokens, t_length))
                    else:
                        sents.append(long_sent)
                except:
                    print('Error in this sentence ', long_sent)
                    continue
            if len(sents) == 0:
                answers.append(0)
                continue
            results = []
            for sent in sents:
                templated_query = self.format_query(sent, category)
                result = self.get_answer(templated_query, tokenizer, model)
                results.append(result)
            if sum(results) > 0:
                answers.append(1)
            else:
                answers.append(0)
        return answers
