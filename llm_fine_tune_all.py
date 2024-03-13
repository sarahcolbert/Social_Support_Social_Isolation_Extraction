# -*- coding: utf-8 -*-
"""This code fine-tune all the models at a time."""


import gc
import pandas as pd
from glob import glob
from datasets import load_dataset
from transformers import Seq2SeqTrainer
from transformers.adapters import IA3Config
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments
from transformers import T5Tokenizer, T5ForConditionalGeneration


class LLMFineTuning:
    def __init__(self):
        self.saving_address = None
        self.epochs = None

    def read_tuning_data(self, fine_tuning_folder):
        """
        :param fine_tuning_folder:
        :return:
        """
        temp = []
        for file_name in glob(fine_tuning_folder):
            print(file_name)
            temp.append(pd.read_csv(file_name))
        temp = pd.concat(temp)
        print(temp.columns)
        print('Total number of tuning instances = ', temp.shape)
        temp.to_csv('temp.csv', index=False)

    def training(self, category):
        """
        This code decides the saving address and loads the examples for fine-tuning.
        This also specifies number of epochs for each fine-tuned categories.
        Finally, it fine-tune the model.
        :param category: (str) fine-grained category name
        """
        if category == 'social_isolation_loneliness':
            self.saving_address = r"tuned_model/si_l/flan-t5-xl"
            self.read_tuning_data('./fine_tuning/SI_L.csv')
            self.epochs = 15
        elif category == 'social_isolation_no_social_network':
            self.saving_address = r"tuned_model/si_no_sn/flan-t5-xl"
            self.read_tuning_data('./fine_tuning/SI_no_SN.csv')
            self.epochs = 15
        elif category == 'social_isolation_no_instrumental_support':
            self.saving_address = r"tuned_model/si_no_is/flan-t5-xl"
            self.read_tuning_data('./fine_tuning/SI_no_IS.csv')
            self.epochs = 20
        elif category == 'social_isolation_no_emotional_support':
            self.saving_address = r"tuned_model/si_no_es/flan-t5-xl"
            self.read_tuning_data('./fine_tuning/SI_no_ES.csv')
            self.epochs = 15
        elif category == 'social_isolation_general':
            self.saving_address = r"tuned_model/si_g/flan-t5-xl"
            self.read_tuning_data('./fine_tuning/SI_G.csv')
            self.epochs = 20
        elif category == 'social_support_social_network':
            self.saving_address = r"tuned_model/su_sn/flan-t5-xl"
            self.read_tuning_data('./fine_tuning/SU_SN.csv')
            self.epochs = 15
        elif category == 'social_support_instrumental_support':
            self.saving_address = r"tuned_model/su_is/flan-t5-xl"
            self.read_tuning_data('./fine_tuning/SU_IS.csv')
            self.epochs = 20
        elif category == 'social_support_general':
            self.saving_address = r"tuned_model/su_g/flan-t5-xl"
            self.read_tuning_data('./fine_tuning/SU_G.csv')
            self.epochs = 20
        elif category == 'social_support_emotional_support':
            self.saving_address = r"tuned_model/su_es/flan-t5-xl"
            self.read_tuning_data('./fine_tuning/SU_ES.csv')
            self.epochs = 15
        else:
            print('Category is something else: ', category)
            import sys
            sys.exit(1)

        def add_template(examples):
            i = 0
            for ins, con, qns, chs in zip(examples['Instruction'], examples['Context'], examples['Question'],
                                          examples['Choices']):
                #print(ins, con, qns, chs)
                try:
                    instruction = 'Instruction: ' + ins
                    context = 'Context: The Clinician wrote: "' + con + '"'
                    question = 'Question: ' + qns
                    choices = 'Choices: ' + chs
                    answer = 'Answer: '
                    template = instruction + '\n' + context + '\n' + question + '\n' + choices + '\n' + answer + '\n\n'
                    examples['Context'][i] = template
                    i += 1
                except:
                    #print(ins, con, qns, chs)
                    print("There were some errors")
            return examples

        dataset = load_dataset('csv', data_files={"train": "temp.csv"})
        dataset = dataset.map(add_template, batched=True)
        split_datasets = dataset["train"].train_test_split(train_size=0.9, seed=1)
        split_datasets["validation"] = split_datasets.pop("test")
        tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-xl', cache_dir='google/flan-t5-tokenizer')
        model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-xl')
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("Total Parameters: ", pytorch_total_params)

        # Tuning Model using Parameter Efficient Tuning Method - IA3
        config = IA3Config()
        model.add_adapter("ia3_adapter", config=config)
        model.train_adapter("ia3_adapter")
        model.train()

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total Parameters: ", pytorch_total_params)

        def preprocess_function(examples):
            inputs = [ex for ex in examples["Context"]]
            targets = [ex for ex in examples["Answer"]]
            # for x in range(len(inputs)):
            #     print("Current training string is ", inputs[x], '\n', targets[x], '\n\n', )
            # print(inputs)
            model_inputs = tokenizer(
                inputs, text_target=targets, max_length=512, truncation=True
            )
            return model_inputs

        tokenized_datasets = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        batch = data_collator([tokenized_datasets["train"][i] for i in range(1, 5)])
        batch.keys()

        # print(batch["labels"])

        epochs = self.epochs
        args = Seq2SeqTrainingArguments(
            f"google/flan-t5-xl",
            learning_rate=3e-3,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            weight_decay=0.05,
            num_train_epochs=epochs,
            predict_with_generate=True,
            greater_is_better=False,
            logging_strategy="epoch",
            no_cuda=True,
            warmup_steps=(len(split_datasets["train"]) * epochs) / 10,
            max_steps=len(split_datasets["train"]) * epochs
        )

        trainer = Seq2SeqTrainer(
            model,
            args,
            train_dataset=tokenized_datasets["train"],
            data_collator=data_collator,
            tokenizer=tokenizer
        )

        # save the model here
        trainer.train()
        trainer.save_model(self.saving_address)
        import shutil
        shutil.rmtree('google/flan-t5-xl')

    def process(self):
        # removed emotional support categories
        from load_documents import LoadDocuments
        load_docs = LoadDocuments()
        for category in load_docs.si_classes + load_docs.su_classes:
            if category == 'social_isolation_no_emotional_support' or \
                    category == 'social_support_emotional_support':
                continue
            self.training(category)
            gc.collect()


def main():
    LLMFineTuning().process()


if __name__ == '__main__':
    main()
