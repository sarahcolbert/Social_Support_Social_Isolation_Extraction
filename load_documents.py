# -*- coding: utf-8 -*-
"""This code loads clinical notes with entity level annotation from BRAT annotated files. The corresponding
note/document level annotations are loaded from a csv file.
It also provides some counts of entities at document level, calculates the inter-annotator-agreement, and
 helps in evaluating the results."""


import os
import spacy
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from brat_parser import get_entities_relations_attributes_groups


class LoadDocuments:

    def __init__(self):
        """This function initializes the configuration and entity categories."""
        self.si_classes = ['social_isolation_loneliness', 'social_isolation_no_social_network',
                           'social_isolation_no_emotional_support', 'social_isolation_no_instrumental_support',
                           'social_isolation_general']
        self.su_classes = ['social_support_social_network', 'social_support_emotional_support',
                           'social_support_instrumental_support', 'social_support_general']

        self.entity_categories = ['social_support_social_network', 'social_support_emotional_support',
                                  'social_support_instrumental_support', 'social_support_general',
                                  'social_support_probable', 'social_isolation_loneliness',
                                  'social_isolation_no_social_network', 'social_isolation_no_emotional_support',
                                  'social_isolation_no_instrumental_support', 'social_isolation_general',
                                  'social_isolation_probable']
        # templates found in Mount Sinai (MS) Data
        self.template_regex = '|'.join([r'(social support\?)',
                                        r'(social isolation or lack of social support network\?)',
                                        r'(social isolation or lack of social service\?)',
                                        r'(social isolation\?)',
                                        r'(single\/divorced\/widowed)',
                                        r'(single \/ divorced \/ widowed\?)',
                                        r'(homeless\?)',
                                        r'(therapist\:)',
                                        r'(family educated about triggers)',
                                        r'(patient and family educated about recommendations including)',
                                        r'(increased social support\?)'])

    def repl(self, m):
        """Replace the multiple # from the text that are found mainly in MS data"""
        return '#' * len(m.group())

    def convert_doc_categories(self, doc_annotation):
        """
        This will change the previous annotation to new two category annotations.
        :param doc_annotation: (str) previous document annotation
        :return su_annotation (str): new social support annotation categories
        si_column (list): new social isolation annotation categories
        """
        if doc_annotation == 'support':
            return 1, 0
        elif doc_annotation == 'isolation':
            return 0, 1
        elif doc_annotation == 'both':
            return 1, 1
        elif doc_annotation == 'none':
            return 0, 0

    def get_annotations(self, file_path=None, doc_annotation_path=None, template_removal=0, doc_annotation_chage=1):
        """
        This will load the annotations and document categories from the files.
        :param file_path: (str) path to BRAT annotation files.
        :param doc_annotation_path: (str) path to document annotation with file name and annotation.
        .. note:: if these paths are not provided, then check the paths in configuration files.
        .. note:: template removal is only for MS.
        :return all_data (dict): dictionary of file names and annotation details.
        """
        all_data = {}
        doc_annotation = None
        if file_path is None and doc_annotation_path is None:
            print('Provide the path for data and annotation')
            import sys
            sys.exit(1)
            # file_path = self.conf.gold_ann_path
            # doc_annotation = pd.read_csv(self.conf.gold_ann_doc_path)
        else:
            if not os.path.exists(file_path):
                raise Exception('File path is not valid', file_path)
            if not os.path.exists(doc_annotation_path):
                raise Exception('Annotation file path is not valid', doc_annotation_path)
            doc_annotation = pd.read_csv(doc_annotation_path)
        for _, row in doc_annotation.iterrows():
            temp = {}
            file_name = row['file_name'].strip()
            su_category, si_category = None, None
            if doc_annotation_chage == 1:
                annotation = row['annotation'].strip().lower()
                #if annotation is np.nan:
                #    annotation = 'none'
                su_category, si_category = self.convert_doc_categories(annotation)
            else:
                su_category, si_category = row['su_category'], row['si_category']
            doc_file = file_path + file_name + '.txt'
            ann_file = file_path + file_name + '.ann'
            temp['text'] = ''.join(open(doc_file, 'r').readlines())
            # this is for template removal
            if template_removal == 1:
                import re
                temp['text'] = re.sub(self.template_regex, self.repl, temp['text'])
            entities, relations, attributes, groups = get_entities_relations_attributes_groups(ann_file)
            temp['entities'] = entities
            temp['attributes'] = attributes
            temp['su_category'] = su_category
            temp['si_category'] = si_category
            # print(entities, attributes, groups)
            all_data[file_name] = temp
        return all_data

    def doc_classification_rule(self, fine_grain_class_counts):
        """This function will get the document categories based on the rules.
        :note:: Based on the presence of si_class and su_class categories, the document tag will be provided.
        :param: fine_grain_class_counts (dict): dict of entities derived by the system.
        :return: str: document category.
        """
        si_class_count = 0
        su_class_count = 0
        for si_class in self.si_classes:
            if fine_grain_class_counts[si_class] >= 1:
                si_class_count += 1
        for su_class in self.su_classes:
            if fine_grain_class_counts[su_class] >= 1:
                su_class_count += 1
        if su_class_count > 0 and si_class_count > 0:
            return 1, 1
        elif su_class_count == 0 and si_class_count > 0:
            return 0, 1
        elif su_class_count > 0 and si_class_count == 0:
            return 1, 0
        else:
            return 0, 0

    def get_token_for_char(self, doc, char_s_idx):
        for i, token in enumerate(doc):
            if char_s_idx > token.idx:
                continue
            if char_s_idx == token.idx:
                return i
            if char_s_idx < token.idx:
                return i-1

    def convert_entity_to_sentence_category(self, all_data):
        """
        This converts entity annotations to sentence level annotation.
        For ex. a sentence in a file has one or more social_isolation_no_instrumental_support,
        then it will annotate that sentence with social_isolation_no_instrumental_support.

        :param: all_data (dict): It contains file name and corresponding entity and document annotations. Refer to
                get_annotations for more details.
        :return: categories_sent_data (pd.DataFrame): dataframe of the sentences with fine-grained categories.
        """
        nlp = spacy.load("en_core_web_sm")
        final_out = []
        for file_name in all_data:
            #print(all_data[file_name].keys())
            entities = all_data[file_name]['entities'].values()
            text_doc = nlp(all_data[file_name]['text'])
            su_category = all_data[file_name]['su_category']
            si_category = all_data[file_name]['si_category']
            #doc_annotation = all_data[file_name]['annotation']
            new_enitites = {}
            for entity in entities:
                start, end = entity.span[0]
                e_text = entity.text
                span = text_doc.char_span(start, end)
                if span is None:
                    start_t = self.get_token_for_char(text_doc, start)
                    e_length = len([token.text for token in nlp(e_text)])
                    span = text_doc[start_t:start_t+e_length]
                new_enitites[span.start] = [span, entity.type]
            if not len(new_enitites) == len(entities):
                print('errors')
            for sent_i, sent in enumerate(text_doc.sents):
                class_counts = {key: 0 for key in self.entity_categories}
                for i_d in new_enitites:
                    if sent.start <= i_d <= sent.end:
                        category = new_enitites[i_d][1]
                        class_counts[category] += 1
                for category in class_counts.keys():
                    if class_counts[category] > 1:
                        class_counts[category] = 1
                final_out.append([file_name, sent_i, sent.text] + list(class_counts.values()) +
                                 [su_category, si_category]) # + [doc_annotation])
        columns = ['file_name', 'sent_id', 'sent_text'] + self.entity_categories + ['su_category', 'si_category']
        categories_sent_data = pd.DataFrame(data=final_out, columns=columns)
        return categories_sent_data

    def convert_sentence_to_document_category(self, categories_sent_data, pred=True):
        """This function will get the document level categories from sentence level categories.
        :param categories_sent_data: dataframe of sentence level data
        :param pred: True for system predicted and False for gold standard data
        :return: dataframe of document level results
        """
        final_results = []
        for file_name, group in categories_sent_data.groupby('file_name'):
            class_counts = {key: 0 for key in self.entity_categories}
            for category in self.entity_categories:
                if group[category].sum() >= 1:
                    class_counts[category] = 1
            su_category, si_category = None, None
            if pred:
                su_category, si_category = self.doc_classification_rule(class_counts)
            else:
                su_category, si_category = group['su_category'].tolist()[0], group['si_category'].tolist()[0]
            final_results.append([file_name] + list(class_counts.values()) + [su_category, si_category])
        columns = ['file_name'] + self.entity_categories + ['su_category', 'si_category']
        return pd.DataFrame(data=final_results, columns=columns)

    def convert_entity_to_document_category(self, all_data):
        """
        This converts entity annotations to document level annotation.
        For ex. a file has one or more social_isolation_no_instrumental_support, then it will annotate that document
        with social_isolation_no_instrumental_support.

        :param: all_data (dict): It contains file name and corresponding entity and document annotations. Refer to
        get_annotations for more details.
        :return: categories_data (pd.DataFrame): dataframe of the document coarse and fine-grained categories.
        """
        categories_data = []
        for file_name in all_data:
            class_counts = {key: 0 for key in self.entity_categories}
            entities = all_data[file_name]['entities'].values()
            for entity in entities:
                class_counts[entity.type] += 1
            vals = [file_name] + list(class_counts.values()) + [all_data[file_name]['su_category'],
                                                                all_data[file_name]['si_category']]
            categories_data.append(vals)
        #TODO: Changes based on the
        column_names = ['file_name'] + self.entity_categories + ['su_category', 'si_category']
        categories_data = pd.DataFrame(data=categories_data, columns=column_names)
        #categories_data.to_csv('annotation_categories_before.csv')
        for category in self.entity_categories:
            categories_data[category] = np.where(categories_data[category] > 1, 1, categories_data[category])
        #print(categories_data.head())
        #categories_data.to_csv('annotation_categories.csv')
        return categories_data

    def calculate_kappa(self, ann_categories1, ann_categories2):
        """
        This function calculates the kappa inter annotator agreement for the annotations created by two users/systems.

        :param ann_categories1 (pd.DataFrame): DataFrame created from the annotation by user1/system1
        :param ann_categories2 (pd.DataFrame): DataFrame created from the annotation by user2/system2
        :return none
        """
        iaa_cat = []
        for category in self.entity_categories:
            iaa = cohen_kappa_score(ann_categories1[category], ann_categories2[category])
            iaa_cat.append(iaa)
            print('IAA for ', category, ' : ', iaa)
        print('Mean IAA for entity categories: ', np.mean(iaa_cat))
        iaa_cat = []
        for category in ['su_category', 'si_category']:
            iaa = cohen_kappa_score(ann_categories1[category], ann_categories2[category])
            iaa_cat.append(iaa)
            print('IAA for ', category, ' : ', iaa)
        print('IAA for document categories : ', np.mean(iaa_cat))

    def evaluation_matrix(self, category, ann_true, ann_pred):
        print('Analysis for ', category, ' :\n Accuracy = ', accuracy_score(ann_true, ann_pred))
        print('Classification report for ', category, ' :')
        print(classification_report(ann_true, ann_pred))
        print('Confusion matrix for ', category, ' :')
        print(confusion_matrix(ann_true, ann_pred))

    def calculate_iaa(self, ann_categories_true, ann_categories_pred):
        """
        This function calculates the inter annotator agreement for the annotations created by two users/systems.

        :param ann_categories1 (pd.DataFrame): DataFrame created from the annotation by user1/system1
        :param ann_categories2 (pd.DataFrame): DataFrame created from the annotation by user2/system2
        :return none
        """
        joined_cats_true = []
        joined_cats_pred = []

        print("Coarse grained annotations: ")
        for doc_category in ['su_category', 'si_category']:
            ann_true = ann_categories_true[doc_category].tolist()
            ann_pred = ann_categories_pred[doc_category].tolist()
            joined_cats_true.extend(ann_true)
            joined_cats_pred.extend(ann_pred)
            self.evaluation_matrix(doc_category, ann_true, ann_pred)
        print('Joined classification report: ', classification_report(joined_cats_true, joined_cats_pred))
        print('\n\n\n')

        joined_cats_true = []
        joined_cats_pred = []
        # print('Change evaluation after sometime.')
        # for entity_category in self.entity_categories: # this will remove probable category.
        for entity_category in self.su_classes + self.si_classes:
            # While evaluating LLM results we did not include the emotional support categories (both SI and SS)
            # TODO: comment out this while running rule-based algorithm.
            # if entity_category == 'social_isolation_no_emotional_support' \
            #         or entity_category == 'social_support_emotional_support':
            #     continue
            ann_true = ann_categories_true[entity_category].tolist()
            ann_pred = ann_categories_pred[entity_category].tolist()
            joined_cats_true.extend(ann_true)
            joined_cats_pred.extend(ann_pred)
            self.evaluation_matrix(entity_category, ann_true, ann_pred)
        print('Joined classification report: ', classification_report(joined_cats_true, joined_cats_pred))

    def get_entities_info(self, all_data):
        """This function provide details of entities text for verification purpose.
        :param: all_data (dict): It contains file name and corresponding entity and document annotations. Refer to
        get_annotations for more details.
        :return none
        """
        attribute_with_cat = {}
        for file_name in all_data:
            entities = all_data[file_name]['entities'].values()
            for entity in entities:
                if entity.type in attribute_with_cat:
                    ''''''
                else:
                    attribute_with_cat[entity.type] = [entity.text]
        max = 0
        for item in attribute_with_cat.values():
            if max < len(item):
                max = len(item)
        data = pd.DataFrame()
        for item in attribute_with_cat:
            texts = attribute_with_cat[item]
            if len(texts) < max:
                texts = texts + ([np.nan] * (max - len(texts)))
            data[item] = texts
        data.to_csv('annotation_text_with_category.csv')

    def get_entities(self, all_data):
        """
        :param: all_data (dict): It contains file name and corresponding entity and document annotations. Refer to
        get_annotations for more details.
        :return: entities_text (pd.DataFrame): dictionaries of entities in
        """
        entities_text = []
        for file_name in all_data:
            entities = all_data[file_name]['entities'].values()
            entity_text = []
            for entity in entities:
                entity_text.append(entity.text)
            entities_text.append([file_name, ', '.join(entity_text)])
        entities_text = pd.DataFrame(data=entities_text, columns=['file_name', 'entity_text'])
        return entities_text

    def get_counts(self, all_data):
        """
        This function counts the frequencies of entity and document categories for providing details in publication.

        :param: all_data (dict): It contains file name and corresponding entity and document annotations. Refer to
        get_annotations for more details.
        :rtype: none
        """
        entity_categories = []
        su_categories = []
        si_categories = []
        for file_name in all_data:
            # print(file_name)
            entities = all_data[file_name]['entities'].values()
            su_categories.append(all_data[file_name]['su_category'])
            si_categories.append(all_data[file_name]['si_category'])
            # print(entities)
            unique_entity = []
            for entity in entities:
                if entity.type in unique_entity:
                    continue
                else:
                    unique_entity.append(entity.type)
            entity_categories.extend(unique_entity)
        from collections import Counter
        print('Document categories SU: ')
        print(Counter(su_categories))
        print('Document categories SI: ')
        print(Counter(si_categories))
        print('Entity categories: ')
        print(Counter(entity_categories))


def main():
    load_docs = LoadDocuments()
    # Loading the data from BRAT annotation.
    annotated_data = load_docs.get_annotations(
        './Psych_notes/annotation_sisu_psych_notes_final/support_notes/',
        './Psych_notes/annotation_sisu_psych_notes_final/social_support_files.csv')
    annotated_data.update(load_docs.get_annotations(
        './Psych_notes/annotation_sisu_psych_notes_final/isolation_notes/',
        './Psych_notes/annotation_sisu_psych_notes_final/social_isolation_files.csv'))
    load_docs.get_counts(annotated_data)
    # For verifying entity categories.
    # ld.get_entities_info(annotated_data)

    # Evaluating IAA for two annotation.
    class_annotation_df = load_docs.convert_entity_to_document_category(annotated_data)
    all_data = load_docs.get_annotations('./Psych_notes/isolation_notes_mohit/',
                                  './Psych_notes/mohit_annotation.csv')
    load_docs.get_counts(all_data)
    class_annotation_df1 = load_docs.convert_entity_to_document_category(all_data)
    load_docs.get_entities_info(all_data)
    load_docs.calculate_kappa(class_annotation_df, class_annotation_df1)


if __name__ == '__main__':
    main()