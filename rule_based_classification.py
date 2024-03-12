# -*- coding: utf-8 -*-
"""This will classify the notes in to fine and coarse grain categories of social support and isolation using
the available lexicons.
1. This will extract the entities from notes
2. Based on the entities, this program will provide the fine and coarse grained categories.
"""

import re
import spacy
from spacy.matcher import PhraseMatcher
import numpy as np
import pandas as pd
from lexicons.rdl import LoadLexicons
from load_documents import LoadDocuments
from configuration_file import Configuration


class RuleBasedClassification:
    def __init__(self):
        """"This function will load configuration files, spacy, lexicons
        and create matchers for each lexicon category """
        self.conf = Configuration()
        # loading spacy
        self.nlp = spacy.load("en_core_web_sm")
        self.load_docs = LoadDocuments()
        # loading all the lexicons
        self.si_loneliness, self.si_no_social_network, self.si_no_emotional_support, self.si_no_instrumental_support,\
            self.si_general, self.si_probable, self.su_social_network, self.su_emotional_support, \
            self.su_instrumental_support, self.su_general, self.su_probable = LoadLexicons().lexicons()
        self.exclusion_terms = LoadLexicons().get_exclusion_terms()
        #print(len(self.si_loneliness) + len(self.si_no_social_network) + len(self.si_no_emotional_support) +
        #      len(self.si_no_instrumental_support) + len(self.si_general) + len(self.si_probable) +
        #      len(self.su_social_network) + len(self.su_emotional_support) + len(self.su_instrumental_support) +
        #      len(self.su_general) + len(self.su_probable))
        # print(len(self.exclusion_terms))

        # converting all lexicons into matcher
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.create_matchers()

    def create_matchers(self):
        """This function create matcher for each lexicon categories.
        return all lexicon matchers
        """
        self.create_matcher(self.si_loneliness, 'social_isolation_loneliness')
        self.create_matcher(self.si_no_social_network, 'social_isolation_no_social_network')
        self.create_matcher(self.si_no_emotional_support, 'social_isolation_no_emotional_support')
        self.create_matcher(self.si_no_instrumental_support, 'social_isolation_no_instrumental_support')
        self.create_matcher(self.si_general, 'social_isolation_general')
        self.create_matcher(self.si_probable, 'social_isolation_probable')
        self.create_matcher(self.su_social_network, 'social_support_social_network')
        self.create_matcher(self.su_emotional_support, 'social_support_emotional_support')
        self.create_matcher(self.su_instrumental_support, 'social_support_instrumental_support')
        self.create_matcher(self.su_general, 'social_support_general')
        self.create_matcher(self.su_probable, 'social_support_probable')
        self.create_matcher(self.exclusion_terms, 'exclusion')

    def create_matcher(self, lexicon_list, category):
        """This creates a lexicon matchers using spacy PhaseMatcher.
        :param: lexicon_list (list): list of lexicons
        :param: category (str): name of the lexicon category
        :return: matcher(spacy.function): matcher created by spacy
        """
        for item in lexicon_list:
            self.matcher.add(category, [self.nlp(item)])

    def extract_entities(self, text_doc):
        """This function identifies the lexicons using the spacy matchers.
        :param: text_doc (str): text in note
        :return: entities (list): list of identified entities.
        :return: entity_texts (str): joined entity texts
        """
        text_doc = re.sub(' +', ' ', text_doc)
        text_doc = self.nlp(text_doc)
        entities = []
        new_spans = []
        matches = self.matcher(text_doc)
        spans = [text_doc[start:end] for _, start, end in matches]
        for span in spacy.util.filter_spans(spans):
            #print(span.start, span.end, span.text)
            new_spans.append(span)
        entity_texts = []
        for match_id, start, end in matches:
            string_id = self.nlp.vocab.strings[match_id]  # Get string representation
            span = text_doc[start:end]  # The matched span
            if span not in new_spans:
                continue
            # print(char_index(text, span.text, start, end))
            start_char = text_doc[start].idx
            end_char = start_char + len(span.text)
            # print(start_char, end_char)
            # print(doc[start].idx)
            # print(doc[end-1].idx)
            temp_entity = {'type': string_id, 'start': start_char, 'end': end_char, 'text': span.text}
            entity_texts.append(span.text)
            entities.append(temp_entity)
            # print(match_id, string_id, start, end, span.text)
        # TODO: negation identification.
        entity_texts = ', '.join(entity_texts)
        return entities, entity_texts

    def save_derived_entities(self, entities, file_name):
        """This function saves the identified entities in BRAT format.
        :parameter entities: (list) list of entities identified by system
        :parameter file_name: (str) file path where the BRAT formatted output will be saved.
        """
        text = ''
        entity_id = 'T'
        id_count = 1
        for entity in entities:
            text += entity_id + str(id_count) + '\t' + entity['type'] + ' ' + str(entity['start']) + ' ' + \
                    str(entity['end']) + '\t' + entity['text'] + '\n'
            id_count += 1
        with open(file_name, 'w') as f:
            f.write(text)

    def format_derived_entities(self, entities):
        """This function will get the entity counts for a document.
        :param entities: (list) list of entities identified for each note.
        :return list of all entity categories counts.
        """
        class_counts = {key: 0 for key in self.load_docs.entity_categories}
        for entity in entities:
            entity_type = entity['type']
            if entity_type == 'exclusion':
                continue
            class_counts[entity_type] += 1
        return list(class_counts.values())

    def doc_classification_rule(self, entities):
        """This function will get the document categories based on the rules.
        :note:: Based on the presence of si_class and su_class categories, the document tag will be provided.
        :param: entities (list): list of entities derived by the system.
        :return: str: document category.
        """
        si_class_count = 0
        su_class_count = 0
        for entity in entities:
            category = entity['type']
            if category in self.load_docs.si_classes:
                si_class_count += 1
            elif category in self.load_docs.su_classes:
                su_class_count += 1
            else:
                pass
                #print(category)
        if su_class_count > 0 and si_class_count > 0:
            return [1, 1]
        elif su_class_count == 0 and si_class_count > 0:
            return [0, 1]
        elif su_class_count > 0 and si_class_count == 0:
            return [1, 0]
        else:
            return [0, 0]

    def process(self):
        """This is the main function which controls the entity identification, providing fine grain categories, and
        identifying coarse grain document categories.
        """
        # loading entity and document annotations here.
        #isolation_notes = self.load_docs.get_annotations('./Psych_notes/isolation_notes_braja/',
        #                                                 './Psych_notes/braja_veer_annotation_isolation.csv')
        #print('loaded isolation notes')
        #support_notes = self.load_docs.get_annotations('./Psych_notes/support_notes_veer/',
        #                                               './Psych_notes/braja_veer_annotation_support.csv')
        #annotated_data = support_notes
        # annotated_data = isolation_notes.update(support_notes)
        # annotated_data = ld.get_annotations('./Psych_notes/support_notes_veer/',
        #                                    './Psych_notes/braja_veer_annotation_support.csv')
        #print(type(annotated_data))
        #print(annotated_data.keys())
        #annotated_data = ld.get_annotations('./Psych_notes/support_notes_veer/',
        #                                    './Psych_notes/braja_veer_annotation_support.csv')
        #annotated_data = self.load_docs.get_annotations('/Users/brajapatra/PycharmProjects/SDoH/PICORI/SI_notes/braja_notes 2/',
        #                                                '/Users/brajapatra/PycharmProjects/SDoH/PICORI/SI_notes/doc_annotation.csv')

        annotated_data = self.load_docs.get_annotations(
            './Psych_notes/annotation_sisu_psych_notes_final/support_notes/',
            './Psych_notes/annotation_sisu_psych_notes_final/social_support_files.csv')
        annotated_data.update(self.load_docs.get_annotations(
            './Psych_notes/annotation_sisu_psych_notes_final/isolation_notes/',
            './Psych_notes/annotation_sisu_psych_notes_final/social_isolation_files.csv'))
        annotation_categories_df = self.load_docs.convert_entity_to_document_category(annotated_data)
        result_categories_df = []
        e_texts = []
        for file_name in annotated_data:
            # print('Processing: ', file_name)
            derived_entities, e_text = self.extract_entities(annotated_data[file_name]['text'])
            e_texts.append(e_text)
            # self.save_derived_entities(derived_entities, './Psych_notes/temp_results/'+file_name+'ann')
            entity_results = self.format_derived_entities(derived_entities)
            document_categories = self.doc_classification_rule(derived_entities)
            result_categories_df.append([file_name] + entity_results + document_categories)
        columns = ['file_name'] + self.load_docs.entity_categories + ['su_category', 'si_category']
        result_categories_df = pd.DataFrame(data=result_categories_df, columns=columns)
        for category in self.load_docs.entity_categories:
            result_categories_df[category] = np.where(result_categories_df[category] > 1, 1,
                                                      result_categories_df[category])
        entity_texts = self.load_docs.get_entities(annotated_data)
        entity_texts['derived_entity_text'] = e_texts
        self.load_docs.calculate_iaa(annotation_categories_df, result_categories_df)
        #annotation_categories_df.join(result_categories_df, entity_texts, 'inner').drop(result_categories_df.file_name)
        merged_results = pd.concat([annotation_categories_df, result_categories_df, entity_texts], axis=1)
        merged_results.to_csv('annotation_vs_system_output.csv')


def main():
    rbc = RuleBasedClassification()
    rbc.process()


if __name__ == '__main__':
    main()