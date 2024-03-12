from configuration_file import Configuration


class LoadLexicons:
    def __init__(self):
        # path to social isolation lexicons
        self.si_path = './lexicons/social_isolation/'
        # path to social support lexicons
        self.su_path = './lexicons/social_support/'
        # path to exclusion terms
        self.exclusion_terms_path = './lexicons/exclusion_terms.txt'

    def read_lexicon(self, file_path):
        """
        Reading lexicon list from a file.
        :param file_path (str): address to lexicon file.
        :return lexicons (list): list of words in lexicon file
        """
        lexicons = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or line == '':
                    continue
                lexicons.append(line.lower())
        lexicons = list(set(lexicons))
        return lexicons

    def lexicons(self):
        si_loneliness = self.read_lexicon(self.si_path + 'loneliness')
        si_no_social_network = self.read_lexicon(self.si_path + 'no_social_network')
        si_no_emotional_support = self.read_lexicon(self.si_path + 'no_emotional_support')
        si_no_instrumental_support = self.read_lexicon(self.si_path + 'no_instrumental_support')
        si_general = self.read_lexicon(self.si_path + 'general')
        si_probable = self.read_lexicon(self.si_path + 'probable')

        su_social_network = self.read_lexicon(self.su_path + 'social_network')
        su_emotional_support = self.read_lexicon(self.su_path + 'emotional_support')
        su_instrumental_support = self.read_lexicon(self.su_path + 'instrumental_support')
        su_general = self.read_lexicon(self.su_path + 'general')
        su_probable = self.read_lexicon(self.su_path + 'probable')
        return si_loneliness, si_no_social_network, si_no_emotional_support, si_no_instrumental_support,\
               si_general, si_probable,  su_social_network, su_emotional_support, su_instrumental_support,\
               su_general, su_probable

    def get_exclusion_terms(self):
        return self.read_lexicon(self.exclusion_terms_path)