import pandas as pd
import spacy
import re

def tag_cnt(languages:list, num_file:int, Z:int=0):
    if Z == 0:
        list_tags = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'F1', 'F2', 'F3', 'F4', 'G1', 'G2', 'G3', 'H1', 'H2', 'H3', 'H4', 'H5', 'I1', 'I2', 'I3', 'I4', 'K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'O1', 'O2', 'O3', 'O4', 'P1', 'Q1', 'Q2', 'Q3', 'Q4', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'T1', 'T2', 'T3', 'T4', 'W1', 'W2', 'W3', 'W4', 'W5', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'Y1', 'Y2']

    elif Z == 1:
        list_tags = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'F1', 'F2', 'F3', 'F4', 'G1', 'G2', 'G3', 'H1', 'H2', 'H3', 'H4', 'H5', 'I1', 'I2', 'I3', 'I4', 'K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'O1', 'O2', 'O3', 'O4', 'P1', 'Q1', 'Q2', 'Q3', 'Q4', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'T1', 'T2', 'T3', 'T4', 'W1', 'W2', 'W3', 'W4', 'W5', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'Y1', 'Y2', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'Z99']

    df_tagcount = pd.DataFrame([[0 for m in range(len(languages)*num_file)
                               ] for n in range(len(list_tags))
                              ], index=list_tags, columns=[
                                  f'{language[:2]}_{n}' for language in languages for n in range(1, num_file+1)
                                                        ]
                             )


    for language in languages:
        if language == 'English':
            # We exclude the following components as we do not need them. ``
            nlp = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])
            # Load the PyMUSAS rule based tagger in a seperate spaCy pipeline
            tagger_pipeline = spacy.load('en_dual_none_contextual')
            # Adds the PyMUSAS rule based tagger to the main spaCy pipeline
            nlp.add_pipe('pymusas_rule_based_tagger', source=tagger_pipeline)

        elif language == 'Chinese':
            # We exclude the following components as we do not need them. ``
            nlp = spacy.load('zh_core_web_sm', exclude=['parser', 'ner'])
            # Load the PyMUSAS rule based tagger in a seperate spaCy pipeline
            tagger_pipeline = spacy.load('cmn_dual_upos2usas_contextual')
            # Adds the PyMUSAS rule based tagger to the main spaCy pipeline
            nlp.add_pipe('pymusas_rule_based_tagger', source=tagger_pipeline)

        elif language == 'Dutch':
            # We exclude the following components as we do not need them. ``
            nlp = spacy.load('nl_core_news_sm', exclude=['parser', 'ner'])
            # Load the PyMUSAS rule based tagger in a seperate spaCy pipeline
            tagger_pipeline = spacy.load('nl_single_upos2usas_contextual')
            # Adds the PyMUSAS rule based tagger to the main spaCy pipeline
            nlp.add_pipe('pymusas_rule_based_tagger', source=tagger_pipeline)

        elif language == 'French':
            # We exclude the following components as we do not need them. ``
            nlp = spacy.load('fr_core_news_sm', exclude=['parser', 'ner'])
            # Load the PyMUSAS rule based tagger in a seperate spaCy pipeline
            tagger_pipeline = spacy.load('fr_single_upos2usas_contextual')
            # Adds the PyMUSAS rule based tagger to the main spaCy pipeline
            nlp.add_pipe('pymusas_rule_based_tagger', source=tagger_pipeline)

        elif language == 'Finnish':
            # We exclude the following components as we do not need them. ``
            nlp = spacy.load('fi_core_news_sm', exclude=['parser', 'ner'])
            # Load the PyMUSAS rule based tagger in a seperate spaCy pipeline
            tagger_pipeline = spacy.load('fi_single_upos2usas_contextual')
            # Adds the PyMUSAS rule based tagger to the main spaCy pipeline
            nlp.add_pipe('pymusas_rule_based_tagger', source=tagger_pipeline)

        elif language == 'Italian':
            # We exclude the following components as we do not need them. ``
            nlp = spacy.load('it_core_news_sm', exclude=['parser', 'ner'])
            # Load the PyMUSAS rule based tagger in a seperate spaCy pipeline
            tagger_pipeline = spacy.load('it_dual_upos2usas_contextual')
            # Adds the PyMUSAS rule based tagger to the main spaCy pipeline
            nlp.add_pipe('pymusas_rule_based_tagger', source=tagger_pipeline)

        elif language == 'Portuguese':
            # We exclude the following components as we do not need them. ``
            nlp = spacy.load('pt_core_news_sm', exclude=['parser', 'ner'])
            # Load the PyMUSAS rule based tagger in a seperate spaCy pipeline
            tagger_pipeline = spacy.load('pt_dual_upos2usas_contextual')
            # Adds the PyMUSAS rule based tagger to the main spaCy pipeline
            nlp.add_pipe('pymusas_rule_based_tagger', source=tagger_pipeline)

        elif language == 'Spanish':
            # We exclude the following components as we do not need them. ``
            nlp = spacy.load('es_core_news_sm', exclude=['parser', 'ner'])
            # Load the PyMUSAS rule based tagger in a seperate spaCy pipeline
            tagger_pipeline = spacy.load('es_dual_upos2usas_contextual')
            # Adds the PyMUSAS rule based tagger to the main spaCy pipeline
            nlp.add_pipe('pymusas_rule_based_tagger', source=tagger_pipeline)



        # dictionary for counting freqencies of each tag
        dict_semtags_zero = {'A1':0, 'A2':0, 'A3':0, 'A4':0, 'A5':0, 'A6':0, 'A7':0,
                            'A8':0, 'A9':0, 'A10':0, 'A11':0, 'A12':0, 'A13':0, 'A14':0, 'A15':0,
                            'B1':0, 'B2':0, 'B3':0, 'B4':0, 'B5':0,
                            'C1':0,
                            'E1':0, 'E2':0, 'E3':0, 'E4':0, 'E5':0, 'E6':0, 'A7':0,
                            'F1':0, 'F2':0, 'F3':0, 'F4':0,
                            'G1':0, 'G2':0, 'G3':0,
                            'H1':0, 'H2':0, 'H3':0, 'H4':0, 'H5':0,
                            'I1':0, 'I2':0, 'I3':0, 'I4':0,
                            'K1':0, 'K2':0, 'K3':0, 'K4':0, 'K5':0, 'K6':0,
                            'L1':0, 'L2':0, 'L3':0,
                            'M1':0, 'M2':0, 'M3':0, 'M4':0, 'M5':0, 'M6':0, 'M7':0, 'M8':0,
                            'N1':0, 'N2':0, 'N3':0, 'N4':0, 'N5':0, 'N6':0,
                            'O1':0, 'O2':0, 'O3':0, 'O4':0,
                            'P1':0,
                            'Q1':0, 'Q2':0, 'Q3':0, 'Q4':0,
                            'S1':0, 'S2':0, 'S3':0, 'S4':0, 'S5':0, 'S6':0, 'S7':0, 'S8':0, 'S9':0,
                            'T1':0, 'T2':0, 'T3':0, 'T4':0,
                            'W1':0, 'W2':0, 'W3':0, 'W4':0, 'W5':0,
                            'X1':0, 'X2':0, 'X3':0, 'X4':0, 'X5':0, 'X6':0, 'X7':0, 'X8':0, 'X9':0,
                            'Y1':0, 'Y2':0,
                            'Z1':0, 'Z2':0, 'Z3':0, 'Z4':0, 'Z5':0, 'Z6':0, 'Z7':0, 'Z8':0, 'Z9':0, 'Z99':0,
                            }

        for idx_file in range(1,num_file+1):
            file = open(f'doc/{language[:2]}_texts/{language[:2]}_{idx_file}.txt').read() # file path needs to be 'current_dir/{lang_id}_texts/{lang_id}_{index}.txt'
            output_doc = nlp(file) # tag on the text of the file

            dict_semtags_cnt = dict_semtags_zero # set the tag-frequency counter
            for token in output_doc:
                if len(token._.pymusas_tags) != 0:
                    tag = re.match(r'[A-Z][0-9]+|[a-zA-Z0-9.]+', token._.pymusas_tags[0]).group() # take the first tag

                    if tag in list_tags:
                        dict_semtags_cnt[tag] += 1 # increment the counter
        
            for tag in list_tags:
                df_tagcount[f'{language[:2]}_{idx_file}'][tag] = dict_semtags_cnt[tag]

    return df_tagcount