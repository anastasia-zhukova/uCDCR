import xml.etree.ElementTree as ET
import os
import json
import string
import copy
import re
import pandas as pd
import numpy as np
from nltk import Tree
import spacy
import sys
import shortuuid
from tqdm import tqdm
from setup import *
from utils import *
from logger import LOGGER
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


NP4E_PARSING_FOLDER = os.path.join(os.getcwd())
OUT_PATH = os.path.join(NP4E_PARSING_FOLDER, OUTPUT_FOLDER_NAME)

nlp = spacy.load('en_core_web_sm')


source_path = os.path.join(NP4E_PARSING_FOLDER, NP4E_FOLDER_NAME)
result_path = os.path.join(OUT_PATH, 'test_parsing')
out_path = os.path.join(OUT_PATH)

subtopic_fullname_dict = {
    "bukavu": "Bukavu_bombing",
    "peru": "Peru_hostages",
    "tajikistan": "Tajikistan_hostages",
    "israel": "Israel_suicide_bomb",
    "china": "China-Taiwan_hijack"
}


def conv_files():
    entity_mentions = []
    event_mentions = []
    conll_df = pd.DataFrame(columns=[TOPIC_SUBTOPIC_DOC, DOC_ID, SENT_ID, TOKEN_ID, TOKEN, REFERENCE])
    topic_name = "0_bomb_explosion_kidnap"
    preprocessed_df = pd.DataFrame()
    topic_sent_dict = {}
    annotated_counter = 0
    parsed_counter = 0

    for subtopic_id, subtopic in enumerate(os.listdir(source_path)):
        entity_mentions_local = []

        LOGGER.info(f"Parsing of NP4E topic {subtopic}...")
        subtopic_name_composite_full = f'{subtopic_id}_{subtopic_fullname_dict[subtopic]}'
        subtopic_name_composite = f'{subtopic_id}_{subtopic}'
        topic_folder = os.path.join(source_path, subtopic)

        docs_folder = os.path.join(topic_folder, "Basedata")
        coref_pre_dict = {}
        coref_pre_df = pd.DataFrame()

        for doc_text_name in os.listdir(docs_folder):
            doc_id = re.sub(r'\D+', "", str(doc_text_name))
            topic_subtopic_doc = f'{topic_name.split("_")[0]}/{subtopic_name_composite}/{doc_id}'

            if doc_text_name.split(".")[-1] != "xml":
                continue

            doc_text_file_path = os.path.join(docs_folder, doc_text_name)
            tree = ET.parse(doc_text_file_path)
            root = tree.getroot()

            # Step 1: collect original tokenized text
            for t_id, elem in enumerate(root):
                conll_df = pd.concat([conll_df, pd.DataFrame({
                    TOPIC_SUBTOPIC_DOC: topic_subtopic_doc,
                    DOC_ID: doc_id,
                    SENT_ID: 0,
                    TOKEN_ID: t_id,
                    TOKEN: elem.text,
                    REFERENCE: "-"
                }, index=[f'{doc_id}/{elem.attrib["id"]}'])], axis=0)

            sentence_markables = ET.parse(os.path.join(topic_folder, "markables", doc_text_name.split("_")[0]+ "_sentence_level.xml"))
            root = sentence_markables.getroot()
            for sent_id, elem in enumerate(root):
                # span="word_243..word_284"
                try:
                    word_start, word_end = elem.attrib["span"].split("..")
                except ValueError:
                    word_start, word_end = elem.attrib["span"], elem.attrib["span"]
                conll_df.loc[f'{doc_id}/{word_start}': f'{doc_id}/{word_end}', SENT_ID] = sent_id
                local_df = conll_df.loc[f'{doc_id}/{word_start}': f'{doc_id}/{word_end}']
                conll_df.loc[f'{doc_id}/{word_start}': f'{doc_id}/{word_end}', TOKEN_ID] = list(range(len(local_df)))

            # Step 4: reparse the documents
            doc_df_orig = conll_df[conll_df[DOC_ID] == doc_id]
            token_id_global = 0
            for sent_id, sent_df in doc_df_orig.groupby(SENT_ID):
                sent_tokens = sent_df[TOKEN].tolist()
                sentence_str = ""
                for t in sent_tokens:
                    sentence_str, _, _ = append_text(sentence_str, t)

                sentence_str = sentence_str.replace("_", "").replace("$-", "").replace("$ -", "")
                sentence_str_cor = correct_whitespaces(sentence_str)
                if all([t in sentence_str_cor for t in sent_tokens]):
                    sentence_str = sentence_str_cor

                doc = nlp(sentence_str)
                tokens_id_global = list(range(token_id_global, token_id_global + len(doc)))
                preprocessed_df = pd.concat([preprocessed_df, pd.DataFrame({
                    TOPIC_SUBTOPIC_DOC: topic_subtopic_doc,
                    DOC_ID: doc_id,
                    SENT_ID: int(sent_id),
                    TOKEN_ID: list(range(len(doc))),
                    TOKEN: [t.text for t in doc],
                    TOKEN_ID_GLOBAL: tokens_id_global,
                    CHAR_ID_START: [t.idx for t in doc],
                    WHITESPACE_AFTER: [bool(len(t.whitespace_)) for t in doc],
                    REFERENCE: "-"
                }, index=[f'{topic_subtopic_doc}_{i}' for i in tokens_id_global])])

                if topic_subtopic_doc not in topic_sent_dict:
                    topic_sent_dict[topic_subtopic_doc] = {}
                topic_sent_dict[topic_subtopic_doc][sent_id] = doc
                token_id_global += len(doc)

            # Step 5: match the originally tokenized mentions to the reparsed texts to collect mention attributes, e.g., POS, NER, etc
            markables = ET.parse(os.path.join(topic_folder, "markables", doc_text_name.split("_")[0]+"_coref_level.xml"))
            root = markables.getroot()

            for markable in root:
                marker_id = markable.get("id")
                span_str = markable.get("span")
                coref_id = markable.get("coref_set")

                if span_str == "p":
                    continue

                try:
                    word_start, word_end = span_str.split("..")
                except ValueError:
                    word_start, word_end = span_str, span_str

                markable_df = conll_df.loc[f'{doc_id}/{word_start}': f'{doc_id}/{word_end}']
                if not len(markable_df):
                    continue

                annotated_counter += 1

                # mention attributes
                init_token_numbers = list(markable_df[TOKEN_ID].values)
                tokens = {}
                token_str = ""
                tokens_text = list(markable_df[TOKEN].values)
                for token in tokens_text:
                    token_str, word_fixed, no_whitespace = append_text(token_str, token)

                token_str = token_str.replace("_", "").replace("$-", "")
                sent_id = int(list(markable_df[SENT_ID].values)[0])

                # determine the sentences as a string
                doc = topic_sent_dict[topic_subtopic_doc][sent_id]
                sent_str = "".join([t.text + t.whitespace_ for t in doc])
                sent_df = preprocessed_df[
                    (preprocessed_df[SENT_ID] == sent_id) & (preprocessed_df[TOPIC_SUBTOPIC_DOC] == topic_subtopic_doc)]
                sent_tokens_text = [t.text for t in doc]
                start_token_id, _ = find_sub_list(tokens_text, sent_tokens_text)

                if start_token_id == -1:
                    try_again = 2
                    best_matching_start_token = []

                    while try_again > 0:
                        if try_again == 1:
                            token_str = correct_whitespaces(correct_whitespaces(token_str))

                        try:
                            positions = [(match.start(), match.end()) for match in
                                         re.finditer(fr'{token_str}', sent_str)]
                        except:
                            positions = []

                        if not len(positions):
                            if token_str in sent_str:
                                start_pos = sent_str.index(token_str)
                                end_pos = start_pos + len(token_str)
                                positions = [(start_pos, end_pos)]
                            elif token_str.replace(" ", "") in sent_str:
                                token_str = token_str.replace(" ", "")
                                start_pos = sent_str.index(token_str)
                                end_pos = start_pos + len(token_str)
                                positions = [(start_pos, end_pos)]

                        for pos in positions:
                            selected_token_df = sent_df[
                                (sent_df[CHAR_ID_START] >= pos[0]) & (sent_df[CHAR_ID_START] < pos[1])]
                            if not len(selected_token_df):
                                continue
                            best_matching_start_token.append(selected_token_df[TOKEN_ID].values.tolist())

                        if not len(best_matching_start_token):
                            try_again -= 1
                        else:
                            try_again = 0

                    if not len(best_matching_start_token):
                        LOGGER.warning(f"\'{token_str}\' not found in \'{sent_str}\'")
                        continue

                    best_matching_start_token_id = int(
                        np.argmin([abs(p[0] - init_token_numbers[0]) for p in best_matching_start_token]))
                    start_token_id = best_matching_start_token[best_matching_start_token_id][0]

                    tokens_text = [t.text for t in doc[start_token_id: start_token_id + len(
                        best_matching_start_token[best_matching_start_token_id])]]

                    token_ids = list(range(start_token_id, start_token_id + len(
                        best_matching_start_token[best_matching_start_token_id])))
                else:
                    token_ids = list(range(start_token_id, start_token_id + len(tokens_text)))

                head_tokens_text = [t.head.text for t in doc[start_token_id: start_token_id + len(tokens_text)]]

                if len(tokens_text) == 1:
                    mention_head = tokens_text[0]
                else:
                    if head_tokens_text[0] in tokens_text:
                        mention_head = head_tokens_text[0]
                    else:
                        mention_head_cand = set(tokens_text).intersection(set(head_tokens_text))
                        if len(mention_head_cand):
                            mention_head = list(mention_head_cand)[0]
                        else:
                            mention_head = tokens_text[0]

                mention_head_id = token_ids[tokens_text.index(mention_head)]

                token_mention_start_id = sent_df[sent_df[TOKEN_ID] == start_token_id][TOKEN_ID_GLOBAL][0]
                doc_df = preprocessed_df[preprocessed_df[TOPIC_SUBTOPIC_DOC] == topic_subtopic_doc]

                if token_mention_start_id - CONTEXT_RANGE < 0:
                    context_min_id = 0
                    tokens_number_context = list(sent_df[sent_df[TOKEN_ID].isin(token_ids)][TOKEN_ID_GLOBAL])
                    head_id_context = list(sent_df[sent_df[TOKEN_ID] == mention_head_id][TOKEN_ID_GLOBAL])[0]
                    context_max_id = min(token_mention_start_id + CONTEXT_RANGE, len(doc_df))
                    # round it to the full sentence
                    context_sent_id = doc_df[doc_df[TOKEN_ID_GLOBAL] == context_max_id - 1][SENT_ID][0]
                    context_max_id = doc_df[doc_df[SENT_ID] == context_sent_id][TOKEN_ID_GLOBAL][-1]
                else:
                    context_min_id = token_mention_start_id - CONTEXT_RANGE
                    # round it to the full sentence
                    context_sent_id = doc_df[doc_df[TOKEN_ID_GLOBAL] == context_min_id][SENT_ID][0]
                    context_min_id = doc_df[doc_df[SENT_ID] == context_sent_id][TOKEN_ID_GLOBAL][0]

                    global_token_ids = list(sent_df[sent_df[TOKEN_ID].isin(token_ids)][TOKEN_ID_GLOBAL])
                    tokens_number_context = [int(t - context_min_id) for t in global_token_ids]
                    head_id_context = list(sent_df[sent_df[TOKEN_ID] == mention_head_id][TOKEN_ID_GLOBAL])[
                                          0] - context_min_id
                    context_max_id = min(token_mention_start_id + CONTEXT_RANGE, len(doc_df))
                    # round it to the full sentence
                    context_sent_id = doc_df[doc_df[TOKEN_ID_GLOBAL] == context_max_id - 1][SENT_ID][0]
                    context_max_id = doc_df[doc_df[SENT_ID] == context_sent_id][TOKEN_ID_GLOBAL][-1]

                mention_context_text = list(doc_df.iloc[context_min_id:context_max_id + 1][TOKEN].values)
                mention_context_id = list(doc_df.iloc[context_min_id:context_max_id + 1][TOKEN_ID_GLOBAL].values)
                context_sent_id_start = mention_context_id.index(sent_df[TOKEN_ID_GLOBAL].values.tolist()[0])
                mention_ner = doc[mention_head_id].ent_type_ if doc[mention_head_id].ent_type_ != "" else "O"
                mention_id = f"NP4E_{shortuuid.uuid(marker_id + token_str + str(doc_id) + str(sent_id))}"

                # Step 6: form a mention with all attributes
                mention = {
                    COREF_CHAIN: None,
                           MENTION_NER: mention_ner,
                           MENTION_HEAD_POS: doc[mention_head_id].pos_,
                           MENTION_HEAD_LEMMA: doc[mention_head_id].lemma_,
                           MENTION_HEAD: mention_head,
                           MENTION_HEAD_ID: int(mention_head_id),
                           DOC_ID: doc_id,
                           DOC: doc_id,
                           IS_SINGLETON: len(tokens) == 1,
                           MENTION_ID: mention_id,
                           MENTION_TYPE: None,
                           MENTION_FULL_TYPE: None,
                           SENT_ID: sent_id,
                           MENTION_CONTEXT: mention_context_text,
                           TOKENS_NUMBER_CONTEXT: tokens_number_context,
                           TOKENS_NUMBER: token_ids,
                           CONTEXT_START_END_GLOBAL_ID: [int(context_min_id), int(context_max_id)],
                           MENTION_SENTENCE_CONTEXT_START_END_ID: [context_sent_id_start, context_sent_id_start + len(sent_df)],
                           MENTION_HEAD_ID_CONTEXT: int(head_id_context),
                           TOKENS_STR: token_str,
                           TOKENS_TEXT: tokens_text,
                           TOPIC_ID: topic_subtopic_doc.split("/")[0],
                           TOPIC: topic_name,
                           SUBTOPIC_ID: topic_subtopic_doc.split("/")[1],
                           SUBTOPIC: subtopic_name_composite_full,
                           COREF_TYPE: IDENTITY,
                           DESCRIPTION: None,
                           CONLL_DOC_KEY: topic_subtopic_doc,
                           SPLIT: "tbd"
                           }
                mention = reorganize_field_order(mention)
                parsed_counter += 1
                coref_pre_dict[f'{doc_id}/{coref_id}/{mention_id}'] = mention
                coref_pre_df = pd.concat([coref_pre_df, pd.DataFrame({
                    COREF_CHAIN: coref_id,
                    DOC_ID: doc_id,
                    MENTION_ID: mention_id,
                    TOKENS_STR: token_str,
                    MENTION_HEAD: mention_head,
                    MENTION_HEAD_POS: doc[mention_head_id].pos_,
                    MENTION_NER: mention_ner
                }, index=[mention_id])], axis=0)

        grouped_dfs = coref_pre_df.groupby([COREF_CHAIN, DOC_ID])
        cand_chains_df = pd.DataFrame(np.zeros((len(grouped_dfs), len(grouped_dfs))),
                                      index=[f'{doc_id_1}/{coref_chain_orig_id_1}'
                                             for (coref_chain_orig_id_1, doc_id_1), group_df_1 in grouped_dfs],
                                      columns=[f'{doc_id_1}/{coref_chain_orig_id_1}'
                                               for (coref_chain_orig_id_1, doc_id_1), group_df_1 in grouped_dfs])

        LOGGER.info("Building matrix of the cross-document chains overlap...")
        for i, ((coref_chain_orig_id_1, doc_id_1), group_df_1) in tqdm(list(enumerate(grouped_dfs))):
            # no pronouns
            cand_df_1 = group_df_1[group_df_1[MENTION_HEAD_POS] != "PRON"]
            # check full mentions and their heads
            mentions_n_heads_1 = set(cand_df_1[TOKENS_STR].values).union(set(cand_df_1[MENTION_HEAD].values))

            for j, ((coref_chain_orig_id_2, doc_id_2), group_df_2) in enumerate(grouped_dfs):
                cand_df_2 = group_df_2[group_df_2[MENTION_HEAD_POS] != "PRON"]
                if j <= i:
                    continue

                # check full mentions and their heads
                mentions_n_heads_2 = set(cand_df_2[TOKENS_STR].values).union(set(cand_df_2[MENTION_HEAD].values))
                overlap = mentions_n_heads_1.intersection(mentions_n_heads_2)
                overlap_size = len(overlap)

                if len(overlap) == 1:
                    # handle a special case when there is only one overlap and it is a proper noun: give a bit bigger
                    # size to pass the upcoming check
                    if list(overlap)[0][0].isupper():
                        overlap_size = 1.5
                cand_chains_df.loc[f'{doc_id_1}/{coref_chain_orig_id_1}', f'{doc_id_2}/{coref_chain_orig_id_2}'] = overlap_size
                cand_chains_df.loc[ f'{doc_id_2}/{coref_chain_orig_id_2}', f'{doc_id_1}/{coref_chain_orig_id_1}',] = overlap_size

        # Step 3: collect coreference chains, i..e, information about the chain ids to later decide which mentions are singletons
        chain_dict = {}
        checked_sets = set()
        for col in cand_chains_df.columns:
            if col in checked_sets:
                continue

            to_check = {col}
            # first: candidates that are similar to the seed chain
            to_check = to_check.union(set(cand_chains_df[cand_chains_df[col] > 1].index))
            # second: candidates that are similar to the similar chains to the seed chain
            to_check = to_check.union(set(cand_chains_df.loc[list(to_check)][cand_chains_df.loc[list(to_check)][col] > 1].index))

            sim_mentions_df = pd.DataFrame()
            for cand_key in to_check:
                if cand_key in checked_sets:
                    continue

                doc_id, coref_chain = cand_key.split("/")
                sim_mentions_df = pd.concat([sim_mentions_df,
                                             coref_pre_df[(coref_pre_df[DOC_ID] == doc_id) & (coref_pre_df[COREF_CHAIN] == coref_chain)]], axis=0)
            # ignore chains that consist only of pronouns
            non_pron_df = sim_mentions_df[sim_mentions_df[MENTION_HEAD_POS] != "PRON"]
            if not len(non_pron_df):
                continue

            mention_type_df = non_pron_df.groupby(MENTION_NER).count()
            if len(mention_type_df) == 1:
                mention_type = mention_type_df[COREF_CHAIN].idxmax()
            else:
                mention_type_df = mention_type_df.sort_values(COREF_CHAIN, ascending=False)
                if mention_type_df.index[0] != "O":
                    mention_type = mention_type_df.index[0]
                else:
                    # next best
                    mention_type = mention_type_df.index[1]

            mention_type = mention_type if mention_type != "O" else "OTHER"
            description = non_pron_df.groupby(TOKENS_STR).count()[COREF_CHAIN].idxmax()
            chain_id = f'{mention_type[:3]}{shortuuid.uuid()}'
            for mention_index, mention_row in sim_mentions_df.iterrows():
                mention = coref_pre_dict[f'{mention_row[DOC_ID]}/{mention_row[COREF_CHAIN]}/{mention_index}']
                mention[COREF_CHAIN] = chain_id
                if chain_id not in chain_dict:
                    chain_dict[chain_id] = []
                chain_dict[chain_id].append(mention)

                # np4e only has entities
                entity_mentions_local.append(mention)
            checked_sets = checked_sets.union(to_check)

        entity_mentions.extend(entity_mentions_local)

    entity_mentions_unique = {}
    for m in entity_mentions:
        entity_mentions_unique[m[MENTION_ID]] = m

    # assign chain properties
    df_mentions = pd.DataFrame()
    for mention in tqdm(entity_mentions_unique.values()):
        df_mentions = pd.concat([df_mentions, pd.DataFrame({
            attr: str(value) if type(value) == list else value for attr, value in mention.items()
        }, index=[mention[MENTION_ID]])], axis=0)

    for chain_name, chain_local_df in tqdm(df_mentions.groupby(COREF_CHAIN),desc="Assigning chain properties"):
        non_pron_df = chain_local_df[chain_local_df[MENTION_HEAD_POS] != "PRON"]
        if len(non_pron_df):
            mention_type_df = non_pron_df.groupby(MENTION_NER).count()
            if len(mention_type_df) == 1:
                mention_type = mention_type_df[COREF_CHAIN].idxmax()
            else:
                mention_type_df = mention_type_df.sort_values(COREF_CHAIN, ascending=False)
                if mention_type_df.index[0] != "O":
                    mention_type = mention_type_df.index[0]
                else:
                    # next best
                    mention_type = mention_type_df.index[1]
            mention_type = mention_type if mention_type != "O" else "OTHER"
            description = non_pron_df.groupby(TOKENS_STR).count()[COREF_CHAIN].idxmax()
        else:
            mention_type = "OTHER"
            description = chain_local_df.groupby(TOKENS_STR).count()[COREF_CHAIN].idxmax()

        for m_id in chain_local_df[MENTION_ID].values:
            entity_mentions_unique[m_id][IS_SINGLETON] = len(chain_local_df) == 1
            entity_mentions_unique[m_id][MENTION_TYPE] = mention_type[:3]
            entity_mentions_unique[m_id][MENTION_FULL_TYPE] = mention_type
            entity_mentions_unique[m_id][DESCRIPTION] = description

    LOGGER.info(
        f'The annotated mentions ({annotated_counter}) and parsed mentions ({len(entity_mentions_unique)}).')

    LOGGER.info("Splitting the dataset into train/val/test subsets...")

    with open("train_val_test_split.json", "r") as file:
        train_val_test_dict = json.load(file)
    conll_df_labels = pd.DataFrame()
    all_mentions = []

    LOGGER.info(f'Splitting NP4E into train/dev/test subsets...')
    for subset, subtopic_ids in train_val_test_dict.items():
        LOGGER.info(f'Creating data for {subset} subset...')

        split_folder = os.path.join(OUT_PATH, subset)
        if subset not in os.listdir(OUT_PATH):
            os.mkdir(split_folder)

        selected_entity_mentions = []
        for mention in entity_mentions_unique.values():
            if any([subtopic_id in mention[SUBTOPIC_ID] for subtopic_id in subtopic_ids]):
                mention[SPLIT] = subset
                selected_entity_mentions.append(mention)

        all_mentions.extend(selected_entity_mentions)

        with open(os.path.join(split_folder, MENTIONS_ENTITIES_JSON), "w", encoding='utf-8') as file:
            json.dump(selected_entity_mentions, file)

        with open(os.path.join(split_folder, MENTIONS_EVENTS_JSON), "w", encoding='utf-8') as file:
            json.dump([], file)

        conll_df_split = pd.DataFrame()
        for t_id in subtopic_ids:
            conll_df_split = pd.concat([conll_df_split,
                                        preprocessed_df[preprocessed_df[TOPIC_SUBTOPIC_DOC].str.contains(f'0/{t_id}')]],
                                       axis=0)
        conll_df_split[SPLIT] = subset
        conll_df_split_labels = make_save_conll(conll_df_split, selected_entity_mentions,
                                                split_folder, return_df_only=True)
        conll_df_labels = pd.concat([conll_df_labels, conll_df_split_labels])

    df_all_mentions = pd.DataFrame()
    for mention in tqdm(all_mentions):
        df_all_mentions = pd.concat([df_all_mentions, pd.DataFrame({
            attr: str(value) if type(value) == list else value for attr, value in mention.items()
        }, index=[mention[MENTION_ID]])], axis=0)

    df_all_mentions.to_parquet(os.path.join(OUT_PATH, MENTIONS_ALL_PARQUET), engine="pyarrow")
    conll_df_labels.to_parquet(os.path.join(OUT_PATH, DOCUMENTS_ALL_PARQUET), engine="pyarrow")

    LOGGER.info(f'Done! \nNumber of unique mentions: {len(df_all_mentions)} '
                f'\nNumber of unique chains: {len(set(df_all_mentions[COREF_CHAIN].values))} ')
    LOGGER.info(
        f'The annotated mentions ({annotated_counter}) and parsed mentions ({len(entity_mentions_unique)}).')
    LOGGER.info(f'Parsing of NP4E done!')


if __name__ == '__main__':
    conv_files()
