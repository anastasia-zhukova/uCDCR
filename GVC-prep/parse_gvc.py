import io
import xml.etree.ElementTree as ET
import os
import json
import sys
import string
import spacy
import copy
import re
import pandas as pd
import numpy as np
sys.path.insert(0, '..')
from utils import *
import shortuuid
from nltk import Tree
from tqdm import tqdm
import warnings
from setup import *
from logger import LOGGER
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

GVC_PARSING_FOLDER = os.path.join(os.getcwd())
OUT_PATH = os.path.join(GVC_PARSING_FOLDER, OUTPUT_FOLDER_NAME)
source_path = os.path.join(GVC_PARSING_FOLDER, GVC_FOLDER_NAME)

nlp = spacy.load('en_core_web_sm')


def conv_files():
    topic_name = "0_gun_violence"
    topic_id = "0"
    train_dev_test_split_dict = {}
    conll_df = pd.DataFrame()
    need_manual_review_mention_head = {}
    event_mentions = []
    topic_sent_dict = {}
    preprocessed_df = pd.DataFrame()

    for file_name in ["dev", "test", "train"]:
        df = pd.read_csv(os.path.join(source_path, f'{file_name}.csv'), header=None)
        train_dev_test_split_dict[file_name] = list(df[1])

    subtopic_structure_dict = {}
    for index, row in pd.read_csv(os.path.join(source_path, "gvc_doc_to_event.csv")).iterrows():
        subtopic_structure_dict[row["doc-id"]] = row["event-id"]

    #
    with open(os.path.join(source_path, 'verbose.conll'), encoding="utf-8") as f:
        conll_str = f.read()

    conll_lines = conll_str.split("\n")
    doc_id_prev = ""
    orig_sent_id_prev = ""
    sent_id = 0
    token_id = 0
    mentions_dict = {}
    mention_id = ""
    doc_title_dict = {}
    coref_dict = {}

    for i, conll_line in tqdm(enumerate(conll_lines), total=len(conll_lines)):
        if i+1 == len(conll_lines):
            break

        if "#begin document" in conll_line or "#end document" in conll_line:
            continue

        original_key, token, part_of_text, chain_description, chain_value = conll_line.split("\t")
        try:
            doc_id, orig_sent_id, _ = original_key.split(".")
        except ValueError:
            doc_id, orig_sent_id = original_key.split(".")

        chain_id = re.sub("\D+", "", chain_value)
        subtopic_id = subtopic_structure_dict[doc_id]
        topic_subtopic_doc = f'{topic_id}/{subtopic_id}/{doc_id}'

        # Step 1: collect original tokenized text
        if doc_id != doc_id_prev:
            sent_id = 0
            token_id = 0
        else:
            if orig_sent_id_prev != orig_sent_id:
                sent_id += 1
                token_id = 0

        if part_of_text == "TITLE":
            if doc_id not in doc_title_dict:
                doc_title_dict[doc_id] = []
            doc_title_dict[doc_id].append(token)

        # Step 3: collect coreference chains
        if chain_value.strip() == f'({chain_id}' or chain_value.strip() == f'({chain_id})':
            coref_dict[chain_id] = coref_dict.get(chain_id, 0) + 1
            mention_id = shortuuid.uuid(original_key)
            if chain_id == "0":
                chain_id = mention_id
                coref_dict[chain_id] = coref_dict.get(chain_id, 0) + 1

            # Step 2: collect original mentions
            mentions_dict[mention_id] = {
                COREF_CHAIN: chain_id,
                DESCRIPTION: chain_description,
                MENTION_ID: mention_id,
                DOC_ID: doc_id,
                DOC: "",
                SENT_ID: int(sent_id),
                SUBTOPIC_ID: str(subtopic_id),
                SUBTOPIC: str(subtopic_id),
                TOPIC_ID: topic_id,
                TOPIC: topic_name,
                COREF_TYPE: IDENTITY,
                CONLL_DOC_KEY: topic_subtopic_doc,
                "words": []}
            mentions_dict[mention_id]["words"].append((token, token_id))
            if chain_value.strip() == f'({chain_id})':
                mention_id = ""

        elif mention_id and chain_value == chain_id:
            mentions_dict[mention_id]["words"].append((token, token_id))

        elif chain_value.strip() == f'{chain_id})':
            mentions_dict[mention_id]["words"].append((token, token_id))
            mention_id = ""

        conll_df = pd.concat([conll_df, pd.DataFrame({
            TOPIC_SUBTOPIC_DOC: topic_subtopic_doc,
            DOC_ID: conll_line.split("\t")[0].split(".")[0],
            SENT_ID: sent_id,
            TOKEN_ID: token_id,
            TOKEN: token.replace("NEWLINE", "//n"),
            REFERENCE: "-"
        }, index=[f'{doc_id}/{sent_id}/{token_id}'])])
        token_id += 1
        doc_id_prev = doc_id
        orig_sent_id_prev = orig_sent_id

    # Step 4: reparse the documents
    for topic_subtopic_doc in tqdm(conll_df[TOPIC_SUBTOPIC_DOC].unique(), desc="Reparsing documents"):
        doc_id = topic_subtopic_doc.split("/")[-1]
        doc_df_orig = conll_df[(conll_df[TOPIC_SUBTOPIC_DOC] == topic_subtopic_doc)]
        # generate sentence doc with spacy
        token_id_global = 0
        for sent_id, sent_df in doc_df_orig.groupby(SENT_ID):
            sent_tokens = sent_df[TOKEN].tolist()
            sentence_str = ""
            for t in sent_tokens:
                sentence_str, _, _ = append_text(sentence_str, t)
            sentence_str = correct_whitespaces(sentence_str)

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

    # parse information about mentions
    for mention_id, mention in tqdm(mentions_dict.items()):
        word_start = mention["words"][0][1]
        word_end = mention["words"][-1][1]
        # coref_id = mention_orig["entity"]
        doc_id = mention[DOC_ID]
        sent_id = mention[SENT_ID]
        markable_df = conll_df.loc[f'{doc_id}/{sent_id}/{word_start}': f'{doc_id}/{sent_id}/{word_end}']
        if not len(markable_df):
            continue

        sent_id = list(markable_df[SENT_ID].values)[0]
        # mention attributes
        init_token_numbers = [int(t) for t in list(markable_df[TOKEN_ID].values)]
        token_str = ""
        tokens_text = list(markable_df[TOKEN].values)
        for token in tokens_text:
            token_str, word_fixed, no_whitespace = append_text(token_str, token)

        # Step 5: match the originally tokenized mentions to the reparsed texts to collect mention attributes, e.g., POS, NER, etc
        topic_subtopic_doc = mention[CONLL_DOC_KEY]
        doc = topic_sent_dict[topic_subtopic_doc][sent_id]
        sent_str = "".join([t.text + t.whitespace_ for t in doc])

        sent_df = preprocessed_df[
            (preprocessed_df[SENT_ID] == sent_id) & (preprocessed_df[TOPIC_SUBTOPIC_DOC] == topic_subtopic_doc)]
        sent_tokens_text = [t.text for t in doc]
        start_token_id, _ = find_sub_list(tokens_text, sent_tokens_text)
        if start_token_id == -1:
            # LOGGER.warning(f"\'{token_str}\' not found in \'{sent_str}\'")
            # continue

            try_again = 2
            best_matching_start_token = []

            while try_again > 0:
                if try_again == 1:
                    token_str = correct_whitespaces(correct_whitespaces(token_str))

                try:
                    positions = [(match.start(), match.end()) for match in re.finditer(fr'{token_str}', sent_str)]
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

            token_ids = list(
                range(start_token_id, start_token_id + len(best_matching_start_token[best_matching_start_token_id])))
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
        mention_context_ids = list(doc_df.iloc[context_min_id:context_max_id + 1][TOKEN_ID_GLOBAL].values)
        sentence_start_id = mention_context_ids.index(sent_df[TOKEN_ID_GLOBAL].values.tolist()[0])

        mention_ner = doc[mention_head_id].ent_type_ if doc[mention_head_id].ent_type_ != "" else "O"
        mention_type = "EVENT"
        mention.update({
            MENTION_NER: mention_ner,
            MENTION_HEAD_POS: doc[mention_head_id].pos_,
            MENTION_HEAD_LEMMA: doc[mention_head_id].lemma_,
            MENTION_HEAD: doc[mention_head_id].text,
            MENTION_HEAD_ID: int(mention_head_id),
            DOC: "_".join(doc_title_dict[doc_id]),
            IS_SINGLETON: coref_dict[mention[COREF_CHAIN]] == 1,
            MENTION_TYPE: mention_type[:3],
            MENTION_FULL_TYPE: mention_type,
            MENTION_CONTEXT: mention_context_text,
            TOKENS_NUMBER_CONTEXT: tokens_number_context,
            TOKENS_NUMBER: token_ids,
            TOKENS_STR: token_str,
            TOKENS_TEXT: tokens_text,
            CONTEXT_START_END_GLOBAL_ID: [int(context_min_id), int(context_max_id)],
            MENTION_SENTENCE_CONTEXT_START_END_ID: [sentence_start_id, sentence_start_id + len(sent_df)],
            MENTION_HEAD_ID_CONTEXT: int(head_id_context),
            SPLIT: "tbd"
        })
        mention.pop("words")
        mention = reorganize_field_order(mention)
        event_mentions.append(mention)

    conll_df_labeled = pd.DataFrame()
    all_mentions = []
    LOGGER.info(f'Splitting GVC into train/dev/test subsets...')
    for subset, subtopic_ids in train_dev_test_split_dict.items():
        if subset == "dev":
            subset = "val"

        LOGGER.info(f'Creating data for {subset} subset...')
        split_folder = os.path.join(OUT_PATH, subset)
        if subset not in os.listdir(OUT_PATH):
            os.mkdir(split_folder)

        selected_event_mentions = []
        for mention in event_mentions:
            if int(mention[SUBTOPIC_ID]) in subtopic_ids:
                mention[SPLIT] = subset
                selected_event_mentions.append(mention)
        all_mentions.extend(selected_event_mentions)

        with open(os.path.join(split_folder, MENTIONS_ENTITIES_JSON), "w", encoding='utf-8') as file:
            json.dump([], file)

        with open(os.path.join(split_folder, MENTIONS_EVENTS_JSON), "w", encoding='utf-8') as file:
            json.dump(selected_event_mentions, file)

        conll_df_split = pd.DataFrame()
        for t_id in subtopic_ids:
            conll_df_split = pd.concat([conll_df_split,
                                        preprocessed_df[preprocessed_df[TOPIC_SUBTOPIC_DOC].str.contains(f'{t_id}/')]], axis=0)
        conll_df_split[SPLIT] = subset
        conll_df_labeled = pd.concat([conll_df_labeled, make_save_conll(conll_df_split, selected_event_mentions, split_folder, return_df_only=True)])

    df_all_mentions = pd.DataFrame()
    for mention in tqdm(all_mentions):
        df_all_mentions = pd.concat([df_all_mentions, pd.DataFrame({
            attr: str(value) if type(value) == list else value for attr, value in mention.items()
        }, index=[mention[MENTION_ID]])], axis=0)

    df_all_mentions.to_parquet(os.path.join(OUT_PATH, MENTIONS_ALL_PARQUET), engine="pyarrow")
    conll_df_labeled.to_parquet(os.path.join(OUT_PATH, DOCUMENTS_ALL_PARQUET), engine="pyarrow")

    LOGGER.info(f'Done! \nNumber of unique mentions: {len(df_all_mentions)} '
                f'\nNumber of unique chains: {len(set(df_all_mentions[COREF_CHAIN].values))} ')
    LOGGER.info(f'Parsing of GVC done!')


if __name__ == '__main__':
    LOGGER.info(f"Processing GVC {source_path[-34:].split('_')[2]}.")
    conv_files()
