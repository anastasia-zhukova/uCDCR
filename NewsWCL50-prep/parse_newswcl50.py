from setup import *
from logger import LOGGER
from utils import make_save_conll
AGGR_FILENAME = 'aggr_m_conceptcategorization.csv'

import spacy
import os
import json
import pandas as pd
import shortuuid
import re
from tqdm import tqdm
from typing import List
from utils import reorganize_field_order

DATASET_PATH = os.path.join(os.getcwd(), NEWSWCL50_FOLDER_NAME)
TOPICS_FOLDER = os.path.join(DATASET_PATH, "topics")
ANNOTATIONS_FOLDER = os.path.join(DATASET_PATH, "annotations")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# loading spacy NLP
nlp = spacy.load('en_core_web_sm')


def check_continuous(token_numbers: List[int]):
    """
    Checks whether a list of token ints is continuous
    Args:
        token_numbers:

    Returns: boolean
    """
    return token_numbers == list(range(token_numbers[0], token_numbers[-1] + 1))


if __name__ == '__main__':
    LOGGER.info("Starting routine... Retrieving data from specified directories.")
    with open(os.path.join(os.getcwd(), SAMPLE_DOC_JSON), "r") as file:
        sample_fields = list(json.load(file).keys())
    topic_df = pd.DataFrame(columns=sample_fields)
    topic_counter = 0
    preprocessed_df = pd.DataFrame()
    topic_sent_dict = {}

    # read topic files
    for subfolder_name in os.listdir(TOPICS_FOLDER):
        LOGGER.info(f'Reading files from {subfolder_name}.')

        # iterate over every file
        for file_name in os.listdir(os.path.join(TOPICS_FOLDER, subfolder_name)):
            full_filename = os.path.join(TOPICS_FOLDER, subfolder_name, file_name)

            with open(full_filename, encoding='utf-8', mode='r') as file:
                topic_dict = {}
                for k,v in json.load(file).items():
                    topic_dict[k] = v if type(v) != list else ", ".join(v)
                topic_df = pd.concat([topic_df, pd.DataFrame(topic_dict, index=[topic_counter])], ignore_index=True, axis=0)
            topic_counter =+ 1

    # Step 4: reparse the documents
    docs = {}
    topic_df[CONCAT_TEXT] = ["\n".join([topic_df.loc[index, field] for field in [TITLE, DESCRIPTION, TEXT]
                                          if topic_df.loc[index, field]])
                               for index in list(topic_df.index)]

    # appending all articles as spacy docs to the dataframe
    for index, row in tqdm(topic_df.iterrows(), total=topic_df.shape[0]):
        doc = {}
        text = row[CONCAT_TEXT]
        token_id_global = 0
        doc_preproc = nlp(text)
        par_id = -1
        doc_id = row[SOURCE_DOMAIN]

        for sent_id, sent in enumerate(doc_preproc.sents):
            if "\n" in sent.orth_[:3] or sent_id == 0:
                par_id += 1
                doc[par_id] = {}
            doc[par_id][sent_id] = sent
            if sent.orth_[-1] == "\n":
                par_id += 1
                doc[par_id] = {}
            # tokens_id_global = list(range(token_id_global, token_id_global + len(sent)))
            # preprocessed_df = pd.concat([preprocessed_df, pd.DataFrame({
            #     TOPIC_SUBTOPIC_DOC: topic_subtopic_doc,
            #     DOC_ID: doc_id,
            #     SENT_ID: int(sent_id),
            #     TOKEN_ID: list(range(len(sent))),
            #     TOKEN: [t.text for t in sent],
            #     TOKEN_ID_GLOBAL: tokens_id_global,
            #     CHAR_ID_START: [t.idx for t in sent],
            #     WHITESPACE_AFTER: [bool(len(t.whitespace_)) for t in sent]
            # }, index=[f'{topic_subtopic_doc}_{i}' for i in tokens_id_global])])
            #
            # if topic_subtopic_doc not in topic_sent_dict:
            #     topic_sent_dict[topic_subtopic_doc] = {}
            # topic_sent_dict[topic_subtopic_doc][sent_id] = sent
            # token_id_global += len(doc)

        docs[row[SOURCE_DOMAIN]] = doc

    #read other csvs for the annotations
    df_annotations = pd.DataFrame()
    topics_dict = {}

    for file_name in os.listdir(ANNOTATIONS_FOLDER):
        full_filename = os.path.join(ANNOTATIONS_FOLDER, file_name)
        LOGGER.info(f'Executing code for {str(full_filename)}')
        df_tmp = pd.read_csv(full_filename)
        df_tmp = df_tmp[~df_tmp[CODE].str.contains("Properties")]
        topic_id = int(file_name.split("_")[0])
        df_tmp[TOPIC_ID] = topic_id

        topics_dict[topic_id] = file_name.split(".")[0]
        df_annotations = pd.concat([df_annotations, df_tmp], ignore_index=True, axis=0)

    # Open the aggregated file to get the entity types per code
    concept_df = pd.read_csv(os.path.join(os.getcwd(), NEWSWCL50_FOLDER_NAME, AGGR_FILENAME))
    concept_df = concept_df[~concept_df[TOPIC_ID].str.contains("ecb")]
    concept_df[TOPIC_ID]= concept_df[TOPIC_ID].astype(int)

    # assign every segment that gets mentioned to a specific code and entity type
    df_annotations = pd.merge(df_annotations, concept_df, how="left", on=[TOPIC_ID, CODE])

    # make sure no NA values are present after merging
    df_annotations = df_annotations[df_annotations.type.notna()]
    df_annotations.reset_index(drop=True, inplace=True)

    # create coref_chain ids which show the connection/corellation of many segment mentions within the same topic
    coref_chains = {}
    # chains_list = []
    df_annotations[COREF_CHAIN] = [""] * len(df_annotations)
    for index, row in df_annotations.iterrows():
        unique_chain_name = "_".join(str(row[col]) for col in [TOPIC_ID, CODE, TYPE])
        unique_chain_name = unique_chain_name.replace("\\", "_").replace(" ", "_")
        df_annotations.loc[index, COREF_CHAIN] = unique_chain_name
        coref_chains[unique_chain_name] = {MENTION_FULL_TYPE: row[TYPE],
                                           MENTION_TYPE: row[TYPE],
                                           COREF_CHAIN: unique_chain_name}

    LOGGER.info(f'Parsing and matching annotations in the texts...')
    mentions_df = pd.DataFrame()
    mentions_dict = {}
    not_found_list = {}

    for (coref_chain, doc_name, mention_orig, paragraph_orig, topic_id, code), group_df in\
            tqdm(df_annotations.groupby([COREF_CHAIN, DOCUMENT_NAME, SEGMENT, BEGINNING, TOPIC_ID, CODE])):
        # tokenize the segment
        mention_orig = re.sub("’", "'", mention_orig)
        mention_orig = re.sub("‘", "'", mention_orig)
        mention_orig = re.sub("“", "\"", mention_orig)
        mention_orig = re.sub("”", "\"", mention_orig)
        segment_doc = nlp(mention_orig)
        segment_tokenized = [t for t in segment_doc]
        found_mentions_counter = 0

        for paragraph_correction in [-1, -2, 0, -3, -4, 1, 2]:
            modified_par = paragraph_orig + paragraph_correction
            try:
                paragraph = docs[doc_name][modified_par]
            except KeyError:
                continue

            # iterate over every token of the sentence
            for sent_id, sentence in paragraph.items():
                # a counter to show with which token in the tokenized segment to match
                start_token_ids = []
                for i, token in enumerate(sentence):
                    if token.norm_.lower() == segment_tokenized[0].norm_.lower():
                        start_token_ids.append(i)
                if not start_token_ids:
                    continue

                for start_token_id in start_token_ids:
                    found_tokens = []

                    for i in range(len(segment_tokenized)):
                        try:
                            if sentence[i + start_token_id].norm_.lower() == segment_tokenized[i].norm_.lower():
                                found_tokens.append(sentence[i + start_token_id])
                        except IndexError:
                            continue

                    norm_annot = re.sub(r'\W+', "",  " ".join([t.norm_ for t in segment_tokenized])).lower()
                    norm_found = re.sub(r'\W+', "",  " ".join([t.norm_ for t in found_tokens])).lower()
                    if norm_annot != norm_found:
                        a = 1
                        continue
                    else:
                        found_mentions_counter += 1

                        # determine the head of the mention tokens
                        found_token_ids = list(range(start_token_id, start_token_id + len(found_tokens)))
                        found_tokens_global_ids = [t.i for t in found_tokens]
                        tokens_text = [t.text for t in found_tokens]
                        mention_head_token = None
                        mention_head_token_id = -1

                        for t_id, token in zip(found_token_ids, found_tokens):
                            # mention head's ancestors should not be in the found tokens
                            if all([a.i not in found_tokens_global_ids for a in token.ancestors]):
                                if token.pos_ in ["DET", "PUNCT"]:
                                    continue
                                mention_head_token = token
                                mention_head_token_id = t_id

                        if mention_head_token is None:
                            LOGGER.warning(f"A head for mention \'{tokens_text}\' not found. First token assigned: {tokens_text[0]}.")
                            mention_head_token = found_tokens[0]
                            mention_head_token_id = found_token_ids[0]

                        mention_id = "_".join(["NewsWCL50r", doc_name, str(sent_id), str(mention_head_token_id), shortuuid.uuid()[:4]])

                        context_min_id = max(min(found_tokens_global_ids) - CONTEXT_RANGE, 0)
                        if context_min_id > 0:
                            # round up to the full sentence
                            context_min_id = min(context_min_id, mention_head_token.doc[context_min_id].sent[0].i)

                        context_max_id = min(max(found_tokens_global_ids) + CONTEXT_RANGE, len(mention_head_token.doc) - 1 )
                        # round up to the full sentence
                        context_max_id = mention_head_token.doc[context_max_id].sent[-1].i
                        mention_context_str = [t.text for t in mention_head_token.doc[context_min_id: context_max_id + 1]]

                        if min(found_tokens_global_ids) - CONTEXT_RANGE < 0:
                            tokens_number_context = found_tokens_global_ids
                        else:
                            # context_min_id = min(found_tokens_global_ids) - CONTEXT_RANGE
                            tokens_number_context = [int(t - context_min_id) for t in found_tokens_global_ids]

                        head_id_context = tokens_number_context[found_token_ids.index(mention_head_token_id)]
                        used_token_ids_sent = [t.i for t in mention_head_token.doc[context_min_id:context_max_id+1]]

                        m = {COREF_CHAIN: coref_chain,
                             TOKENS_NUMBER: found_token_ids,
                             DOC_ID: doc_name,
                             DOC: doc_name,
                             SENT_ID: sent_id,
                             MENTION_TYPE: coref_chains[coref_chain][MENTION_TYPE][:3],
                             MENTION_FULL_TYPE: coref_chains[coref_chain][MENTION_FULL_TYPE],
                             MENTION_ID: mention_id,
                             TOPIC_ID: str(topic_id),
                             TOPIC: str(topic_id),
                             SUBTOPIC: topics_dict[topic_id],
                             SUBTOPIC_ID: str(topic_id),
                             DESCRIPTION: code,
                             COREF_TYPE: IDENTITY,
                             MENTION_NER: mention_head_token.ent_type_ if mention_head_token.ent_type_ else "O",
                             MENTION_HEAD_POS: mention_head_token.pos_,
                             MENTION_HEAD_LEMMA: mention_head_token.lemma_,
                             MENTION_HEAD: mention_head_token.text,
                             MENTION_HEAD_ID: mention_head_token_id,
                             IS_SINGLETON: False,
                             MENTION_CONTEXT: mention_context_str,
                             MENTION_HEAD_ID_CONTEXT: int(head_id_context),
                             TOKENS_NUMBER_CONTEXT: tokens_number_context,
                             CONTEXT_START_END_GLOBAL_ID: [int(context_min_id),
                                                           int(context_max_id)],
                             MENTION_SENTENCE_CONTEXT_START_END_ID: [used_token_ids_sent.index(mention_head_token.sent[0].i), used_token_ids_sent.index(mention_head_token.sent[-1].i),],
                             TOKENS_STR: "".join([t.text_with_ws for t in found_tokens]).strip(),
                             TOKENS_TEXT: [t.text for t in found_tokens],
                             CONLL_DOC_KEY: f'{topic_id}/{topic_id}/{doc_name}',
                             SPLIT: "tbd"
                             }
                        mentions_dict[mention_id] = reorganize_field_order(m)

                        mentions_df = pd.concat([mentions_df, pd.DataFrame({
                                               COREF_CHAIN: coref_chain,
                                               DOC_ID: doc_name,
                                               SENT_ID: sent_id,
                                               MENTION_HEAD_ID: mention_head_token_id,
                                               "char_length": len("".join([t.text_with_ws for t in found_tokens]))},
                            index=[mention_id])], axis=0)
            if found_mentions_counter:
                break

        if not found_mentions_counter:
            LOGGER.warning(f'A mention \"{mention_orig}\" was not found in document {doc_name} and will be skipped. ')

    LOGGER.warning(f'Not found annotations in the text ({len(not_found_list)}): \n{list(not_found_list)}')

    mentions_df = mentions_df.sort_values(by=[COREF_CHAIN, DOC_ID, SENT_ID, MENTION_HEAD_ID, "char_length"],
                                          ascending=[True, True, True, True, False])
    mentions_df_unique = mentions_df.drop_duplicates([COREF_CHAIN, DOC_ID, SENT_ID, MENTION_HEAD_ID], keep="first")
    mentions_unique_dict = {k:v for k, v in mentions_dict.items() if k in list(mentions_df_unique.index)}

    events = ["ACTION", "EVENT", "MISC"]
    mentions_events_list = []
    mentions_entities_list = []
    chain_df = mentions_df_unique[[DOC_ID, COREF_CHAIN]].groupby(COREF_CHAIN).count()
    for index, row in mentions_df_unique.iterrows():
        if mentions_unique_dict[index][MENTION_FULL_TYPE] in ["ACTOR-I"]:
            continue
        mentions_unique_dict[index][IS_SINGLETON] = bool(chain_df.loc[row[COREF_CHAIN], DOC_ID] == 1)
        if mentions_unique_dict[index][MENTION_FULL_TYPE] in events:
            mentions_events_list.append(mentions_unique_dict[index])
        else:
            mentions_entities_list.append(mentions_unique_dict[index])

    LOGGER.info("Generating conll...")
    conll_list = []
    token_keys_conll = []
    annot_token_dict = {}

    # first create a conll dataframe that gets a new row for each token
    for doc_id, doc in tqdm(docs.items()):
        if doc_id not in annot_token_dict:
            annot_token_dict[doc_id] = {}
        tokens_id_global = 0
        for par_id, par in doc.items():
            for sentence_id, sentence in par.items():
                if sentence_id not in annot_token_dict[doc_id]:
                    annot_token_dict[doc_id][sentence_id] = {}

                for token_id, token in enumerate(sentence):
                    if token_id not in annot_token_dict[doc_id][sentence_id]:
                        annot_token_dict[doc_id][sentence_id][token_id] = pd.DataFrame()

                    token_text = token.text
                    if "\n" in token_text:
                        token_text = token_text.replace("\n", "\\n")  # avoid unwanted like breaks in the conll file

                    conll_list.append(
                        {TOPIC_SUBTOPIC_DOC: f'{doc_id.split("_")[0]}/{doc_id.split("_")[0]}/{doc_id}',
                         DOC_ID: doc_id,
                         SENT_ID: sentence_id,
                         TOKEN_ID: token_id,
                         TOKEN: token_text,
                         TOKEN_ID_GLOBAL: tokens_id_global,
                         CHAR_ID_START: token.idx,
                         WHITESPACE_AFTER: bool(len(token.whitespace_)),
                         REFERENCE: "-"})
                    token_keys_conll.append("_".join([doc_id, str(sentence_id), str(token_id)]))
                    tokens_id_global += 1

    df_conll = pd.DataFrame(conll_list, index=token_keys_conll)
    output_path = os.path.join(os.getcwd(), OUTPUT_FOLDER_NAME)

    conll_df_labeled = make_save_conll(df_conll, mentions_entities_list + mentions_events_list, output_path, assign_reference_labels=True, return_df_only=True)

    with open("train_val_test_split.json", "r") as file:
        train_val_test_dict = json.load(file)

    conll_df_split_all = pd.DataFrame()

    all_mentions_list = []
    for split, topic_ids in train_val_test_dict.items():
        conll_df_split = pd.DataFrame()
        for topic_id in topic_ids:
            conll_df_split = pd.concat([conll_df_split,
                                        conll_df_labeled[conll_df_labeled[TOPIC_SUBTOPIC_DOC].str.contains(f'^{topic_id}/')]])
        # event_mentions_split = [m for m in mentions_events_list if m[TOPIC_ID] in topic_ids]
        event_mentions_split = []
        for m in mentions_events_list:
            if m[TOPIC_ID] in topic_ids:
                m[SPLIT] = split
                event_mentions_split.append(m)

        entity_mentions_split = []
        for m in mentions_entities_list:
            if m[TOPIC_ID] in topic_ids:
                m[SPLIT] = split
                entity_mentions_split.append(m)

        all_mentions_list.extend(event_mentions_split)
        all_mentions_list.extend(entity_mentions_split)

        output_folder_split = os.path.join(output_path, split)
        if not os.path.exists(output_folder_split):
            os.mkdir(output_folder_split)

        with open(os.path.join(output_folder_split, MENTIONS_EVENTS_JSON), 'w', encoding='utf-8') as file:
            json.dump(event_mentions_split, file)

        with open(os.path.join(output_folder_split, MENTIONS_ENTITIES_JSON), 'w', encoding='utf-8') as file:
            json.dump(entity_mentions_split, file)

        # make_save_conll(conll_df_split, event_mentions_split+entity_mentions_split, output_folder_split, assign_reference_labels=False)
        conll_df_split[SPLIT] = split
        conll_df_split_all = pd.concat([conll_df_split_all, conll_df_split])

    # save all mentions as csv
    df_all_mentions = pd.DataFrame()
    for mention in all_mentions_list:
        df_all_mentions = pd.concat([df_all_mentions, pd.DataFrame({
            attr: str(value) if type(value) == list else value for attr, value in mention.items()
        }, index=[mention[MENTION_ID]])], axis=0)

    df_all_mentions.to_parquet(os.path.join(output_path, MENTIONS_ALL_PARQUET), engine="pyarrow")

    conll_df_split_all.to_parquet(os.path.join(output_path, DOCUMENTS_ALL_PARQUET), engine="pyarrow")

    LOGGER.info(f'\nNumber of unique mentions: {len(mentions_df_unique)} \n'
                f'Number of unique chains: {len(set(df_all_mentions[COREF_CHAIN].values))} ')
    LOGGER.info("Parsing of NewsWCL50 is done!")
