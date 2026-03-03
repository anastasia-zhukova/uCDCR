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
from utils import *
from nltk import Tree
from tqdm import tqdm
import warnings
import shortuuid
from setup import *
from logger import LOGGER

warnings.filterwarnings('ignore')

ECB_PARSING_FOLDER = os.path.join(os.getcwd())
ECBPLUS_FILE = "ecbplus.xml"
ECB_FILE = "ecb.xml"
IS_TEXT, TEXT = "is_text", TEXT

source_path = os.path.join(ECB_PARSING_FOLDER, ECBPLUS_FOLDER_NAME)
result_path = os.path.join(ECB_PARSING_FOLDER, OUTPUT_FOLDER_NAME, "test_parsing")
out_path = os.path.join(ECB_PARSING_FOLDER, OUTPUT_FOLDER_NAME)
path_sample = os.path.join(os.getcwd(), "..", SAMPLE_DOC_JSON)

nlp = get_spacy()

validated_sentences_df = pd.read_csv(os.path.join(ECB_PARSING_FOLDER, ECBPLUS_FOLDER_NAME,
                                                  "ECBplus_coreference_sentences.csv")).set_index(
    ["Topic", "File", "Sentence Number"])

with open(os.path.join(ECB_PARSING_FOLDER, "train_val_test_split.json"), "r") as file:
    train_dev_test_split_dict = json.load(file)

with open(os.path.join(ECB_PARSING_FOLDER, "subtopic_names.json"), "r") as file:
    subtopic_names_dict = json.load(file)


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


def convert_files(topic_number_to_convert=3, check_with_list=True):
    coref_dics = {}
    topic_sent_dict = {}
    preprocessed_df = pd.DataFrame()
    selected_topics = os.listdir(source_path)[:topic_number_to_convert]
    conll_df = pd.DataFrame()
    entity_mentions = []
    event_mentions = []
    topic_names = []
    need_manual_review_mention_head = {}
    counter_annotated_mentions = 0
    counter_parsed_mentions = 0

    for topic_id in selected_topics:
        # if topic_id not in ["30"]:
        #     continue
        if topic_id == "__MACOSX":
            continue

        # if a file with confirmed sentences
        if os.path.isfile(os.path.join(source_path, topic_id)):
            continue

        LOGGER.info(f'Converting topic {topic_id}')
        diff_folders = {ECB_FILE: [], ECBPLUS_FILE: []}

        # assign the different folders according to the topics in the variable "diff_folders"
        for topic_file in os.listdir(os.path.join(source_path, topic_id)):
            if ECBPLUS_FILE in topic_file:
                diff_folders[ECBPLUS_FILE].append(topic_file)
            else:
                diff_folders[ECB_FILE].append(topic_file)

        for annot_folders in list(diff_folders.values()):
            t_number = annot_folders[0].split(".")[0].split("_")[0]
            t_name = re.search(r'[a-z]+', annot_folders[0].split(".")[0])[0]
            subtopic_id = t_number + t_name
            topic_names.append(subtopic_id)
            coref_dict = {}

            # for every themed-file in "commentated files"
            for topic_file in tqdm(annot_folders):
                doc_id = re.search(r'[\d+]+', topic_file.split(".")[0].split("_")[1])[0]
                topic_subtopic_doc = f'{topic_id}/{subtopic_id}/{doc_id}'
                topic_sent_dict[topic_subtopic_doc] = {}

                # import the XML-Datei topic_file
                tree = ET.parse(os.path.join(source_path, topic_id, topic_file))
                root = tree.getroot()

                token_dict, mentions, mentions_map = {}, {}, {}

                t_id = -1
                old_sent = -1
                for elem in root:
                    # Step 1: collect original tokenized text
                    if elem.tag == "token":
                        try:
                            # # increase t_id value by 1 if the sentence value in the xml element ''equals the value of old_sent
                            if old_sent == int(elem.attrib[SENTENCE]):
                                t_id += 1
                                # else set old_sent to the value of sentence and t_id to 0
                            else:
                                old_sent = int(elem.attrib[SENTENCE])
                                t_id = 0

                            # fill the token-dictionary with fitting attributes
                            token_dict[elem.attrib[T_ID]] = {TEXT: elem.text, SENT: elem.attrib[SENTENCE], NUM: t_id,
                                                             ID: elem.attrib[NUM]}

                            conll_df = pd.concat([conll_df, pd.DataFrame({
                                TOPIC_SUBTOPIC_DOC: topic_subtopic_doc,
                                DOC_ID: doc_id,
                                SENT_ID: int(token_dict[elem.attrib[T_ID]][SENT]),
                                # TOKEN_ID: int(t_id),
                                TOKEN_ID: int(token_dict[elem.attrib[T_ID]][ID]),
                                TOKEN: token_dict[elem.attrib[T_ID]][TEXT],
                                REFERENCE: "-"
                            }, index=[elem.attrib[T_ID]])])

                        except KeyError as e:
                            LOGGER.warning(f'Value with key {e} not found and will be skipped from parsing.')

                    if elem.tag == "Markables":
                        # Step 2: collect original mentions
                        for i, subelem in enumerate(elem):
                            mention_tokens_ids_global = [token.attrib[T_ID] for token in subelem]
                            mention_tokens_ids_global.sort(key=int)  # sort tokens by their id
                            # doc_df = conll_df[(conll_df[TOPIC_SUBTOPIC_DOC] == topic_subtopic_doc)]

                            # if "tokens" has values -> fill the "mention" dict with the value of the corresponding m_id
                            if len(mention_tokens_ids_global):
                                # take in all consecutive tokes
                                mention_tokens_ids_global_int = [int(i) for i in mention_tokens_ids_global]
                                mention_tokens_ids_global = list(range(min(mention_tokens_ids_global_int), max(mention_tokens_ids_global_int) + 1))

                                sent_id = int(token_dict[str(mention_tokens_ids_global[0])][SENT])
                                counter_annotated_mentions += 1

                                # tokenize the mention text
                                tokens_text, token_ids = [], []
                                tokens_str = ""
                                for t_id in mention_tokens_ids_global:
                                    tokens_text.append(token_dict[str(t_id)][TEXT])
                                    token_ids.append(int(token_dict[str(t_id)][NUM]))
                                    tokens_str, _, _ = append_text(tokens_str, token_dict[str(t_id)][TEXT])

                                if len(set(tokens_str).intersection(set(string.punctuation))):
                                    tokens_str = correct_whitespaces(correct_whitespaces(tokens_str))

                                d_id =  topic_file.split(".")[0]
                                mention_id = "ECBPlus_" + shortuuid.uuid(f"{tokens_str}_{topic_file}_{d_id}_{sent_id}_{subelem.tag}")

                                mentions[subelem.attrib[M_ID]] = {MENTION_TYPE: subelem.tag,
                                                                  MENTION_FULL_TYPE: subelem.tag,
                                                                  TOKENS_STR: tokens_str.strip(),
                                                                  MENTION_ID: mention_id,
                                                                  TOKENS_NUMBER: token_ids,
                                                                  TOKENS_TEXT: tokens_text,
                                                                  DOC:d_id,
                                                                  SENT_ID: sent_id,
                                                                  TOPIC_SUBTOPIC_DOC: topic_subtopic_doc,
                                                                  TOPIC: topic_subtopic_doc}
                            else:
                                # form coreference chain
                                # m_id points to the target
                                if "ent_type" in subelem.attrib:
                                    mention_type_annot = subelem.attrib["ent_type"]
                                elif "class" in subelem.attrib:
                                    mention_type_annot = subelem.attrib["class"]
                                elif "type" in subelem.attrib:
                                    mention_type_annot = subelem.attrib["type"]
                                else:
                                    mention_type_annot = subelem.tag

                                if "instance_id" in subelem.attrib:
                                    id_ = subelem.attrib["instance_id"]
                                else:
                                    descr = subelem.attrib["TAG_DESCRIPTOR"]
                                    id_ = ""

                                    for coref_id, coref_vals in coref_dict.items():
                                        if coref_vals[DESCRIPTION] == descr and coref_vals[COREF_TYPE] == mention_type_annot \
                                                and coref_vals["subtopic"] == subtopic_id and mention_type_annot:
                                            id_ = coref_id
                                            break

                                    if not len(id_):
                                        LOGGER.warning(
                                            f"Document {doc_id}: {subelem.tag} {subelem.attrib} doesn\'t have attribute instance_id. It will be created.")
                                        id_ = f'{mention_type_annot[:3]}_{topic_subtopic_doc}_{subelem.attrib["m_id"]}_{shortuuid.uuid()}'

                                    if not len(id_):
                                        continue

                                    subelem.attrib["instance_id"] = id_

                                if not len(id_):
                                    continue

                                if id_ not in coref_dict:
                                    coref_dict[id_] = {DESCRIPTION: subelem.attrib["TAG_DESCRIPTOR"],
                                                       COREF_TYPE: mention_type_annot,
                                                       "subtopic": subtopic_id}

                    if elem.tag == "Relations":
                        # Step 3: collect coreference chains
                        # for every false create a false-value in "mentions_map"
                        mentions_map = {m: False for m in list(mentions)}
                        for i, subelem in enumerate(elem):
                            tmp_instance_id = "None"
                            for j, subsubelm in enumerate(subelem):
                                if subsubelm.tag == "target":
                                    for prevelem in root:
                                        if prevelem.tag != "Markables":
                                            continue

                                        for k, prevsubelem in enumerate(prevelem):
                                            if prevsubelem.get("instance_id") is None:
                                                continue

                                            if subsubelm.attrib["m_id"] == prevsubelem.attrib["m_id"]:
                                                tmp_instance_id = prevsubelem.attrib["instance_id"]
                                                break

                            if tmp_instance_id != "None":
                                try:
                                    if "r_id" not in coref_dict[tmp_instance_id]:
                                        coref_dict[tmp_instance_id].update({
                                            "r_id": subelem.attrib["r_id"],
                                            # "coref_type": subelem.tag,
                                            "mentions": {mentions[m.attrib["m_id"]][MENTION_ID]: mentions[m.attrib["m_id"]]
                                                         for m in subelem if
                                                         m.tag == "source"}
                                        })
                                    else:
                                        for m in subelem:
                                            if m.tag == "source":
                                                mention_id_local = mentions[m.attrib["m_id"]][MENTION_ID]
                                                if mention_id_local in coref_dict[tmp_instance_id]["mentions"]:
                                                    continue

                                                coref_dict[tmp_instance_id]["mentions"][mention_id_local] = mentions[
                                                    m.attrib["m_id"]]
                                except KeyError as e:
                                    LOGGER.warning(
                                        f'Document {doc_id}: Mention with ID {str(e)} is not among the Markables and will be skipped.')
                            for m in subelem:
                                mentions_map[m.attrib[M_ID]] = True

                        for i, (m_id, used) in enumerate(mentions_map.items()):
                            if used:
                                continue

                            m = mentions[m_id]
                            chain_id_created = "Singleton_" + m[MENTION_TYPE][:3] + shortuuid.uuid()[:7]
                            if chain_id_created not in coref_dict:
                                coref_dict[chain_id_created] = {
                                    "r_id": str(10000 + i),
                                    COREF_TYPE: m[MENTION_TYPE],
                                    MENTIONS: {m_id: m},
                                    DESCRIPTION: m[TOKENS_STR],
                                    "subtopic": subtopic_id
                                }
                            else:
                                coref_dict[chain_id_created].update(
                                    {
                                        "r_id": str(10000 + i),
                                        COREF_TYPE: m[MENTION_TYPE],
                                        MENTIONS: {m_id: m},
                                        "subtopic": subtopic_id,
                                        DESCRIPTION:  m[TOKENS_STR],
                                    })

                # Step 4: reparse the documents
                doc_df_orig = conll_df[(conll_df[TOPIC_SUBTOPIC_DOC] == topic_subtopic_doc)]
                # generate sentence doc with spacy
                token_id_global = 0
                for sent_id, sent_df in doc_df_orig.groupby(SENT_ID):
                    sent_tokens = sent_df[TOKEN].tolist()
                    sentence_str = ""
                    for t in sent_tokens:
                        sentence_str, _, _ = append_text(sentence_str, t)

                    sentence_str_cor = correct_whitespaces(sentence_str)
                    if not all([t in sentence_str_cor for t in sent_tokens]):
                        sentence_str_cor = sentence_str

                    doc = nlp(sentence_str_cor)
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

                    topic_sent_dict[topic_subtopic_doc][sent_id] = doc
                    token_id_global += len(doc)

            coref_dics[topic_id] = coref_dict
            entity_mentions_local = []
            event_mentions_local = []
            mentions_local = []

            for chain_id, chain_vals in coref_dict.items():

                if MENTIONS not in chain_vals:
                    continue

                prev_new_chain_id = ""

                # Step 5: match the originally tokenized mentions to the reparsed texts to collect mention attributes, e.g., POS, NER, etc
                for m in chain_vals[MENTIONS].values():
                    tokens_text = m[TOKENS_TEXT]
                    token_str = m[TOKENS_STR]
                    sent_id = int(m[SENT_ID])
                    init_token_numbers = [int(t) for t in m[TOKENS_NUMBER]]
                    doc = topic_sent_dict[m[TOPIC_SUBTOPIC_DOC]][sent_id]
                    sent_str = "".join([t.text + t.whitespace_ for t in doc])
                    sent_tokens_text = [t.text for t in doc]

                    start_token_id, _ = find_sub_list(tokens_text, sent_tokens_text)
                    sent_df = preprocessed_df[
                        (preprocessed_df[SENT_ID] == sent_id) & (
                                preprocessed_df[TOPIC_SUBTOPIC_DOC] == m[TOPIC_SUBTOPIC_DOC])]

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

                    counter_parsed_mentions += 1

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
                    doc_df = preprocessed_df[preprocessed_df[TOPIC_SUBTOPIC_DOC] == m[TOPIC_SUBTOPIC_DOC]]

                    if token_mention_start_id - CONTEXT_RANGE < 0:
                        context_min_id = 0
                        tokens_number_context = list(sent_df[sent_df[TOKEN_ID].isin(token_ids)][TOKEN_ID_GLOBAL])
                        head_id_context = list(sent_df[sent_df[TOKEN_ID] == mention_head_id][TOKEN_ID_GLOBAL])[0]
                        context_max_id = min(token_mention_start_id + CONTEXT_RANGE, len(doc_df))
                        # round it to the full sentence
                        context_sent_id = doc_df[doc_df[TOKEN_ID_GLOBAL] == context_max_id-1][SENT_ID][0]
                        context_max_id = doc_df[doc_df[SENT_ID] == context_sent_id][TOKEN_ID_GLOBAL][-1]
                    else:
                        context_min_id = token_mention_start_id - CONTEXT_RANGE
                        # round it to the full sentence
                        context_sent_id = doc_df[doc_df[TOKEN_ID_GLOBAL] == context_min_id][SENT_ID][0]
                        context_min_id = doc_df[doc_df[SENT_ID] == context_sent_id][TOKEN_ID_GLOBAL][0]

                        global_token_ids = list(sent_df[sent_df[TOKEN_ID].isin(token_ids)][TOKEN_ID_GLOBAL])
                        tokens_number_context = [int(t - context_min_id) for t in global_token_ids]
                        head_id_context = list(sent_df[sent_df[TOKEN_ID] == mention_head_id][TOKEN_ID_GLOBAL])[0] - context_min_id
                        context_max_id = min(token_mention_start_id + CONTEXT_RANGE, len(doc_df))
                        # round it to the full sentence
                        context_sent_id = doc_df[doc_df[TOKEN_ID_GLOBAL] == context_max_id-1][SENT_ID][0]
                        context_max_id = doc_df[doc_df[SENT_ID] == context_sent_id][TOKEN_ID_GLOBAL][-1]

                    mention_context_text = list(doc_df.iloc[context_min_id:context_max_id+1][TOKEN].values)
                    mention_context_id = list(doc_df.iloc[context_min_id:context_max_id+1][TOKEN_ID_GLOBAL].values)
                    context_sent_id_start = mention_context_id.index(sent_df[TOKEN_ID_GLOBAL].values.tolist()[0])

                    mention_ner = doc[mention_head_id].ent_type_ if doc[mention_head_id].ent_type_ != "" else "O"

                    if chain_id[:3] == "UNK":
                        if prev_new_chain_id.split("-")[0] == m[MENTION_TYPE]:
                            chain_id_new = prev_new_chain_id
                        else:
                            chain_id_new = f"{m[MENTION_TYPE]}-{subtopic_id}-regener-{shortuuid.uuid()}"
                            prev_new_chain_id = chain_id_new
                        LOGGER.info(f"New coref_id is generated: \"{m[TOKENS_STR]}\" in {chain_id} -> {chain_id_new}")
                        description = f"t{subtopic_id}_{m[MENTION_TYPE].lower()}_regener_{chain_id_new.split('-')[-1]}"
                    else:
                        chain_id_new = chain_id
                        description = chain_vals[DESCRIPTION]

                    # create variable "mention_id"
                    mention_id = f'ECBPlus_{m[DOC]}_{str(chain_id_new)}_{str(m[SENT_ID])}_{str(m[TOKENS_NUMBER][0])}_{shortuuid.uuid()[:4]}'

                    # Step 6: form a mention with all attributes
                    # create the dict. "mention" with all corresponding values
                    mention = {COREF_CHAIN: chain_id_new,
                               MENTION_ID: mention_id,
                               TOKENS_STR: m[TOKENS_STR],
                               DESCRIPTION: description,
                               COREF_TYPE: IDENTITY,
                               MENTION_TYPE: m[MENTION_TYPE][:3],
                               MENTION_FULL_TYPE: m[MENTION_TYPE],
                               TOKENS_TEXT: tokens_text,
                               TOKENS_NUMBER: [int(i) for i in token_ids],
                               MENTION_HEAD: mention_head,
                               MENTION_HEAD_ID: int(mention_head_id),
                               MENTION_HEAD_POS: doc[mention_head_id].pos_,
                               MENTION_HEAD_LEMMA: doc[mention_head_id].lemma_,
                               MENTION_NER: mention_ner,
                               SENT_ID: int(sent_id),
                               TOPIC_ID: t_number,
                               TOPIC: t_number,
                               SUBTOPIC_ID: subtopic_id,
                               SUBTOPIC: subtopic_names_dict[subtopic_id],
                               DOC_ID: m[TOPIC_SUBTOPIC_DOC].split("/")[-1],
                               DOC: m[DOC],
                               MENTION_CONTEXT: mention_context_text,
                               CONTEXT_START_END_GLOBAL_ID: [int(context_min_id), int(context_max_id)],
                               MENTION_SENTENCE_CONTEXT_START_END_ID: [context_sent_id_start, context_sent_id_start + len(sent_df)],
                               TOKENS_NUMBER_CONTEXT: tokens_number_context,
                               MENTION_HEAD_ID_CONTEXT: int(head_id_context),
                               IS_SINGLETON: bool(len(chain_vals[MENTIONS]) == 1),
                               CONLL_DOC_KEY: m[TOPIC_SUBTOPIC_DOC],
                               SPLIT: "tbd"
                               }
                    mention = reorganize_field_order(mention)

                    # if the first two entries of chain_id are "ACT" or "NEG", add the "mention" to the array "event_mentions_local"
                    if chain_id[:3] in ["ACT", "NEG"]:
                        event_mentions_local.append(mention)
                    # else add the "mention" to the array "event_mentions_local" and add the following values to the DF "summary_df"
                    else:
                        entity_mentions_local.append(mention)

                    # if not mention[IS_SINGLETON]:
                    mentions_local.append(mention)
            entity_mentions.extend(entity_mentions_local)
            event_mentions.extend(event_mentions_local)
            LOGGER.info(
                f'The annotated mentions ({counter_annotated_mentions}) and parsed mentions ({counter_parsed_mentions}).')

    # take only validated sentences
    entity_mention_validated = []
    for mention in entity_mentions:
        subtopic_suf = re.sub(r'\d+', '', mention[SUBTOPIC_ID])
        if (int(mention[TOPIC_ID]), f"{mention[DOC_ID]}{subtopic_suf}", mention[SENT_ID]) not in validated_sentences_df.index:
            continue
        entity_mention_validated.append(mention)

    # take only validated sentences
    event_mention_validated = []
    for mention in event_mentions:
        subtopic_suf = re.sub(r'\d+', '', mention[SUBTOPIC_ID])
        if (int(mention[TOPIC_ID]), f"{mention[DOC_ID]}{subtopic_suf}", mention[SENT_ID]) not in validated_sentences_df.index:
            continue
        event_mention_validated.append(mention)

    # mentions_df = pd.DataFrame()
    # for mention in tqdm(entity_mention_validated + event_mention_validated):
    #     mentions_df = pd.concat([mentions_df, pd.DataFrame({
    #         attr: str(value) if type(value) == list else value for attr, value in mention.items()
    #     }, index=[mention[MENTION_ID]])], axis=0)

    for save_options in [
        # [entity_mentions, event_mentions, os.path.join(ECB_PARSING_FOLDER, OUTPUT_FOLDER_NAME+"-unvalidated")],
                         [entity_mention_validated, event_mention_validated, out_path]]:
        entity_m, event_m, save_folder = save_options
        LOGGER.info(f'Saving ECB+ into {save_folder}...')

        conll_df_labels = pd.DataFrame()
        all_mentions = []

        LOGGER.info(f'Splitting ECB+ into train/dev/test subsets...')
        for subset, topic_ids in train_dev_test_split_dict.items():
            LOGGER.info(f'Creating data for {subset} subset...')
            split_folder = os.path.join(save_folder, subset)
            if subset not in os.listdir(save_folder):
                os.mkdir(split_folder)

            selected_entity_mentions = []
            for mention in entity_m:
                if int(mention[TOPIC_ID]) in topic_ids:
                    mention[SPLIT] = subset
                    selected_entity_mentions.append(mention)

            selected_event_mentions = []
            for mention in event_m:
                if int(mention[TOPIC_ID]) in topic_ids:
                    mention[SPLIT] = subset
                    selected_event_mentions.append(mention)

            all_mentions.extend(selected_event_mentions)
            all_mentions.extend(selected_entity_mentions)

            with open(os.path.join(split_folder, MENTIONS_ENTITIES_JSON), "w", encoding='utf-8') as file:
                json.dump(selected_entity_mentions, file)

            with open(os.path.join(split_folder, MENTIONS_EVENTS_JSON), "w", encoding='utf-8') as file:
                json.dump(selected_event_mentions, file)

            conll_df_split = pd.DataFrame()
            for t_id in topic_ids:
                conll_df_split = pd.concat([conll_df_split,
                                            preprocessed_df[preprocessed_df[TOPIC_SUBTOPIC_DOC].str.contains(f'{t_id}/')]], axis=0)
            conll_df_split[SPLIT] = subset
            conll_df_split_labels = make_save_conll(conll_df_split, selected_event_mentions+selected_entity_mentions, split_folder, return_df_only=True)
            conll_df_labels = pd.concat([conll_df_labels, conll_df_split_labels])

        # create a csv. file out of the mentions summary_df
        df_all_mentions = pd.DataFrame()
        for mention in all_mentions:
            df_all_mentions = pd.concat([df_all_mentions, pd.DataFrame({
                attr: str(value) if type(value) == list else value for attr, value in mention.items()
            }, index=[mention[MENTION_ID]])], axis=0)

        df_all_mentions.to_parquet(os.path.join(save_folder, MENTIONS_ALL_PARQUET), engine="pyarrow")
        LOGGER.info(f'Done! \nNumber of unique mentions: {len(df_all_mentions)} '
                    f'\nNumber of unique chains: {len(set(df_all_mentions[COREF_CHAIN].values))} ')

    conll_df_labels.to_parquet(os.path.join(save_folder, DOCUMENTS_ALL_PARQUET), engine="pyarrow")
    LOGGER.info(f'Parsing ECB+ is done!')
    LOGGER.info(f'The annotated mentions ({counter_annotated_mentions}) and parsed mentions ({counter_parsed_mentions}).')


# main function for the input which topics of the ecb corpus are to be converted
if __name__ == '__main__':
    topic_num = 45
    convert_files(topic_num)

