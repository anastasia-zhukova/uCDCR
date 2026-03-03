import xml.etree.ElementTree as ET
import string
import copy
import re
import pandas as pd
import numpy as np
from nltk import Tree
import shortuuid
from tqdm import tqdm
from setup import *
from utils import *
from logger import LOGGER
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

MEANTIME_PARSING_FOLDER = os.path.join(os.getcwd())
OUT_PATH = os.path.join(MEANTIME_PARSING_FOLDER, OUTPUT_FOLDER_NAME)
MEANTIME_LANGS = [EN, ES, NL, IT]

if not spacy.util.is_package(SPACY_ES):
    spacy.cli.download(SPACY_ES)

if not spacy.util.is_package(SPACY_NL):
    spacy.cli.download(SPACY_NL)

if not spacy.util.is_package(SPACY_IT):
    spacy.cli.download(SPACY_IT)

lang_paths = {
    EN: {
        "source": os.path.join(MEANTIME_PARSING_FOLDER, MEANTIME_FOLDER_NAME, MEANTIME_FOLDER_NAME_ENGLISH),
        "nlp": spacy.load(SPACY_EN)},
    ES: {
        "source": os.path.join(MEANTIME_PARSING_FOLDER, MEANTIME_FOLDER_NAME,  MEANTIME_FOLDER_NAME_SPANISH),
        "nlp": spacy.load(SPACY_ES)},
    NL: {
        "source": os.path.join(MEANTIME_PARSING_FOLDER, MEANTIME_FOLDER_NAME,  MEANTIME_FOLDER_NAME_DUTCH),
        "nlp": spacy.load(SPACY_NL)},
    IT: {
        "source": os.path.join(MEANTIME_PARSING_FOLDER, MEANTIME_FOLDER_NAME, MEANTIME_FOLDER_NAME_ITALIAN),
        "nlp": spacy.load(SPACY_IT)}
}

meantime_types = {"PRO": "PRODUCT",
                  "FIN": "FINANCE",
                  "LOC": "LOCATION",
                  "ORG": "ORGANIZATION",
                  "OTH": "OTHER",
                  "PER": "PERSON",
                  "GRA": "GRAMMATICAL",
                  "SPE": "SPEECH_COGNITIVE",
                  "MIX": "MIXTURE"}


def to_nltk_tree(node):
    """
        Converts a sentence to a visually helpful tree-structure output.
        Can be used to double-check if a determined head is correct.
    """
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


def conv_files(languages_to_parse: List[str]=MEANTIME_LANGS):
    entity_mentions = []
    event_mentions = []
    topic_list = set()
    conll_df = pd.DataFrame(columns=[TOPIC_SUBTOPIC_DOC, DOC_ID, SENT_ID, TOKEN_ID, TOKEN, "lang", REFERENCE])

    coref_dict = {}
    preprocessed_df = pd.DataFrame()
    topic_sent_dict = {}
    annotated_counter = 0
    parsed_counter = 0

    for lang_key in lang_paths.keys():
        if lang_key not in languages_to_parse:
            continue

        source_path = lang_paths[lang_key]["source"]
        language = source_path[-34:].split("_")[2]

        LOGGER.info(f"Processing MEANTIME language {language}.")
        annotation_folders = [
            os.path.join(source_path, 'intra_cross-doc_annotation'),
            # os.path.join(source_path, 'intra-doc_annotation')
        ]

        for path in annotation_folders:
            dirs = os.listdir(path)

            for topic_id, topic in enumerate(dirs):  # for each topic folder
                LOGGER.info(f"Parsing of {topic} ({language}) [{path[-20:]}]. Please wait...")
                topic_id_compose = f'{topic_id}_{topic}'
                topic_files = os.listdir(os.path.join(path, topic))
                topic_list.add(topic_id_compose)

                # Step 1: collect original tokenized text
                for doc_file in tqdm(topic_files):
                    tree = ET.parse(os.path.join(path, topic, doc_file))
                    root = tree.getroot()
                    subtopic_full = doc_file.split(".")[0]
                    subtopic = subtopic_full.split("_")[0]
                    doc_id_full = f'{language}{doc_file.split(".")[0]}'
                    doc_id = f'{language}{subtopic}'
                    topic_subtopic_doc = f'{topic_id}/{subtopic}/{doc_id}'

                    token_dict, mentions, mentions_map = {}, {}, {}

                    t_id = -1
                    old_sent = 0

                    for elem in root:
                        if elem.tag == "token":
                            try:
                                if old_sent == int(elem.attrib["sentence"]):
                                    t_id += 1
                                else:
                                    old_sent = int(elem.attrib["sentence"])
                                    t_id = 0
                                token_dict[elem.attrib["t_id"]] = {"text": elem.text, "sent": elem.attrib["sentence"],
                                                                   "id": t_id}

                                if elem.tag == "token" and len(conll_df.loc[(conll_df[TOPIC_SUBTOPIC_DOC] == topic_subtopic_doc) &
                                                                            (conll_df[DOC_ID] == doc_id) &
                                                                            (conll_df[SENT_ID] == int(
                                                                                elem.attrib["sentence"])) &
                                                                            (conll_df[TOKEN_ID] == t_id)]) < 1:
                                    conll_df.loc[len(conll_df)] = {
                                        TOPIC_SUBTOPIC_DOC: topic_subtopic_doc,
                                        DOC_ID: doc_id,
                                        SENT_ID: int(elem.attrib["sentence"]),
                                        TOKEN_ID: t_id,
                                        TOKEN: elem.text,
                                        "lang": lang_key,
                                        REFERENCE: "-"
                                    }

                            except KeyError as e:
                                LOGGER.warning(f'Value with key {e} not found and will be skipped from parsing.')

                        if elem.tag == "Markables":

                            # Step 2: collect original mentions
                            for i, subelem in enumerate(elem):
                                if "SIGNAL" in subelem.tag:
                                    continue

                                mention_tokens_ids_global = [token.attrib[T_ID] for token in subelem]
                                mention_tokens_ids_global.sort(key=int)  # sort tokens by their id
                                sent_tokens = [int(token_dict[t]["id"]) for t in mention_tokens_ids_global]

                                # skip if the token is contained more than once within the same mention
                                # (i.e. ignore entries with error in meantime tokenization)
                                if len(mention_tokens_ids_global) != len(list(set(mention_tokens_ids_global))):
                                    continue

                                tokens_str = ""
                                for t in mention_tokens_ids_global:
                                    tokens_str, _, _ = append_text(tokens_str, token_dict[t][TEXT])

                                if len(mention_tokens_ids_global):
                                    sent_id = int(token_dict[mention_tokens_ids_global[0]][SENT])
                                    mention_id = f'{topic_subtopic_doc}-{sent_id}-{shortuuid.uuid(str(sent_tokens))}-{tokens_str.replace(" ", "_")}'

                                    mention_text = ""
                                    for t in mention_tokens_ids_global:
                                        mention_text, _, _ = append_text(mention_text, token_dict[t]["text"])

                                    mentions[subelem.attrib["m_id"]] = {"type": subelem.tag,
                                                                        TOKENS_STR: mention_text,
                                                                        # "sent_doc": doc,
                                                                        "source": path.split("\\")[-1].split("-")[0],
                                                                        LANGUAGE: language,
                                                                        MENTION_ID: mention_id,
                                                                        TOKENS_NUMBER: sent_tokens,
                                                                        TOKENS_TEXT: [str(token_dict[t]["text"]) for t in mention_tokens_ids_global],
                                                                        DOC_ID: doc_id,
                                                                        DOC: doc_id_full,
                                                                        SENT_ID: int(sent_id),
                                                                        SUBTOPIC: subtopic_full,
                                                                        TOPIC_SUBTOPIC_DOC: topic_subtopic_doc,
                                                                        TOPIC: topic_id_compose}
                                else:
                                    # form coreference chain
                                    # m_id points to the target

                                    if "ent_type" in subelem.attrib:
                                        mention_type_annot = meantime_types.get(subelem.attrib["ent_type"], "")
                                    elif "class" in subelem.attrib:
                                        mention_type_annot = subelem.attrib["class"]
                                    elif "type" in subelem.attrib:
                                        mention_type_annot = subelem.attrib["type"]
                                    else:
                                        mention_type_annot = ""

                                    if "instance_id" in subelem.attrib:
                                        id_ = subelem.attrib["instance_id"]
                                    else:
                                        descr = subelem.attrib["TAG_DESCRIPTOR"]
                                        id_ = ""

                                        for coref_id, coref_vals in coref_dict.items():
                                            if coref_vals["descr"] == descr and coref_vals["type"] == mention_type_annot \
                                                    and coref_vals["subtopic"] == subtopic and mention_type_annot:
                                                id_ = coref_id
                                                break

                                        if not len(id_):
                                            # LOGGER.warning(f"Document {doc_id}: {subelem.attrib} doesn\'t have attribute instance_id. It will be created")
                                            if "ent_type" in subelem.attrib:
                                                id_ = subelem.attrib["ent_type"] + shortuuid.uuid()[:17]
                                            elif "class" in subelem.attrib:
                                                id_ = subelem.attrib["class"][:3] + shortuuid.uuid()[:17]
                                            elif "type" in subelem.attrib:
                                                id_ = subelem.attrib["type"][:3] + shortuuid.uuid()[:17]
                                            else:
                                                id_ = ""

                                        if not len(id_):
                                            continue

                                        subelem.attrib["instance_id"] = id_

                                    if not len(id_):
                                        continue

                                    if id_ not in coref_dict:
                                        coref_dict[id_] = {"descr": subelem.attrib["TAG_DESCRIPTOR"],
                                                           "type": mention_type_annot,
                                                           "subtopic": subtopic}

                        if elem.tag == "Relations":
                            # Step 3: collect coreference chains
                            mentions_map = {m: False for m in list(mentions)}
                            # use only REFERS_TO
                            for i, subelem in enumerate(elem):
                                if subelem.tag != "REFERS_TO":
                                    continue

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

                                if tmp_instance_id == "None":
                                    LOGGER.warning(f"Document {doc_id}: not found target for: {str([v.attrib for v in subelem])}")
                                    continue
                                try:
                                    if "r_id" not in coref_dict[tmp_instance_id]:
                                        coref_dict[tmp_instance_id].update({
                                            "r_id": subelem.attrib["r_id"],
                                            "coref_type": subelem.tag,
                                            "mentions": {mentions[m.attrib["m_id"]][MENTION_ID]: mentions[m.attrib["m_id"]] for m in subelem if
                                                         m.tag == "source"}
                                        })
                                    else:
                                        for m in subelem:
                                            if m.tag == "source":
                                                mention_id = mentions[m.attrib["m_id"]][MENTION_ID]
                                                if mentions[m.attrib["m_id"]][MENTION_ID] in coref_dict[tmp_instance_id]["mentions"]:
                                                    LOGGER.info(f"A mention {mention_id} ({mentions[m.attrib['m_id']][TOKENS_STR]}) is already in a chain {tmp_instance_id}")
                                                    continue

                                                coref_dict[tmp_instance_id]["mentions"][mention_id] = mentions[m.attrib["m_id"]]
                                except KeyError as e:
                                    LOGGER.warning(f'Document {doc_id}: Mention with ID {str(e)} is not amoung the Markables and will be skipped.')
                                for m in subelem:
                                    mentions_map[m.attrib["m_id"]] = True
                LOGGER.info(f'The annotated mentions ({annotated_counter}) and parsed mentions ({parsed_counter}).')
        LOGGER.info(f'Parsing of MEANTIME annotation with language {language} done!')

    # Step 4: reparse the documents
    for topic_subtopic_doc in tqdm(conll_df[TOPIC_SUBTOPIC_DOC].unique(), desc="Preprocessing docs"):
        doc_df_orig = conll_df[(conll_df[TOPIC_SUBTOPIC_DOC] == topic_subtopic_doc)]
        doc_id = topic_subtopic_doc.split("/")[-1]
        lang_key = doc_df_orig["lang"].unique()[0]
        nlp = lang_paths[lang_key]["nlp"]
        # generate sentence doc with spacy
        token_id_global = 0
        for sent_id, sent_df in doc_df_orig.groupby(SENT_ID):
            sent_tokens = sent_df[TOKEN].tolist()
            sentence_str = ""
            for t in sent_tokens:
                sentence_str, _, _ = append_text(sentence_str, t)

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

    for chain_index, (chain_id, chain_vals) in enumerate(coref_dict.items()):
        if chain_vals.get("mentions") is None or chain_id == "":
            LOGGER.warning(f"Chain {chain_id}, {chain_vals} doesn\'t have any mentions and will be excluded.")
            continue

        # Step 5: match the originally tokenized mentions to the reparsed texts to collect mention attributes, e.g., POS, NER, etc
        for m_d, m in chain_vals["mentions"].items():
            annotated_counter += 1
            init_token_numbers = [int(t) for t in m[TOKENS_NUMBER]]
            tokens_text = m[TOKENS_TEXT]
            token_str = m[TOKENS_STR]
            topic_subtopic_doc = m[TOPIC_SUBTOPIC_DOC]

            sent_id = m["sent_id"]
            mention_id = f'MEANTIME_{m["doc_id"]}_{str(chain_id)}_{str(m["sent_id"])}_{shortuuid.uuid(m[TOKENS_STR])}'
            doc = topic_sent_dict[m[TOPIC_SUBTOPIC_DOC]][sent_id]
            sent_str = "".join([t.text + t.whitespace_ for t in doc])
            sent_tokens_text = [t.text for t in doc]
            start_token_id, _ = find_sub_list(tokens_text, sent_tokens_text)
            sent_df = preprocessed_df[
                (preprocessed_df[SENT_ID] == sent_id) & (preprocessed_df[TOPIC_SUBTOPIC_DOC] == topic_subtopic_doc)]

            if start_token_id == -1:
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

            # context_max_id = min(token_mention_start_id + CONTEXT_RANGE, len(doc_df))
            mention_context_text = list(doc_df.iloc[context_min_id:context_max_id + 1][TOKEN].values)
            mention_context_ids = list(
                doc_df.iloc[context_min_id:context_max_id + 1][TOKEN_ID_GLOBAL].values)
            sentence_start_id = mention_context_ids.index(sent_df[TOKEN_ID_GLOBAL].values.tolist()[0])
            mention_ner = doc[mention_head_id].ent_type_ if doc[mention_head_id].ent_type_ != "" else "O"

            try:
                mention_type = meantime_types[chain_id[:3]]
                if mention_type == "OTHER" and "EVENT" in m["type"]:
                    mention_type = m["type"]

            except KeyError:
                mention_type =  m["type"]

            mention = {COREF_CHAIN: chain_id,
                       MENTION_NER: mention_ner,
                       MENTION_HEAD_POS: doc[mention_head_id].pos_,
                       MENTION_HEAD_LEMMA: doc[mention_head_id].lemma_,
                       MENTION_HEAD: mention_head,
                       MENTION_HEAD_ID: int(mention_head_id),
                       DOC_ID: m[DOC_ID],
                       DOC: m[DOC],
                       IS_SINGLETON: len(chain_vals["mentions"]) == 1,
                       MENTION_ID: mention_id,
                       MENTION_TYPE: mention_type[:3],
                       MENTION_FULL_TYPE: mention_type,
                       SENT_ID: sent_id,
                       MENTION_CONTEXT: mention_context_text,
                       TOKENS_NUMBER_CONTEXT: tokens_number_context,
                       TOKENS_NUMBER: [int(i) for i in token_ids],
                       TOKENS_STR: m[TOKENS_STR],
                       TOKENS_TEXT: m[TOKENS_TEXT],
                       TOPIC_ID: m[TOPIC].split("_")[0],
                       TOPIC: m[TOPIC],
                       SUBTOPIC_ID: m[TOPIC_SUBTOPIC_DOC].split("/")[1],
                       SUBTOPIC: m[SUBTOPIC],
                       COREF_TYPE: IDENTITY,
                       DESCRIPTION: chain_vals["descr"],
                       CONTEXT_START_END_GLOBAL_ID: [int(context_min_id), int(context_max_id)],
                       MENTION_SENTENCE_CONTEXT_START_END_ID: [sentence_start_id, sentence_start_id + len(sent_df)],
                       MENTION_HEAD_ID_CONTEXT: int(head_id_context),
                       LANGUAGE: m[LANGUAGE],
                       CONLL_DOC_KEY: m[TOPIC_SUBTOPIC_DOC],
                       SPLIT: "tbd"
                       }
            mention = reorganize_field_order(mention)
            if "EVENT" in m["type"]:
                event_mentions.append(mention)
            else:
                entity_mentions.append(mention)
            parsed_counter += 1

    LOGGER.info("Splitting the dataset into train/val/test subsets...")
    LOGGER.info(
        f'The annotated mentions ({annotated_counter}) and parsed mentions ({parsed_counter}).')

    with open("train_val_test_split.json", "r") as file:
        train_val_test_dict = json.load(file)

    all_mentions = []
    conll_df_labels = pd.DataFrame()

    for split, topic_ids in train_val_test_dict.items():
        output_folder_split = os.path.join(OUT_PATH, split)
        if not os.path.exists(output_folder_split):
            os.mkdir(output_folder_split)

        selected_entity_mentions = []
        for mention in entity_mentions:
            if int(mention[TOPIC_ID]) in topic_ids:
                mention[SPLIT] = split
                selected_entity_mentions.append(mention)

        selected_event_mentions = []
        for mention in event_mentions:
            if int(mention[TOPIC_ID]) in topic_ids:
                mention[SPLIT] = split
                selected_event_mentions.append(mention)
        all_mentions.extend(selected_event_mentions)
        all_mentions.extend(selected_entity_mentions)

        with open(os.path.join(output_folder_split, MENTIONS_ENTITIES_JSON), "w", encoding='utf-8') as file:
            json.dump(selected_entity_mentions, file)

        with open(os.path.join(output_folder_split, MENTIONS_EVENTS_JSON), "w", encoding='utf-8') as file:
            json.dump(selected_event_mentions, file)

        conll_df_split = pd.DataFrame()
        for t_id in topic_ids:
            conll_df_split = pd.concat([conll_df_split,
                                        preprocessed_df[preprocessed_df[TOPIC_SUBTOPIC_DOC].str.contains(f'{t_id}/')]],
                                       axis=0)
        conll_df_split[SPLIT] = split
        conll_df_split_labels = make_save_conll(conll_df_split, selected_event_mentions + selected_entity_mentions,
                                                output_folder_split, return_df_only=True)
        conll_df_labels = pd.concat([conll_df_labels, conll_df_split_labels])

    # create a csv. file out of the mentions summary_df
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
        f'The annotated mentions ({annotated_counter}) and parsed mentions ({parsed_counter}).')
    LOGGER.info(f'Parsing of MEANTIME annotation done!')


if __name__ == '__main__':
    conv_files([EN])
