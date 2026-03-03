import os
import json
import sys
import string
import spacy
import re
import pandas as pd
import shortuuid
from utils import *
from tqdm import tqdm
from setup import *
from logger import LOGGER
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

FCC_PARSING_FOLDER = os.path.join(os.getcwd())
source_path = os.path.join(FCC_PARSING_FOLDER, FCC_FOLDER_NAME)
nlp = spacy.load('en_core_web_sm')
LOGGER.info("Spacy model loaded.")


def conv_files():
    topic_name = "0_football_matches"
    topic_id = "0"
    documents_df = pd.DataFrame()
    # conll_df_fcc = pd.DataFrame()
    conll_df_fcc_t = pd.DataFrame()
    all_mentions_dict = {"event": [], "entity": [], "event_sentence": []}
    
    other_event_counter = 0
    topic_sent_dict = {}
    preprocessed_df = pd.DataFrame()

    for split in ["dev", "test", "train"]:
        if split == "dev":
            split_to_use = "val"
        else:
            split_to_use = split
        LOGGER.info(f"Reading {split} split...")
        all_mentions_dict_local = {"event": [], "entity": [], "event_sentence": []}
        documents_df = pd.concat([documents_df,
                                  pd.read_csv(os.path.join(source_path, "2020-10-05_FCC_cleaned", split, "documents.csv"), index_col=[0]).fillna("")])
        # if a document is not assigned to any seminal event, create a new event
        for index, row in documents_df.iterrows():
            if not row["seminal-event"]:
                documents_df.loc[index, "seminal-event"] = f"other_seminal_event-{other_event_counter}"
                other_event_counter += 1
        documents_df["subtopic-id"] = [shortuuid.uuid(v) for v in documents_df["seminal-event"].values]

        # Step 1: collect original tokenized text
        tokens_df = pd.read_csv(os.path.join(source_path, "2020-10-05_FCC_cleaned", split, "tokens.csv"))

        conll_df_local = pd.DataFrame()
        for index, row in tqdm(tokens_df.iterrows(), total=tokens_df.shape[0]):
            topic_subtopic_doc = f"{topic_id}/{documents_df.loc[row['doc-id'], 'subtopic-id']}/{row['doc-id']}"
            conll_df_local = pd.concat([conll_df_local, pd.DataFrame({
                TOPIC_SUBTOPIC_DOC: topic_subtopic_doc,
                DOC_ID: row["doc-id"],
                SENT_ID: int(row["sentence-idx"]),
                TOKEN_ID: int(row["token-idx"]),
                TOKEN: row["token"],
                REFERENCE: "-"
            }, index=[f'{row["doc-id"]}/{row["sentence-idx"]}/{row["token-idx"]}'])])

        # conll_df_local.fillna("", inplace=True)
        # Step 4: reparse the documents

        for topic_subtopic_doc in tqdm(conll_df_local[TOPIC_SUBTOPIC_DOC].unique(), desc="Reparsing documents"):

            doc_df_orig = conll_df_local[(conll_df_local[TOPIC_SUBTOPIC_DOC] == topic_subtopic_doc)]
            doc_id = topic_subtopic_doc.split("/")[-1]
            # generate sentence doc with spacy
            token_id_global = 0
            for sent_id, sent_df in doc_df_orig.groupby(SENT_ID):
                sent_tokens = sent_df[TOKEN].tolist()
                sentence_str = ""

                for t in sent_tokens:
                    if type(t) != str:
                        # print("None found!")
                        continue
                    sentence_str, _, _ = append_text(sentence_str, t)

                sentence_str = correct_whitespaces(sentence_str)
                # sentence_str = correct_whitespaces(correct_whitespaces(" ".join(sent_tokens)))
                doc = nlp(sentence_str)
                tokens_id_global = list(range(token_id_global, token_id_global + len(doc)))
                preprocessed_df = pd.concat([preprocessed_df, pd.DataFrame({
                    SPLIT: split_to_use,
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

        # Step 2: collect original mentions
        mentions_sent_level_df_local = pd.read_csv(
            os.path.join(source_path, "2020-10-05_FCC_cleaned", split, "mentions_cross_subtopic.csv"))
        mentions_sent_level_df_local["chain-id"] = [shortuuid.uuid(mention_row["event"]) for index, mention_row in
                                         mentions_sent_level_df_local.iterrows()]

        event_mentions_df = pd.read_csv(os.path.join(source_path, "2020-10-05_FCC-T", split, "with_stacked_actions",
                                                     "cross_subtopic_mentions_action.csv"))
        event_mentions_df["chain-id"] = [shortuuid.uuid(mention_row["event"]) for index, mention_row in event_mentions_df.iterrows()]
        event_mentions_df["event"] = [re.sub("other_event", "other_event-" + documents_df.loc[row["doc-id"], "collection"], row["event"])
                                      if "other_event" in row["event"] else row["event"]
                                      for index, row in event_mentions_df.iterrows()]

        semantic_roles_df = pd.read_csv(
            os.path.join(source_path, "2020-10-05_FCC-T", split, "with_stacked_actions", "cross_subtopic_semantic_roles.csv"))

        entity_mentions_df_init = pd.DataFrame()
        for file_name in ["cross_subtopic_mentions_location.csv", "cross_subtopic_mentions_participants.csv",
                          "cross_subtopic_mentions_time.csv"]:
            entity_mentions_df_local = pd.read_csv(
                os.path.join(source_path, "2020-10-05_FCC-T", split, "with_stacked_actions", file_name))
            entity_mentions_df_init = pd.concat([entity_mentions_df_init, entity_mentions_df_local])

        entity_mentions_df = pd.merge(entity_mentions_df_init, semantic_roles_df.rename(columns={"mention-id": "event-mention-id"}),
                                      how="left", left_on=["doc-id", "mention-id"],
                                      right_on=["doc-id", "component-mention-id"])
        entity_mentions_df = pd.merge(entity_mentions_df, event_mentions_df[["doc-id", "mention-id", "event"]].rename(columns={"mention-id": "event-mention-id"}),
                                      how="left", left_on=["doc-id", "event-mention-id"],
                                      right_on=["doc-id", "event-mention-id"])
        entity_mentions_df["chain-id"] = [shortuuid.uuid(mention_row["event"]) for index, mention_row in
                                         entity_mentions_df.iterrows()]


        for mention_annot_type, mention_init_df in zip(["event", "entity", "event_sentence"],
                                                            [event_mentions_df, entity_mentions_df, mentions_sent_level_df_local]):
            LOGGER.info(f"Parsing {mention_annot_type} mentions...")

            for index, mention_row in tqdm(mention_init_df.iterrows(), total=mention_init_df.shape[0]):

                doc_id = mention_row["doc-id"]
                # create a unique ID for each mention's occurrence
                mention_id_global = f'FCC-T_{mention_row["doc-id"]}/{mention_row["mention-id"]}/{shortuuid.uuid()[:4]}'
                subtopic_id = documents_df.loc[doc_id, "subtopic-id"]
                topic_subtopic_doc = f'{topic_id}/{subtopic_id}/{doc_id}'
                sent_id = mention_row["sentence-idx"]

                if "token-idx-from" in mention_init_df.columns:
                    # tokenized version
                    markable_df = conll_df_local[(conll_df_local[DOC_ID] == mention_row["doc-id"])
                                                 & (conll_df_local[SENT_ID] == mention_row["sentence-idx"])
                                                 & (conll_df_local[TOKEN_ID] >= mention_row["token-idx-from"])
                                                 & (conll_df_local[TOKEN_ID] < mention_row["token-idx-to"])]
                else:
                    continue

                # Step 5: match the originally tokenized mentions to the reparsed texts to collect mention attributes, e.g., POS, NER, etc
                token_str = ""
                tokens_text = list(markable_df[TOKEN].values)
                init_token_numbers = list(markable_df[TOKEN_ID].values)
                for token in tokens_text:
                    token_str, word_fixed, no_whitespace = append_text(token_str, token)

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

                mention_type = mention_row["mention-type"] if "mention-type" in mention_init_df.columns else "EVENT"
                srl_type = mention_row["mention-type-coarse"] if "mention-type-coarse" in mention_init_df.columns else ""
                chain_id = mention_type[:3] + mention_row["chain-id"] if "event" in mention_annot_type \
                        else f'{mention_type[:3] + mention_row["chain-id"]}_{srl_type}_{shortuuid.uuid(doc[mention_head_id].lemma_)[:4]}'
                is_singleton = len(mention_init_df[mention_init_df["chain-id"] == mention_row["chain-id"]]) == 1 if "event" in mention_init_df.columns else True

                # Step 6: form a mention with all attributes
                mention = {COREF_CHAIN: chain_id,
                           MENTION_NER: mention_ner,
                           MENTION_HEAD_POS:  doc[mention_head_id].pos_,
                           MENTION_HEAD_LEMMA: doc[mention_head_id].lemma_,
                           MENTION_HEAD: doc[mention_head_id].text,
                           MENTION_HEAD_ID: int(mention_head_id),
                           DOC_ID: doc_id,
                           DOC: doc_id,
                           IS_SINGLETON: is_singleton,
                           MENTION_ID: mention_id_global,
                           MENTION_TYPE: mention_type[:3],
                           MENTION_FULL_TYPE: mention_type,
                           SENT_ID: sent_id,
                           MENTION_CONTEXT: mention_context_text,
                           TOKENS_NUMBER_CONTEXT: tokens_number_context,
                           TOKENS_NUMBER: [int(t) for t in token_ids],
                           CONTEXT_START_END_GLOBAL_ID: [int(context_min_id), int(context_max_id)],
                           MENTION_SENTENCE_CONTEXT_START_END_ID: [context_sent_id_start,
                                                                   context_sent_id_start + len(sent_df)],
                           MENTION_HEAD_ID_CONTEXT: int(head_id_context),
                           TOKENS_STR: token_str,
                           TOKENS_TEXT: tokens_text,
                           TOPIC_ID: topic_id,
                           TOPIC: topic_name,
                           SUBTOPIC_ID: subtopic_id,
                           SUBTOPIC: documents_df.loc[doc_id, "seminal-event"],
                           COREF_TYPE: IDENTITY if not srl_type else srl_type,
                           DESCRIPTION: mention_row["event"] if "event" in mention_init_df.columns
                                                    else f'{mention_row["event"]}_{srl_type}_{mention_head.lemma_}',
                           CONLL_DOC_KEY: topic_subtopic_doc,
                           SPLIT:  split_to_use,
                           }
                mention = reorganize_field_order(mention)
                all_mentions_dict_local[mention_annot_type].append(mention)
                all_mentions_dict[mention_annot_type].append(mention)

        # conll_df_local.reset_index(drop=True, inplace=True)
        save_folder_fcc_t = os.path.join(FCC_PARSING_FOLDER, f'{OUTPUT_FOLDER_NAME}', split_to_use)
        if not os.path.exists(os.path.join(FCC_PARSING_FOLDER, f'{OUTPUT_FOLDER_NAME}')):
            os.mkdir(os.path.join(FCC_PARSING_FOLDER, f'{OUTPUT_FOLDER_NAME}'))

        if not os.path.exists(save_folder_fcc_t):
            os.mkdir(save_folder_fcc_t)

        # save_folder_fcc = os.path.join(FCC_PARSING_FOLDER, f'{OUTPUT_FOLDER_NAME}_FCC', split_to_use)
        # if not os.path.exists(os.path.join(FCC_PARSING_FOLDER, f'{OUTPUT_FOLDER_NAME}_FCC')):
        #     os.mkdir(os.path.join(FCC_PARSING_FOLDER, f'{OUTPUT_FOLDER_NAME}_FCC'))

        # if not os.path.exists(save_folder_fcc):
        #     os.mkdir(save_folder_fcc)

        conll_df_local = preprocessed_df[preprocessed_df[SPLIT] == split_to_use]
        if not len(conll_df_local) or not len(all_mentions_dict_local["event"]):
            LOGGER.info(f"No data in {split_to_use}")
            continue

        conll_df_fcc_t = pd.concat([conll_df_fcc_t, make_save_conll(conll_df=conll_df_local,
                                        mentions=all_mentions_dict_local["event"],
                                        # mentions=all_mentions_dict_local["event"] + all_mentions_dict_local["entity"],
                                        output_folder=save_folder_fcc_t)])


        # conll_df_fcc = pd.concat([conll_df_fcc, make_save_conll(conll_df=conll_df_local,
        #                                 mentions=all_mentions_dict_local["event_sentence"],
        #                                 output_folder=save_folder_fcc_t, return_df_only=True)])

        # since there are no entity coreference chains, save attributes separately
        # with open(os.path.join(save_folder_fcc_t, "entity_mentions_attr.json"), "w", encoding='utf-8') as file:
        #     json.dump(all_mentions_dict_local["entity"], file)

        with open(os.path.join(save_folder_fcc_t, MENTIONS_ENTITIES_JSON), "w", encoding='utf-8') as file:
            json.dump([], file)

        with open(os.path.join(save_folder_fcc_t, MENTIONS_EVENTS_JSON), "w", encoding='utf-8') as file:
            json.dump(all_mentions_dict_local["event"], file)

        # with open(os.path.join(save_folder_fcc, MENTIONS_EVENTS_JSON), "w", encoding='utf-8') as file:
        #     json.dump(all_mentions_dict_local["event_sentence"], file)
        #
        # with open(os.path.join(save_folder_fcc, MENTIONS_ENTITIES_JSON), "w", encoding='utf-8') as file:
        #     json.dump([], file)

    # conll_df_fcc.reset_index(drop=True, inplace=True)
    save_folder_fcc_t = os.path.join(FCC_PARSING_FOLDER, f'{OUTPUT_FOLDER_NAME}')
    # save_folder_fcc = os.path.join(FCC_PARSING_FOLDER, f'{OUTPUT_FOLDER_NAME}_FCC')

    df_all_mentions = pd.DataFrame()
    # for mention in all_mentions_dict["event"] + all_mentions_dict["entity"]:
    for mention in all_mentions_dict["event"]:
        df_all_mentions = pd.concat([df_all_mentions, pd.DataFrame({
            attr: str(value) if type(value) == list else value for attr, value in mention.items()
        }, index=[mention[MENTION_ID]])], axis=0)

    df_all_mentions.to_parquet(os.path.join(save_folder_fcc_t, MENTIONS_ALL_PARQUET), engine="pyarrow")
    conll_df_fcc_t.to_parquet(os.path.join(save_folder_fcc_t, DOCUMENTS_ALL_PARQUET), engine="pyarrow")

    # df_all_mentions_sent = pd.DataFrame()
    # for mention in all_mentions_dict["event_sentence"]:
    #     df_all_mentions_sent = pd.concat([df_all_mentions_sent, pd.DataFrame({
    #         attr: str(value) if type(value) == list else value for attr, value in mention.items()
    #     }, index=[mention[MENTION_ID]])], axis=0)
    # df_all_mentions_sent.to_csv(os.path.join(save_folder_fcc, MENTIONS_ALL_CSV))

    # LOGGER.info(f'Parsing of FCC is done!')
    # LOGGER.info(
    #     f'\nNumber of unique mentions in FCC: {len(df_all_mentions_sent)} '
    #     f'\nNumber of unique event chains: {len(set(df_all_mentions_sent[COREF_CHAIN].values))} ')

    LOGGER.info(
        f'\nNumber of unique mentions: {len(df_all_mentions)} '
        f'\nNumber of unique chains: {len(df_all_mentions[COREF_CHAIN].unique())} ')

    LOGGER.info(f'Parsing of FCC-T is done!')


if __name__ == '__main__':
    LOGGER.info(f"Processing FCC corpus...")
    conv_files()
