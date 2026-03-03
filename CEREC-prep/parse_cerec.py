import re
import shortuuid
import pandas as pd
import os
import random
from collections import Counter
from utils import *
from setup import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

source_path = os.path.join(os.getcwd(), CEREC_FOLDER_NAME)
output_path = os.path.join(os.getcwd(), OUTPUT_FOLDER_NAME)
nlp = spacy.load('en_core_web_sm')


def parse_conll():
    conll_df = pd.DataFrame()
    topic_id = "0"
    topic_name = "0_emails"
    entity_mentions = []
    doc_enumeration = -1
    random.seed(41)
    topic_sent_dict = {}
    preprocessed_df = pd.DataFrame()
    annotated_counter = 0
    parsed_counter = 0

    for split, source_file in zip(["train", "test", "val"], ["seed.conll", "cerec.validation.14.conll",
                                                              "cerec.validation.20.conll"
                                                              ]):
        LOGGER.info(f"Reading CoNLL for {split} split...")
        mentions_dict = {}
        coref_dict = {}
        conll_df_split = pd.DataFrame()
        entity_mentions_split = []

        with open(os.path.join(source_path, source_file), "r", encoding="utf-8") as file:
            conll_text = file.readlines()

        subtopic = ""
        subtopic_id = ""
        sent_id = 0
        doc_id_prev = ""
        mention_counter = 0
        token_id = 0
        mention_id_list = []
        orig_sent_id_prev = ""
        debug_count = 0

        for line_id, line in tqdm(enumerate(conll_text), total=len(conll_text)):
            # if debug_count == 10:
            #     break
            if line.startswith("#begin"):
                subtopic = re.sub("#begin document ", "", line)
                subtopic_id = shortuuid.uuid(subtopic)
                continue

            if line.startswith("#end"):
                sent_id = 0
                continue

            if line.startswith("\n"):
                sent_id += 1
                continue

            # Step 1: collect original tokenized text
            token, doc_id_orig, _, speaker, _, _, reference = line.replace("\n", "").split("\t\t")

            if speaker == "-":
                speaker = shortuuid.uuid(str(doc_enumeration))[:4]

            if speaker == "SYSTEM":
                continue

            if f'{doc_id_orig}' not in doc_id_prev:
                doc_enumeration += 1
            # if doc_id != doc_id_prev:
                sent_id = 0
                token_id = 0
                debug_count +=1
            else:
                if orig_sent_id_prev != sent_id:
                    # sent_id += 1
                    token_id = 0

            doc_id = f'{doc_id_orig}_{speaker}_{doc_enumeration}'
            topic_subtopic_doc = f'{topic_id}/{subtopic_id}/{doc_id}'

            # Step 2: collect original mentions
            # add a token to all open mentions
            for mention_id in mention_id_list:
                mentions_dict[mention_id]["words"].append((token, token_id))

            # Step 3: collect coreference chains
            for chain_value in reference.split("|"):
                chain_id = re.sub("\D+", "", chain_value)
                # continue

                chain_composed_id = f'{subtopic.split(".")[0]}_{chain_id}'

                # start of the reference brackets
                if chain_value.strip() == f'({chain_id}' or chain_value.strip() == f'({chain_id})':
                    coref_dict[chain_composed_id] = coref_dict.get(chain_composed_id, 0) + 1
                    mention_id = shortuuid.uuid(f'{chain_composed_id}_{split}')

                    mention_id_compose = f"CEREC_{mention_id}_{mention_counter}"
                    mention_counter += 1

                    # Step 2: collect original mentions
                    mentions_dict[mention_id_compose] = {
                        COREF_CHAIN: chain_composed_id,
                        # DESCRIPTION: "",
                        MENTION_ID: mention_id_compose,
                        DOC_ID: doc_id,
                        DOC: doc_id,
                        SENT_ID: int(sent_id),
                        SUBTOPIC_ID: str(subtopic_id),
                        SUBTOPIC: subtopic.split(".")[0],
                        TOPIC_ID: topic_id,
                        TOPIC: topic_name,
                        COREF_TYPE: IDENTITY,
                        CONLL_DOC_KEY: topic_subtopic_doc,
                        "words": [],
                        SPLIT: split}
                    mentions_dict[mention_id_compose]["words"].append((token, token_id))
                    annotated_counter += 1

                    if chain_value.strip() != f'({chain_id})':
                        mention_id_list.append(mention_id_compose)

                # end of the reference brackets
                elif chain_value.strip() == f'{chain_id})':
                    mention_id_base = shortuuid.uuid(f'{chain_composed_id}_{split}')
                    # there is a weird reference encoding with folded reference for the same entity for which I need a workaround
                    # try:
                    mention_id_compose = ""
                    for v in list(mention_id_list)[::-1]:
                        # stack principle
                        if mention_id_base in v:
                            mention_id_compose = v
                            break

                    if not mention_id_compose:
                        mention_id_compose = mention_id_list[-1]

                    mention_id_list.pop(mention_id_list.index(mention_id_compose))

            # Step 1: collect original tokenized text
            conll_df_split = pd.concat([conll_df_split, pd.DataFrame({
                TOPIC_SUBTOPIC_DOC: topic_subtopic_doc,
                DOC_ID: doc_id,
                SENT_ID: sent_id,
                TOKEN_ID: token_id,
                TOKEN: token.replace("NEWLINE", "//n"),
                REFERENCE: "-"
            }, index=[f'{subtopic_id}/{doc_id}/{sent_id}/{token_id}'])])
            token_id += 1
            doc_id_prev = doc_id
            orig_sent_id_prev = sent_id

        # Step 4: reparse the documents
        for topic_subtopic_doc in tqdm(conll_df_split[TOPIC_SUBTOPIC_DOC].unique(), desc="Reparsing documents"):
            doc_id = topic_subtopic_doc.split("/")[-1]
            doc_df_orig = conll_df_split[(conll_df_split[TOPIC_SUBTOPIC_DOC] == topic_subtopic_doc)]

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
                    SPLIT: split,
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
        coref_type_dict = {}
        LOGGER.info(f"Processing mentions for {split} split...")
        for mention_id, mention in tqdm(mentions_dict.items()):

            orig_token_ids = [w[1] for w in mention["words"]]
            doc_id = mention[DOC_ID]
            sent_id = mention[SENT_ID]
            subtopic_id = mention[SUBTOPIC_ID]
            markable_df = conll_df_split[(conll_df_split[TOPIC_SUBTOPIC_DOC].str.contains(f'/{subtopic_id}')) & (conll_df_split[DOC_ID] == doc_id)
                                         & (conll_df_split[SENT_ID] == sent_id) & (conll_df_split[TOKEN_ID].isin(orig_token_ids))]
            if not len(markable_df):
                continue

            # mention attributes
            init_token_numbers = [int(t) for t in list(markable_df[TOKEN_ID].values)]
            token_str = ""
            tokens_text = list(markable_df[TOKEN].values)
            for token in tokens_text:
                token_str, word_fixed, no_whitespace = append_text(token_str, token)

            # Step 5: match the originally tokenized mentions to the reparsed texts to collect mention attributes, e.g., POS, NER, etc
            topic_subtopic_doc = f'{topic_id}/{subtopic_id}/{doc_id}'
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
                        selected_token_df = sent_df[(sent_df[CHAR_ID_START] >= pos[0]) & (sent_df[CHAR_ID_START] < pos[1])]
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
                token_ids = list(range(start_token_id, start_token_id + len(best_matching_start_token[best_matching_start_token_id])))
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
            mention_ner = doc[mention_head_id].ent_type_ if doc[mention_head_id].ent_type_ != "" else "O"

            if mention[COREF_CHAIN] not in coref_type_dict:
                coref_type_dict[mention[COREF_CHAIN]] = {"mentions": [], "ner": [], "heads": []}

            coref_type_dict[mention[COREF_CHAIN]]["ner"].append(mention_ner)
            coref_type_dict[mention[COREF_CHAIN]]["mentions"].append(mention_id)

            if doc[mention_head_id].pos_ != "PRON":
                coref_type_dict[mention[COREF_CHAIN]]["heads"].append(mention_head)

            mention_context_ids = list(doc_df.iloc[context_min_id:context_max_id + 1][TOKEN_ID_GLOBAL].values)
            sentence_start_id = mention_context_ids.index(sent_df[TOKEN_ID_GLOBAL].values.tolist()[0])

            mention.update({
                MENTION_NER: mention_ner,
                MENTION_HEAD_POS: doc[mention_head_id].pos_,
                MENTION_HEAD_LEMMA: doc[mention_head_id].lemma_,
                MENTION_HEAD: doc[mention_head_id].text,
                MENTION_HEAD_ID: int(mention_head_id),
                IS_SINGLETON: coref_dict[mention[COREF_CHAIN]] == 1,
                MENTION_CONTEXT: mention_context_text,
                CONTEXT_START_END_GLOBAL_ID: [int(context_min_id), int(context_max_id)],
                MENTION_SENTENCE_CONTEXT_START_END_ID: [sentence_start_id, sentence_start_id + len(sent_df)],
                TOKENS_NUMBER_CONTEXT: tokens_number_context,
                MENTION_HEAD_ID_CONTEXT: int(head_id_context),
                TOKENS_NUMBER: token_ids,
                TOKENS_STR: token_str,
                TOKENS_TEXT: tokens_text
            })
            mention.pop("words")
            entity_mentions_split.append(mention)
            parsed_counter += 1

        for coref_chain, values in coref_type_dict.items():
            ner_list = [ner for ner in values["ner"] if ner != 'O']
            if not len(ner_list):
                mention_type = "OTHER"
            else:
                ner_dict = Counter(ner_list)
                # the values are already sorted
                mention_type = list(ner_dict)[0]

            head_dict = Counter(values["heads"])
            if len(head_dict):
                head = list(head_dict)[0]
            else:
                head = ""

            for m_id in values["mentions"]:
                # the mention gets automatically updated in the list of mentions due to the reference variables
                mentions_dict[m_id].update({
                    COREF_CHAIN: f'{mentions_dict[m_id][COREF_CHAIN]}_{mention_type[:3]}',
                    MENTION_TYPE: mention_type[:3],
                    MENTION_FULL_TYPE: mention_type,
                    DESCRIPTION: head
                })
                # reorder the fields
                m = reorganize_field_order(mentions_dict[m_id])
                mentions_dict[m_id] = m

        entity_mentions.extend(entity_mentions_split)

        save_path = os.path.join(output_path, split)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        preprocessed_df_split = preprocessed_df[preprocessed_df[SPLIT] == split]
        conll_df_labels = make_save_conll(preprocessed_df_split, entity_mentions_split, save_path, return_df_only=True)
        conll_df = pd.concat([conll_df, conll_df_labels])

        with open(os.path.join(save_path, MENTIONS_ENTITIES_JSON), "w") as file:
            json.dump(entity_mentions_split, file)

        with open(os.path.join(save_path, MENTIONS_EVENTS_JSON), "w") as file:
            json.dump([], file)

        LOGGER.info(
            f'The annotated mentions ({annotated_counter}) and parsed mentions ({parsed_counter}).')

    df_all_mentions = pd.DataFrame()
    for mention in tqdm(entity_mentions):
        df_all_mentions = pd.concat([df_all_mentions, pd.DataFrame({
            attr: str(value) if type(value) == list else value for attr, value in mention.items()
        }, index=[mention[MENTION_ID]])], axis=0)

    df_all_mentions.to_parquet(os.path.join(output_path, MENTIONS_ALL_PARQUET), engine="pyarrow")
    conll_df.to_parquet(os.path.join(output_path, DOCUMENTS_ALL_PARQUET), engine="pyarrow")

    LOGGER.info(f'Done! \nNumber of unique mentions: {len(df_all_mentions)} '
                f'\nNumber of unique chains: {len(df_all_mentions[COREF_CHAIN].unique())} ')
    LOGGER.info(
        f'The annotated mentions ({annotated_counter}) and parsed mentions ({parsed_counter}).')
    LOGGER.info(f'Parsing of CEREC is done!')


if __name__ == '__main__':
    LOGGER.info(f"Processing CEREC {source_path[-34:].split('_')[2]}.")
    parse_conll()
