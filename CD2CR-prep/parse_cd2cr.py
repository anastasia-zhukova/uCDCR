import pandas as pd
import json
import os
import shortuuid
import string
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils import *
from setup import *

source_folder = os.path.join(os.getcwd(), CD2CR_FOLDER_NAME)
output_folder = os.path.join(os.getcwd(), OUTPUT_FOLDER_NAME)
nlp = spacy.load('en_core_web_sm')


def conv_files():
    conll_df = pd.DataFrame()
    entity_mentions = []
    topic_id = "0"
    topic_name = "0_science_technology"
    subtopic_id_global = 0
    subtopic_map_dict = {}
    annotated_counter = 0
    parsed_counter = 0

    chain_dict = {}
    preprocessed_df = pd.DataFrame()
    topic_sent_dict = {}

    with open(os.path.join(source_folder, "sci_papers.json"), "r") as file:
        sci_papers = json.load(file)

    with open(os.path.join(source_folder, "news_urls.json"), "r") as file:
        news_urls = json.load(file)

    for split, conll_file, mentions_file in zip(["val", "train", "test"],
                                                ["dev.conll", "train.conll", "test.conll"],
                                                ["dev_entities.json", "train_entities.json",  "test_entities.json"]):
        if split not in subtopic_map_dict:
            subtopic_map_dict[split] = {}

        # Step 1: collect original tokenized text
        # read conll file
        LOGGER.info(f'Reading CoNLL files of {split} split...')
        conll_split_df = pd.DataFrame()
        entity_mentions_split = []
        with open(os.path.join(source_folder, conll_file), "r", encoding="utf-8") as file:
            conll_txt = file.readlines()

        sent_id_prev = -1
        doc_id_prev = -1
        token_id = 0

        for line in tqdm(conll_txt):
            if line.startswith("#"):
                continue

            subtopic_id, _, doc_id, sent_id, token_id_global, token, is_headline, chain_value = line.strip().split("\t")
            if subtopic_id not in subtopic_map_dict[split]:
                subtopic_map_dict[split][subtopic_id] = subtopic_id_global
                subtopic_id_global += 1

            # Step 3: collect coreference chains, i..e, information about the chain ids to later decide which mentions are singletons
            chain_id = re.sub("\D+", "", chain_value)

            if chain_value.strip() == f'({chain_id}' or chain_value.strip() == f'({chain_id})':
                chain_dict[chain_id] = chain_dict.get(chain_id, 0) + 1

            if sent_id != sent_id_prev or doc_id != doc_id_prev:
                token_id = 0

            conll_split_df = pd.concat([conll_split_df, pd.DataFrame({
                TOPIC_SUBTOPIC_DOC: f"{topic_id}/{subtopic_map_dict[split][subtopic_id]}/{doc_id}",
                DOC_ID: doc_id,
                SENT_ID: int(sent_id),
                TOKEN_ID: int(token_id),
                TOKEN_ID_GLOBAL: int(token_id_global),
                TOKEN: token,
                # REFERENCE: chain_value
                REFERENCE: '-' #we will need to reassign the references because not mentions are in conll
            }, index=[f'{doc_id}/{token_id_global}'])])

            token_id += 1
            sent_id_prev = sent_id
            doc_id_prev = doc_id

        # Step 4: reparse the documents
        for topic_subtopic_doc in tqdm(conll_split_df[TOPIC_SUBTOPIC_DOC].unique(), desc="Preprocessing docs"):
            doc_df_orig = conll_split_df[(conll_split_df[TOPIC_SUBTOPIC_DOC] == topic_subtopic_doc)]
            doc_id = topic_subtopic_doc.split("/")[-1]
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
                    SPLIT: split,
                    TOPIC_SUBTOPIC_DOC: topic_subtopic_doc,
                    DOC_ID: doc_id,
                    SENT_ID: int(sent_id),
                    TOKEN_ID: list(range(len(doc))),
                    TOKEN: [t.text for t in doc],
                    TOKEN_ID_GLOBAL: tokens_id_global,
                    CHAR_ID_START: [t.idx for t in doc],
                    WHITESPACE_AFTER: [bool(len(t.whitespace_)) for t in doc],
                    REFERENCE: '-'
                }, index=[f'{topic_subtopic_doc}_{i}' for i in tokens_id_global])])

                if topic_subtopic_doc not in topic_sent_dict:
                    topic_sent_dict[topic_subtopic_doc] = {}
                topic_sent_dict[topic_subtopic_doc][sent_id] = doc
                token_id_global += len(doc)

        # Step 2: collect original mentions
         # read mentions file
        coref_info_dict = {}
        LOGGER.info(f'Reading mentions of {split} split...')
        with open(os.path.join(source_folder, mentions_file), "r") as file:
            entity_mention_orig = json.load(file)

        for mention_orig in tqdm(entity_mention_orig):
            mention_id = "CD2CR_" + shortuuid.uuid(f'{mention_orig[DOC_ID]}/{mention_orig["tokens"]}')
            doc_id = mention_orig[DOC_ID]
            sent_id = mention_orig["sentence_id"]
            token_ids_global = mention_orig["tokens_ids"]
            markable_df = conll_split_df[(conll_split_df[DOC_ID] == doc_id) & (conll_split_df[SENT_ID] == sent_id) &
                                         (conll_split_df[TOKEN_ID_GLOBAL].isin(token_ids_global))]
            annotated_counter += 1

            if not len(markable_df):
                LOGGER.warning(f'Mention {mention_orig} is not found and skipped.' )
                continue

            if len(token_ids_global) != len(markable_df):
                token_ids_global = list(markable_df[TOKEN_ID_GLOBAL].values)

            # mention attributes
            init_token_numbers = [int(t) for t in markable_df[TOKEN_ID].values]
            token_str = ""
            tokens_text = list(markable_df[TOKEN].values)
            for token in tokens_text:
                token_str, word_fixed, no_whitespace = append_text(token_str, token)

            # Step 5: match the originally tokenized mentions to the reparsed texts to collect mention attributes, e.g., POS, NER, etc
            subtopic_id = str(subtopic_map_dict[split][str(mention_orig["topic"])])
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
            mention_type = mention_ner if mention_ner != "O" else "OTHER"
            subtopic_id = str(subtopic_map_dict[split][str(mention_orig["topic"])])
            mention_context_ids = list(doc_df.iloc[context_min_id:context_max_id + 1][TOKEN_ID_GLOBAL].values)
            sentence_start_id = mention_context_ids.index(sent_df[TOKEN_ID_GLOBAL].values.tolist()[0])

            doc_id_key = doc_id.split("_")[-1]
            if doc_id_key in sci_papers:
                doc_name = sci_papers[doc_id_key]["doi"]
            elif doc_id_key in news_urls:
                doc_name = news_urls[doc_id_key]["url"].split("/")[-1]
            else:
                doc_name = doc_id

            # Step 3: collect coreference chains
            coref_id = str(mention_orig["cluster_id"])
            if coref_id not in coref_info_dict:
                coref_info_dict[coref_id] = set()
            # required to create chain descriptions later
            coref_info_dict[coref_id].add(mention_head)

            # Step 6: form a mention with all attributes
            mention = {COREF_CHAIN: coref_id,
                       MENTION_ID: mention_id,
                       TOKENS_STR: mention_orig["tokens"],
                       DESCRIPTION: "",  #will be generated below
                       COREF_TYPE: IDENTITY,
                       MENTION_TYPE: mention_type[:3],
                       MENTION_FULL_TYPE: mention_type,
                       TOKENS_TEXT: tokens_text,
                       TOKENS_NUMBER: [int(i) for i in token_ids],
                       MENTION_HEAD: mention_head,
                       MENTION_HEAD_ID: int(mention_head_id),
                       MENTION_HEAD_POS: doc[mention_head_id].pos_,
                       MENTION_HEAD_LEMMA: doc[mention_head_id].lemma_,
                       MENTION_NER: mention_ner,
                       SENT_ID: int(sent_id),
                       TOPIC_ID: topic_id,
                       TOPIC: topic_name,
                       SUBTOPIC_ID: subtopic_id,
                       SUBTOPIC: subtopic_id,
                       DOC_ID: doc_id,
                       DOC: doc_name,
                       MENTION_CONTEXT: mention_context_text,
                       CONTEXT_START_END_GLOBAL_ID: [int(context_min_id), int(context_max_id)],
                       MENTION_SENTENCE_CONTEXT_START_END_ID: [sentence_start_id, sentence_start_id + len(sent_df)],
                       TOKENS_NUMBER_CONTEXT: tokens_number_context,
                       MENTION_HEAD_ID_CONTEXT: int(head_id_context),
                       IS_SINGLETON: coref_id not in chain_dict,
                       CONLL_DOC_KEY: topic_subtopic_doc,
                       SPLIT: split
                       }
            mention = reorganize_field_order(mention)
            entity_mentions_split.append(mention)
            parsed_counter += 1

        #add chain descriptions
        for mention in entity_mentions_split:
            mention[DESCRIPTION] = "_".join(coref_info_dict[mention[COREF_CHAIN]])

        save_path = os.path.join(output_folder, split)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        with open(os.path.join(save_path, MENTIONS_EVENTS_JSON), "w") as file:
            json.dump([], file)

        with open(os.path.join(save_path, MENTIONS_ENTITIES_JSON), "w") as file:
            json.dump(entity_mentions_split, file)

        preprocessed_df_split = preprocessed_df[preprocessed_df[SPLIT] == split]
        conll_split_df_labeled = make_save_conll(preprocessed_df_split, entity_mentions_split, save_path, return_df_only=True)
        conll_df = pd.concat([conll_df, conll_split_df_labeled])
        entity_mentions.extend(entity_mentions_split)
        LOGGER.info(
            f'The annotated mentions ({annotated_counter}) and parsed mentions ({parsed_counter}).')

    df_all_mentions = pd.DataFrame()
    for mention in tqdm(entity_mentions):
        df_all_mentions = pd.concat([df_all_mentions, pd.DataFrame({
            attr: str(value) if type(value) == list else value for attr, value in mention.items()
        }, index=[mention[MENTION_ID]])], axis=0)

    df_all_mentions.to_parquet(os.path.join(output_folder, MENTIONS_ALL_PARQUET), engine="pyarrow")

    conll_df.to_parquet(os.path.join(output_folder, DOCUMENTS_ALL_PARQUET), engine="pyarrow")

    LOGGER.info(
        f'\nNumber of unique mentions: {len(entity_mentions)} '
        f'\nNumber of unique chains: {len(df_all_mentions[COREF_CHAIN].unique())} ')
    LOGGER.info(
        f'The annotated mentions ({annotated_counter}) and parsed mentions ({parsed_counter}).')
    LOGGER.info(f'Parsing of CD2CR is done!')


if __name__ == '__main__':
    LOGGER.info(f"Processing CD2CR {source_folder[-34:].split('_')[2]}.")
    conv_files()
