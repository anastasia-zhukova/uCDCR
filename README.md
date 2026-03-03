# Piecing Together Cross-Document Coreference Resolution Datasets: Systematic Dataset Analysis and Unification

The repository contains a code that parses original formats of CDCR datasets into **uCDCR dataset** with the unified format and calculates summary values that enable comparison of the datasets.

Parsing scripts per dataset are contained in each separate folder, whereas the summary script is located in the root folder. The parsed datasets are available in this repository in the folders listed below. 

The final dataset to use: https://huggingface.co/datasets/AnZhu/uCDCR 

The paper has been accepted to LREC 2026. To cite the paper: 
```
@misc{zhukova2026piecingcrossdocumentcoreferenceresolution,
      title={Piecing Together Cross-Document Coreference Resolution Datasets: Systematic Dataset Analysis and Unification}, 
      author={Anastasia Zhukova and Terry Ruas and Jan Philip Wahle and Bela Gipp},
      year={2026},
      eprint={2603.00621},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      note={accepted to LREC 2026}
      url={https://arxiv.org/abs/2603.00621}, 
}
```

## Installation 

1) __Python 3.10 required__ 
2) Recommended to create a venv.
3) Install libraries: ```pip install -r requirements.txt```
4) Download the datasets and required libraries from spacy: ```python setup.py```
5) (To evaluate the datasets with the same-lemma baseline) Download and install [Perl](https://strawberryperl.com/). Add perl to PATH, restart your computer, and check that perl has been correctly installed. Perl is required to run the CoNLL scorer.

## Dataset information
The following datasets contain input and outful information: ECB+, ECB+METAm, NP4E. For the other datasets, please follow the READMEs. 

| Dataset                                  | Coreference target   | Parsing script                            | Public full documents | Train/val/test splits                                                                                                                 | Original license                                                         |
|:-----------------------------------------|----------------------|:------------------------------------------|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| [CD2CR](CD2CR-prep/README.md)            | entity               | ```CD2CR-prep/parse_cd2cr.py```           | yes                   | reused (original)                                                                                                                     | GNU General Public License (software)                                    |
| [CEREC](CEREC-prep/README.md)            | entity               | ```CEREC-prep/parse_cerec.py```           | yes                   | reused (original)                                                                                                                     | Apache-2.0 license                                                       | 
| [ECB+](ECBplus-prep/README.md)           | event + entity       | ```ECBplus-prep/parse_ecbplus.py```       | yes                   | reused (original)                                                                                                                     | CREATIVE COMMONS Attribution 3.0 Unported                                |
| [ECB+METAm](ECBplusMETAm-prep/README.md) | event + entity       | ```ECBplusMETAm-prep/parse_metam.py```    | yes                   | reused (original)                                                                                                                     | MIT License                                                              |
| [FCC-T](FCC-prep/README.md)              | event                | ```FCC-prep/parse_fcc.py```               | no                    | reused (from [Bugert et al. 2021](https://direct.mit.edu/coli/article/47/3/575/102774/Generalizing-Cross-Document-Event-Coreference)) | Creative Commons Attribution Share-Alike 4.0                             |
| [GVC](GVC-prep/README.md)                | event                | ```GVC-prep/parse_gvc.py```               | yes                   | reused (from [Bugert et al. 2021](https://direct.mit.edu/coli/article/47/3/575/102774/Generalizing-Cross-Document-Event-Coreference)) | Creative Commons Attribution 4.0 International License                   |
| [HyperCoref](HyperCoref-prep/README.md)  | event                | ```HyperCoref-prep/parse-hypercoref.py``` | no                    | reused (original)                                                                                                                     | Apache-2.0 license                                                       |
| [MEANTIME](MEANTIME-prep/README.md)      | event + entity       | ```MEANTIME-prep/parse_meantime.py```     | yes                   | new (didn't exist before)                                                                                                             | Creative Commons Attribution 4.0 International License                   |
| [NewsWCL50r](NewsWCL50-prep/README.md)   | event + entity       | ```NewsWCL50-prep/parse_newswcl50.py```   | no                    | reused (original)                                                                                                                     | Creative Commons Attribution-ShareAlike 4.0 International Public License |
| [NiDENT](NIdent-prep/README.md)          | entity               | ```NiDENT-prep/parse_nident.py```         | yes                   | new (didn't exist before)                                                                                                             | n/a                                                                      |
| [NP4E](NP4E-prep/README.md)              | entity               | ```NP4E-prep/parse_np4e.py```             | yes                   | new (didn't exist before)                                                                                                             | n/a                                                                      |
| [WEC-Eng](WECEng-prep/README.md)         | event                | ```WECEng-prep/parse_weceng.py```         | yes                   | reused (original)                                                                                                                     | Creative Commons Attribution-ShareAlike 3.0 Unported License             |


### Papers that used several datasets for evaluation at once: 

| Paper                                                                                                                   | Datasets                        | Project repo                                           |
|-------------------------------------------------------------------------------------------------------------------------|---------------------------------|--------------------------------------------------------|
| [Bugert et al. 2021](https://direct.mit.edu/coli/article/47/3/575/102774/Generalizing-Cross-Document-Event-Coreference) | ECB+, GVC, FCC                  | https://github.com/UKPLab/cdcr-beyond-corpus-tailored  |
| [Eirew et al. 2021](https://aclanthology.org/2021.naacl-main.198/)                                                      | ECB+, WEC-Eng                   | https://github.com/AlonEirew/cross-doc-event-coref     |
| [Bugert et al. 2021a](https://aclanthology.org/2021.emnlp-main.38/)                                                     | ECB+, GVC, FCC, HyperCoref      | https://github.com/UKPLab/emnlp2021-hypercoref-cdcr    |
| [Held et al. 2021](https://aclanthology.org/2021.emnlp-main.106)                                                        | ECB+, GVC, FCC, CD2CR           | https://github.com/Helw150/event_entity_coref_ecb_plus |
| [Ahmed et al. 2023a](https://aclanthology.org/2023.findings-acl.100/)                                                   | ECB+, GVC                       | https://github.com/ahmeshaf/lemma_ce_coref             |
| [Ahmed et al. 2024](https://aclanthology.org/2024.acl-short.27/)                                                        | ECB+, ECB+METAm                 | https://github.com/ahmeshaf/llms_coref                 |
| [Nath et al. 2024](https://aclanthology.org/2024.lrec-main.1039/)                                                       | ECB+, AIDA Phase 1 (not parsed) | https://github.com/csu-signal/multimodal-coreference   |
| [Gao et al. 2024](https://aclanthology.org/2024.lrec-main.523/)                                                         | ECB+, WEC-Eng                   | https://github.com/cooper12121/DIE-EC                  |
| [Chen at el. 2025](https://aclanthology.org/2025.acl-long.1134)                                                         | ECB+, ECB+METAm, WEC-Eng        | https://github.com/chenxinyu-nlp/CDCR                  |
 

### Train / val / test subsets
Each dataset contains several output files with mentions and fair-share of their context, i.e., approx. 200 tokens, which are split into  **train/ val/ test** folders:
2) ```entity_mentions.json```, i.e., a list of entity mentions with assigned coreference chain IDs. 
3) ```event_mentions.json```, i.e., a list of event mentions with assigned coreference chain IDs.
To enable streamlining of the data analysis of the mentions and their context, each dataset contains a combined version of all mentions in ```all_mentions.parquet```.
If the original documents were publicly released, then a dataset **main** folder also contains a ```all_documents.parquet``` with the CoNLL-style tokenized datasets. This file also provides information about the whitespaces that follow each token, which we have extracted during reparsing of the documents. 
If the CoNLL format is required, use the ```parquet_to_conll.py --dataset_name --dataset_folder``` for the file conversion. 

### CDCR topic structure
The articles are organized in the following structure: 
```
- topic
    - subtopic
        - document
```
 **Topic** contains text documents about the same topic, e.g., presidential elections.
**Subtopic** further organized the documents into _event-specific_ more narrowly related events, e.g., presidential elections in the U.S. in 2018. 
**Document** is a specific text, e.g., a news article. 

The composition of these attributes as ```topic_id/subtopic_id/doc_id``` will be used as a unique identifier within a dataset. 
To make the identifier unique across the datasets, e.g., to distinguish between topics with topic_id = 0, 
modify the key into ```dataset/topic_id/subtopic_id/doc_id```.

If a dataset contains only subtopics, but they are all related to one topic, e.g., football, then they are organized under one topic. 
If a dataset contains multiple subtopics but they do not share same topics, then, for each subtopic, separate topics are artificially created.  

### Input format

(1) ```*_mentions.json```: 
The format is adapted and extended from [WEC-Eng](https://huggingface.co/datasets/Intel/WEC-Eng) and from the mention format used by [Barhom et al. 2019](https://github.com/shanybar/event_entity_coref_ecb_plus/tree/master/data/interim/cybulska_setup). 

To extract some mentions' attributes, we parse document sentences by spaCy. To extract a mention head, we align each mention 
to the corresponding sentences in the documents and extract the head of mention as highest node in the dependency subtree.

| Field                       | Type            | Description                                                                                                                                                                                        |
|-----------------------------|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| coref_chain                 | string          | Unique identifier of a coreference chain to which this mention belongs to.                                                                                                                         |
| mention_id                  | string          | Mention ID.                                                                                                                                                                                        |
| tokens_str                  | string          | A full mention string, i.e., all consequitive chars of the mention as found in the text.                                                                                                           |
| description                 | string          | Description of a coreference chain.                                                                                                                                                                |
| coref_type                  | string          | Type of a coreference link, e.g., strict indentity.                                                                                                                                                |
| mention_type                | string          | Short form of a mention type, e.g., HUM                                                                                                                                                            |
| mention_full_type           | string          | Long form of a mention type, e.g., HUMAN_PART_PER                                                                                                                                                  |
| tokens_text                 | list of strings | A mention split into a list of tokens, text of tokens                                                                                                                                              |
| tokens_number               | list of int     | A mention split into a list of tokens, token id of these tokens (as occurred in a sentence).                                                                                                       |
| mention_head                | string          | A head of mention's phrase, e.g., Barack *Obama*                                                                                                                                                   |
| mention_head_id             | int             | Token id of the head of mention's phrase                                                                                                                                                           |
| mention_head_pos            | string          | POS tag of the head of mention's phrase                                                                                                                                                            |
| mention_head_lemma          | string          | Lemma of the head of mention's phrase                                                                                                                                                              |
| mention_head_ner            | string          | NER tag of the head of mention's phrase                                                                                                                                                            |
| sent_id                     | int             | Sentence ID                                                                                                                                                                                        |
| topic_id                    | string          | Topic ID                                                                                                                                                                                           |
| topic                       | string          | Topic ID with its description (if any)                                                                                                                                                             |
| subtopic_id                 | string          | Subtopic id (optionally with short name)                                                                                                                                                           |
| subtopic                    | string          | Subtopic ID with its description (if any)                                                                                                                                                          |
| doc_id                      | string          | Document ID                                                                                                                                                                                        |
| doc                         | string          | Document ID with its description (if any)                                                                                                                                                          |
| mention_context             | list of strings | approx. -N and +N tokens within one document before and after the mention (N=100) rounded up to the full sentences.                                                                                |
| context_start_end_global_id | list of int     | a list with [start_id, end_id] of the mention context to map the context directly to the document using ```token_ids_global``` (see below)                                                         |
| mention_sentence_start_end  | list of int     | a list of indeces that indicate a start and end index of a sentence where mention is located. Use case: ```sent = mention_context[mention_sentence_start_end[0], mention_sentence_start_end[1]]``` |
| tokens_number_context       | list of int     | Positioning of the mention's tokens within the context.                                                                                                                                            |
| mention_head_id_context     | int             | id of the mention head within the context window.                                                                                                                                                  |
| is_singleton                | bool            | A marker if a mention is a singleton or not.                                                                                                                                                       |
| language                    | string          | Optional. A language of the mention. If not provided, the default value will be considered english.                                                                                                |
| conll_doc_key               | string          | a compositional key for **one-to-one mapping documents between ```all_documents.parquet``` and .json files**.                                                                                      |

Example of one mention: 
```json
[
  {
    "coref_chain": "LOC27735327659249054", 
    "mention_id": "MEANTIME_english113219_LOC27735327659249054_2_LkPLFNuy65zEuPceVBrL4r", 
    "tokens_str": "the United States", 
    "description": "United States of America ", 
    "coref_type": "IDENTITY", 
    "mention_type": "LOC", 
    "mention_full_type": "LOCATION", 
    "tokens_text": ["the", "United", "States"], 
    "tokens_number": [9, 10, 11], 
    "mention_head": "States", 
    "mention_head_id": 11, 
    "mention_head_pos": "PROPN", 
    "mention_head_lemma": "States", 
    "mention_ner": "GPE", 
    "sent_id": 2, 
    "topic": "3_corpus_stock", 
    "topic_id": "3", 
    "subtopic_id": "113219", 
    "subtopic": "113219_Stock_markets_worldwide_fall_dramatically", 
    "doc_id": "english113219", 
    "doc": "english113219_Stock_markets_worldwide_fall_dramatically", 
    "mention_context": ["Stock", "markets", "worldwide", "fall", "dramatically", "September", "17", ",", "2008", "Stock", "markets", "around", "the", "world", ",", "particularly", "those", "in", "the", "United", "States", ",", "have", "fallen", "dramatically", "today", ".", "This", "is", "due", "to", "the", "ongoing", "events", "in", "the", "financial", "world", ",", "including", "the", "bailout", "of", "large", "insurance", "firm", "AIG", "by", "the", "US", "Federal", "Reserve", ".", "The", "primary", "UK", "index", ",", "the", "FTSE", "100", ",", "dropped", "in", "value", "by", "2.36", "%", ",", "which", "is", "118.40", "points", ",", "to", "below", "the", "5000", "mark", "at", "4907.20", ".", "The", "Dow", "Jones", "was", "down", "2.62", "%", "at", "16:08", "UTC", ",", "a", "slight", "increase", "from", "earlier", "today", ".", "The", "Dow", "Jones", "currently", "has", "a", "value", "of", "10769.00", "points", ".", "The", "Nasdaq", "index", "has", "fallen", "by", "3.16", "%", "to", "2138.14", ",", "while", "the", "Dax", "was", "1.75", "%", "lower", "than", "the", "start", "of", "the", "day", "as", "of", "16:08", "UTC", "."], 
    "context_start_end_global_id": [0, 139], 
    "mention_sentence_start_end": [0, 26],
    "tokens_number_context": [18, 19, 20], 
    "mention_head_id_context": 20, 
    "is_singleton": false, 
    "conll_doc_key": "3/113219/english113219"
  }
]
```

(2) ```all_documents.parquet```:
When available (for the previously publicly available full texts), the file contains the following columns: 

| Field                   | Type   | Description                                                                                                                                                                                                                                       |
|-------------------------|--------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| split                   | string | Dataset split                                                                                                                                                                                                                                     |
| topic/subtopic_name/doc | string | A unique document key within each dataset. To match a mention from ```*_mentions.json```, use "conll_doc_key" field and map it to "topic/subtopic_name/doc"                                                                                       |
| sent_id                 | int    | A sentence ID. Matches same attribute in ```*_mentions.json```                                                                                                                                                                                    |
| token_id                | int    | A token ID within each sentence. Matches "tokens_number" attribute in ```*_mentions.json```                                                                                                                                                       |
| token                   | sting  | Token text. Matches "tokens_text" attribute in ```*_mentions.json```                                                                                                                                                                              |
| token_id_global         | int    | A token ID global within each document. To match a mention's context from ```*_mentions.json```, us "context_start_end_global_id" attribute, that indicated the first and last token IDs withing the document global IDs.                         |
| char_id_start           | int    | Indicated the start of the token when a sentence is represented as one string                                                                                                                                                                     |
| whitespace_after        | bool   | When concatenating the tokens into one document, indicates if a whitespace needs to come after this token or not.                                                                                                                                 |
| reference               | string | A reference in the CoNLL format. "(1" means that the token is a start of the mention that belongs to a coref chain 1, whereas "1)" indicated the end of mention. "(1)" means that a mention consists of one token. The references can be nested.  |


## Run dataset analysis 
To run the code for the data analysis and evaluation with the same-head-lemma baseline, please execute ```create_summary.py```.
The main body in the code right configured to read the datasets from the folders directly after parsing. If your location of the datasets in uCDCR is different, 
you will need to provide correct paths to the dataset folders. 

All results reported in the paper are located in ```\summary``` folder. 

## Simple use case: cross-encoder / binary classification CDCR model
To train a simple binary classification mentions, one requires only ```entity_mentions.json``` and ```event_mentions.json``` files. 
Each file contains a list of mentions. To encode a mention, you need to use the following attributes: 
1) ```mention_context``` with a list of tokens within which a mention occurs
2) ```tokens_number_context``` with a list of indexed where a mention occurs in the ```mention_context```, which are needed to position the mention 
3) ```coref_chain``` that indicates if two mentions are coreferencial if the value is identical between two mentions

Similar to [Eirew et al. 2021](https://aclanthology.org/2021.naacl-main.198/), a pair of mentions can be encoded within their contexts and a coreference chain sets a training objective. 
