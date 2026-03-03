---
license: cc-by-sa-4.0
task_categories:
- text-classification
language:
- en
pretty_name: ucdcr
size_categories:
- 10K<n<100K
---
# Dataset Card for uCDCR 

**uCDCR** (unified Cross Document Coreference Resolution) dataset provides a unified format across 12 English CDCR datasets that aim to streamline model training and data analysis within CDCR by avoiding tedious dataset parsing from a diverse formats in which these datasets were released. 

[//]: # (This dataset card aims to be a base template for new datasets. It has been generated using [this raw template]&#40;https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/datasetcard_template.md?plain=1&#41;.)

## Dataset Details

### Dataset Description

Work in Natural Language Understanding increasingly relies on the ability to identify and track entities and events across large, heterogeneous text collections. This task, known as cross-document coreference resolution (CDCR), has a wide range of downstream applications, including multi-document summarization, information retrieval, and knowledge base population.
Research in this area remains fragmented due to heterogeneous datasets, varying annotation standards, and the predominance of event-centric coreference resolution (ECR) approaches. To address these challenges, we introduce **uCDCR**, a unified dataset that consolidates diverse publicly available CDCR corpora across various domains into a consistent format with standardized metrics and evaluation protocols. uCDCR incorporates both entity and event coreference, corrects known inconsistencies, and enriches datasets with missing attributes to facilitate reproducible research. The further analysis compares the datasets using lexical diversity and ambiguity metrics, discusses the annotation rules and principles that lead to high lexical diversity, and examines how these metrics influence performance on the same-head lemma baseline.
We further provide a comprehensive characterization of uCDCR using measures of lexical diversity and ambiguity, along with baseline performance analysis, thereby establishing a cohesive framework for fair, interpretable, and cross-dataset evaluation in CDCR.

- **Curated by:** Anastasia Zhukova
- **Language(s) (NLP):** English
- **License:** BB-by-SA-4.0

### Dataset Sources

- **Repository:** https://github.com/anastasia-zhukova/uCDCR
- **Paper:** TBD



## Dataset Structure

Each folder contains the parsed original dataset with two ```*_mentions.json``` files located in train/val/test folders, i.e., for event and entities. Each dataset also has a concatenated version of these mention in one parquet file per dataset and, if previously publicly released, a CoNLL-like parquet file of the tokenized documents. 

```
val
│   entity_mentions.json
│   event_mentions.json   
|
test
│   entity_mentions.json
│   event_mentions.json   
|
val
│   entity_mentions.json
│   event_mentions.json   
│
*all_documents.parquet
all_mentions.parquet
```


### Dataset format
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


## Uses


### Direct Use

To train a simple binary classification mentions, one requires only ```entity_mentions.json``` and ```event_mentions.json``` files. 
Each file contains a list of mentions. To encode a mention, you need to use the following attributes: 
1) ```mention_context``` with a list of tokens within which a mention occurs
2) ```tokens_number_context``` with a list of indexed where a mention occurs in the ```mention_context```, which are needed to position the mention 
3) ```coref_chain``` that indicates if two mentions are coreferencial if the value is identical between two mentions

Similar to [Eirew et al. 2021](https://github.com/AlonEirew/cross-doc-event-coref), a pair of mentions can be encoded within their contexts and a coreference chain sets a training objective. 


## Dataset Creation

### Curation Rationale

CDCR datasets were published in a diverse formats (e.g., CSV, CoNLL, XML-based, JSON) that required tedious parsing effort to be used for model training and data analysis. The unified format enabled effortless dataset use and experiments, including transfer learning across CDCR datasets.  

### Source Data
The repository contains the following datasets: 

| Dataset                                                                 | Coreference target   | Public full documents | Train/val/test splits                                                                                                                 |
|:------------------------------------------------------------------------|----------------------|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| [CD2CR](https://aclanthology.org/2021.eacl-main.21)                     | entity               | yes                   | reused (original)                                                                                                                     |
| [CEREC](https://aclanthology.org/2020.coling-main.30)                   | entity               | yes                   | reused (original)                                                                                                                     | 
| [ECB+](http://www.lrec-conf.org/proceedings/lrec2014/pdf/840_Paper.pdf) | event + entity       | yes                   | reused (original)                                                                                                                     |
| [ECB+METAm](https://aclanthology.org/2024.acl-short.27/)                | event + entity       | yes                   | reused (original)                                                                                                                     |
| [FCC-T]( http://ceur-ws.org/Vol-2593/paper3.pdf)                        | event                | no                    | reused (from [Bugert et al. 2021](https://direct.mit.edu/coli/article/47/3/575/102774/Generalizing-Cross-Document-Event-Coreference)) |
| [GVC](https://aclanthology.org/L18-1480/)                               | event                | yes                   | reused (from [Bugert et al. 2021](https://direct.mit.edu/coli/article/47/3/575/102774/Generalizing-Cross-Document-Event-Coreference)) |
| [HyperCoref](https://aclanthology.org/2021.emnlp-main.38)               | event                | no                    | reused (original)                                                                                                                     |
| [MEANTIME](https://aclanthology.org/L16-1699)                           | event + entity       | yes                   | new (didn't exist before)                                                                                                             |
| [NewsWCL50r](https://arxiv.org/pdf/2602.17424)                          | event + entity       | no                    | reused (original)                                                                                                                     |
| [NiDENT](https://aclanthology.org/L12-1391/)                            | entity               | yes                   | new (didn't exist before)                                                                                                             |
| [NP4E](https://aclanthology.org/L06-1325/)                              | entity               | yes                   | new (didn't exist before)                                                                                                             |
| [WEC-Eng](https://aclanthology.org/2021.naacl-main.198/)                | event                | yes                   | reused (original)                                                                                                                     |



#### Data Collection and Processing

The data processing is described in the original paper (including Appendix) and more details can be found in the GitHub repository in folders designated to each dataset.  


## Citation


**BibTeX:**
The paper has been accepted to LREC 2026.

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



## Dataset Card Contact

[Anastasia Zhukova](https://gipplab.uni-goettingen.de/team/anastasia-zhukova/), University of Göttingen, Germany
