# Parsing ECB+METAm

Original repository of ECB+METAm dataset: https://github.com/ahmeshaf/llms_coref

### Papers
* For the original ECB+, see Cybulska et al. 2014 https://www.aclweb.org/anthology/L14-1646/ 
  ```
  @inproceedings{cybulska-vossen-2014-using,
      title = "Using a sledgehammer to crack a nut? Lexical diversity and event coreference resolution",
      author = "Cybulska, Agata  and
        Vossen, Piek",
      booktitle = "Proceedings of the Ninth International Conference on Language Resources and Evaluation ({LREC}'14)",
      month = may,
      year = "2014",
      address = "Reykjavik, Iceland",
      publisher = "European Language Resources Association (ELRA)",
      url = "http://www.lrec-conf.org/proceedings/lrec2014/pdf/840_Paper.pdf",
      pages = "4545--4552"
  }
  ```
* For the LLM-reannotation with methaphors, see Ahmed et al. 2024 https://aclanthology.org/2024.acl-short.27/
  ```
  @inproceedings{ahmed-etal-2024-generating,
      title = "Generating Harder Cross-document Event Coreference Resolution Datasets using Metaphoric Paraphrasing",
      author = "Ahmed, Shafiuddin Rehan  and
        Wang, Zhiyong Eric  and
        Baker, George Arthur  and
        Stowe, Kevin  and
        Martin, James H.",
      editor = "Ku, Lun-Wei  and
        Martins, Andre  and
        Srikumar, Vivek",
      booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
      month = aug,
      year = "2024",
      address = "Bangkok, Thailand",
      publisher = "Association for Computational Linguistics",
      url = "https://aclanthology.org/2024.acl-short.27/",
      doi = "10.18653/v1/2024.acl-short.27",
      pages = "276--286",
  }
  ```

### To parse ECB+METAm
1) make sure that you downloaded the dataset by running ```python setup.py``` 
2) execute ```python parse_metam.py```

We assign the subtopic names from [the ECB+ annotation guide](http://www.newsreader-project.eu/files/2013/01/NWR-2014-1.pdf), 
which are saved in ```subtopic_names.json```. The parsing script uses a file ```train_dev_test_split.json``` with the split was proposed by 
[Barhom et al. (2019)](https://aclanthology.org/P19-1409/) to create folders with the train/dev/test splits. 

We create two versions of the dataset: 
1) (commented out) original all annotated mentions and clusters (saved into ```\output_data_unvalidated```)
2) mentions from the manually validated sentences as described in [Cybulska and Vossen (2015)](https://aclanthology.org/W15-0801/)
(saved into ```\output_data```)

### Topic organization
News articles in the dataset are organized as following: 

```
-> topic (original topic_id)
    -> subtopic (topic_id + ecb/ecbplus)
        -> documents (enumerated doc_id extracted from the original doc name)
   ```



----------------------------------------------------------------
# Event Coreference Resolution with LLMs

Modeling code adapted from:
1. [aviclu/CDLM](https://github.com/aviclu/CDLM)
2. [ahmeshaf/lemma_ce_coref](https://github.com/ahmeshaf/lemma_ce_coref)
3. [Helw150/Entity-Mover-Distance](https://github.com/Helw150/Entity-Mover-Distance)

Accompanying code for the ACL 2024 short paper "_Making Event coreference resolution Tough Again. Metaphorically speaking_"

## Contents
  1. [Getting Started](#getting-started)
  2. [Preprocessing](#preprocessing)
  3. [ECB+META Generation](#ecbmeta-generation)
  4. [Annotations](#annotations)
  5. [BiEncoder](#biencoder)
  6. [Lemma Heuristic](#lemma-heuristic)
  7. [Cross-encoder](#cross-encoder)
  8. [Prediction](#prediction)
  9. [Error Analysis](#error-analysis)


## Getting Started
- Install the required packages:

```shell
pip install -r requirements.txt
```

- Spacy model:
```shell
python -m spacy download en_core_web_lg
```
- Change Directory to project
```shell
cd project
```

- OpenAI API Key Setup
The OpenAI API Key can be set up by the below line:
```shell
export OPENAI_API_KEY=<Your-OpenAI-API-Key>
```


## Preprocessing

- These scripts download and process the ECB+ corpus into a pkl corpus file which we call `mention_map.pkl`
```sh
python -m spacy project assets
```
- Preprocess the ECB+ corpus
```sh
python -m spacy project run ecb-setup
```

This will create the corpus file at `corpus/ecb/mention_map.pkl`

### Data Format
Each mention in the corpus file is represented as follows: 
```shell
{
  "mention_id": "12_10ecb.xml_5",
  "topic": "12",
  "doc_id": "12_10ecb.xml",
  "sentence_id": "0",
  "marked_sentence": "The Indian navy has <m> captured </m> 23 Somalian pirates .",
  "marked_doc": "The Indian navy has <m> captured </m> 23 Somalian ...",
  "mention_text": "captured",
  "lemma": "capture",
  "men_type": "evt",
  "gold_cluster": "ACT17403639225065902",
  "sentence": "The Indian navy has captured 23 Somalian pirates .",
  "start_char": 20,
  "end_char": 28,
  "neighbors_left": [],
  "neighbors_right": [sentence_1, sentence_2, ...]
}
```

## ECB+META Generation
### ECB+META<sub>1</sub>
Run the following scripts to generate the corpus file for the single-word metaphoric transformation of ECB+ at: 
`corpus/ecb_meta_single/mention_map.pkl`

- Run GPT-4 pipeline:
```shell
python -m scripts.llm_pipeline corpus/ecb/ test  --experiment-name meta_single
python -m scripts.llm_pipeline corpus/ecb/ dev --experiment-name meta_single
python -m scripts.llm_pipeline corpus/ecb/ debug_split --experiment-name meta_single
```
- Generate corpus file:
```shell
python scripts/merge_meta.py ./outputs/meta_single/merged.pkl ./outputs/meta_single/gpt-4*.pkl
python -m scripts.parse_meta save-doc-sent-map ./outputs/meta_single/merged.pkl ./corpus/ecb/doc_sent_map.pkl ./corpus/ecb_meta_single/doc_sent_map.pkl
python -m scripts.parse_meta parse ./outputs/meta_single/merged.pkl  ./corpus/ecb_meta_single/doc_sent_map.pkl ./corpus/ecb/mention_map.pkl ./corpus/ecb_meta_single/mention_map.pkl
```

### ECB+META<sub>_m_</sub>
Run the following scripts to generate the corpus file for the multi-word metaphoric transformation of ECB+ at: 
`corpus/ecb_meta_multi/mention_map.pkl`

- Run GPT-4 pipeline:
```shell
python -m scripts.llm_pipeline corpus/ecb/ test  --experiment-name meta_multi
python -m scripts.llm_pipeline corpus/ecb/ dev --experiment-name meta_multi
python -m scripts.llm_pipeline corpus/ecb/ debug_split --experiment-name meta_multi
```
- Generate corpus file:
```shell
python scripts/merge_meta.py ./outputs/meta_multi/merged.pkl ./outputs/meta_multi/gpt-4*.pkl
python -m scripts.parse_meta save-doc-sent-map ./outputs/meta_multi/merged.pkl ./corpus/ecb/doc_sent_map.pkl ./corpus/ecb_meta_multi/doc_sent_map.pkl
python -m scripts.parse_meta parse ./outputs/meta_multi/merged.pkl  ./corpus/ecb_meta_multi/doc_sent_map.pkl ./corpus/ecb/mention_map.pkl ./corpus/ecb_meta_multi/mention_map.pkl

```