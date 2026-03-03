# Parsing HyperCoref 

Original repository of HyperCoref dataset: https://github.com/UKPLab/emnlp2021-hypercoref-cdcr/

### Papers
https://aclanthology.org/2021.emnlp-main.38/
```
@inproceedings{bugert2021event,
    title = {{Event Coreference Data (Almost) for Free: Mining Hyperlinks from Online News}},
    author = "Bugert, Michael and Gurevych, Iryna",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = {11},
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.38",
    doi = "10.18653/v1/2021.emnlp-main.38",
    pages = "471--491",
}
  ```


### To Parse HyperCoref
1. follow the guideline on this page: https://github.com/UKPLab/emnlp2021-hypercoref-cdcr/tree/master/hypercoref 
2. we take only part of the dataset, which was used in the experiments by Bugert et al. Moreover, we narrow down HyperCoref only to BBC outlet
because the authors show in the Appendix A2 that ABC news have problems with selecting anchoring text (i.e., mentions) to events. 
3. after downloading a list of urls of the news articles, make sure that you use only the urls for BBC outlet. 
4. after the execution of the data recollection and preprocessing is over, copy the folder of the outlet (i.e., ```bbc.com```) 
into the folder ```HyperCoref-prep/HyperCoref/```. Make sure that each outlet contains a subfolder ```6_CreateSplitsStage_create_splits```. 
You might need to rename a folder from ```7_CreateSplitsStage_create_splits``` to ```6_CreateSplitsStage_create_splits```.
5. execute ```python parse_hypercoref.py``` 

Initially the original paper on HyperCoref stated that the dataset contains only events, but out data analysis revealed that the dataset also contains a few entities. 
We create ```entity_mentions.json``` with the mentions whose POS tags of the heads of phrases are "ADJ", "ADV", "ADP", "NUM", "NOUN", "PRON", or "PROPN". We reuse the train/val/test subset split 
provided in the dataset. We parse the dataset to keep as close as possible to the experimental setup described in the paper, i.e., 
1. we ignore subtopics if they only contain a singleton-mention
2. we downsample the train sets to 25k mentions for each outlet
3. we downsample the val splits to 1.7k mentions for ABC and 2.4k for BBC 
4. when downsampling, we follow a strategy of keeping the larger clusters first and then moving on to smaller sizes. We randomly 
select clusters when we need to make a specified. This strategy minimized the number of singletons in the dataset.
5. from the test tests, we remove subtopics that contain only one mention (required for the correct evaluation on a subtopic level)
6. we keep the original maximum-span annotation style of HyperCoref 
7. we do not reparse the document but keep the original tokenization. We fit the reparsed mentions to the initially tokenized documents. 

To fit HyperCoref to the topic/subtopic/doc structure of the benchmark, we create subtopics from the original document urls: 
1. the first component of a document url is a topic. We uppercase the first letter it to unify across outlets. 
2. the subtopic is formed as a combination of an outlet and the second component of a url. For ABCNews, we take the second component if it was capitalised. For BBC, we take the non-zero and non-digit component and use the first part before the dash. 
3. the default name of subtopic is the same as topic if we failed to parse a url.

### Topic organization
News articles in the dataset are organized as following: 

```
-> topic (i.e., highest folder name in the news articles' urls)
    -> subtopic (i.e., outlien name + a following folder name in the news articles' urls)
        -> documents (news articles' urls)
   ```

___________________________________

# Event Coreference Data (Almost) for Free: Mining Hyperlinks from Online News
This repository accompanies our paper *Event Coreference Data (Almost) for Free: Mining Hyperlinks from Online News* [published at EMNLP 2021](https://aclanthology.org/2021.emnlp-main.38/).

It contains:
- code for (re-)creating the HyperCoref corpus
- the cross-document event coreference resolution (CDCR) system implementation used in our experiments, which is a modified version of [Cattan et al. 2021's system](https://github.com/ariecattan/coref/)
- [CoNLL files and scores from our experiments](archive/)

Please cite our work as follows:
```
@inproceedings{bugert2021event,
    title = {{Event Coreference Data (Almost) for Free: Mining Hyperlinks from Online News}},
    author = "Bugert, Michael and Gurevych, Iryna",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = {11},
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.38",
    doi = "10.18653/v1/2021.emnlp-main.38",
    pages = "471--491",
}
```

> **Abstract:** Cross-document event coreference resolution (CDCR) is the task of identifying which event mentions refer to the same events throughout a collection of documents. Annotating CDCR data is an arduous and expensive process, explaining why existing corpora are small and lack domain coverage. To overcome this bottleneck, we automatically extract event coreference data from hyperlinks in online news: When referring to a significant real-world event, writers often add a hyperlink to another article covering this event. We demonstrate that collecting hyperlinks which point to the same article(s) produces extensive and high-quality CDCR data and create a corpus of 2M documents and 2.7M silver-standard event mentions called HyperCoref. We evaluate a state-of-the-art system on three CDCR corpora and find that models trained on small subsets of HyperCoref are highly competitive, with performance similar to models trained on gold-standard data. With our work, we free CDCR research from depending on costly human-annotated training data and open up possibilities for research beyond English CDCR, as our data extraction approach can be easily adapted to other languages.

Contact person: Michael Bugert

https://ukp.tu-darmstadt.de

https://tu-darmstadt.de

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.


## Content
The code base of our paper is split across three python projects:

1. [hypercoref/](hypercoref/): Data pipeline for (re-)creating the HyperCoref corpus. It outputs a collection of [Apache parquet](https://parquet.apache.org/) files.
2. [The "hypercoref" branch in another project of ours](https://github.com/UKPLab/cdcr-beyond-corpus-tailored/tree/hypercoref): Another data pipeline which shortens hyperlink anchor texts, and exports the data in a format friendly for CDCR systems.
3. [system/](system/): modified version of [Cattan et al. 2021's CDCR system](https://github.com/ariecattan/coref/) which can be trained/tested on corpora other than ECB+.

We also provide all [CoNLL files and scores from our experiments](archive/).