# Parsing NewsWCL50r

Original repository of NewsWCL50r: https://github.com/anastasia-zhukova/NewsWCL50r 

### Paper
https://arxiv.org/pdf/2602.17424

```
@inproceedings{Zhukova2026b,
  author    = {Zhukova, Anastasia and Hamborg, Felix and Donnay, Karsten and Meuschke, Norman and Gipp, Bela},
  title     = {Diverse Word Choices, Same Reference: Annotating Lexically-Rich Cross-Document Coreference},
  booktitle = {Proceedings of the International Conference on Artificial Intelligence, Computer, Data Sciences and Applications (ACDSA 2026)},
  year      = {2026},
  month     = feb,
  address   = {Boracay, Philippines},
  publisher = {IEEE},
  url       = {https://arxiv.org/pdf/2602.17424}
}
```

Original paper:
```
@InProceedings{Hamborg2019a,
  author    = {Hamborg, Felix and Zhukova, Anastasia and Gipp, Bela},
  title     = {Automated Identification of Media Bias by Word Choice and Labeling in News Articles},
  booktitle = {Proceedings of the ACM/IEEE Joint Conference on Digital Libraries (JCDL)},
  year      = {2019},
  month     = {Jun.},
  location  = {Urbana-Champaign, IL, USA},
  doi       = {10.1109/JCDL.2019.00036}
}
```

### To parse NewsWCL50
1) obtain the text files of the news article by contacting Anastasia Zhukova: zhukova@gipplab.org
2) execute ```python parse_newswcl50.py``` 

The dataset contains 10 topics with no subtopic level. But according to [the definition](README.md) subtopics contain 
event-related articles whereas topic aggregate subtopics. To ensure that the dataset fits into the structure of ```topic/subtopic/document```,
we turn each original topic into a subtopic and place the subtopics under the topic of the identical to subtopic ID. 

Since the dataset contained newline delimiters in the original texts to separate paragraphs, we kept them in the datasets as original tokens. 
To ensure that these symbols do cause troubles in parsing CoNLL format, we saved them as ```\\n```. So if you want to 
convert the dataset into the original plain text, remember to replace these symbols with the correct ```\n```. 

We reuse a train-val-test split for NewsWCL50r indicated in the ```train_val_test_split.json``` file. The split is on the topic level
and ensures that phrasing diversity within each split is comparable between the splits.


### Topic organization
News articles in the dataset are organized as following:
```
-> topic (same as subtopic)
    -> subtopic (original topic)
        -> documents (news articles)
   ```

____________________________

# Diverse Word Choices, Same Reference: Annotating Lexically-Rich Cross-Document Coreference

## Description 
The repository provides the supplement information for the lexical-rich cross document coreference resolution annotation scheme described in 

The repository contains the following files for the reannotated NewsWCL50r and ECB+r:
1. coding book,
2. MAXQDA files (download the [free MAXQDA reader](https://www.maxqda.com/products/maxqda-reader)), 
3. CSV explorted annotation files, 
4. JSON files following the unified CDCR format [uCDCR](https://github.com/anastasia-zhukova/uCDCR)

The code for the data analysis is available under: https://github.com/anastasia-zhukova/uCDCR 

To cite the [original NewsWCL50 dataset](https://github.com/fhamborg/NewsWCL50) please use: 
```
@InProceedings{Hamborg2019a,
  author    = {Hamborg, Felix and Zhukova, Anastasia and Gipp, Bela},
  title     = {Automated Identification of Media Bias by Word Choice and Labeling in News Articles},
  booktitle = {Proceedings of the ACM/IEEE Joint Conference on Digital Libraries (JCDL)},
  year      = {2019},
  month     = {Jun.},
  location  = {Urbana-Champaign, IL, USA},
  doi       = {10.1109/JCDL.2019.00036}
}
```


## Related publications 
For more information about related articles to the CDCR-aspect of NewsWCL50, please read: 
```
@InProceedings{Zhukova2021,
    author="Zhukova, Anastasia
        and Hamborg, Felix
        and Donnay, Karsten
        and Gipp, Bela",
    editor="Toeppe, Katharina
        and Yan, Hui
        and Chu, Samuel Kai Wah",
    title="Concept Identification of Directly and Indirectly Related Mentions Referring to Groups of Persons",
    booktitle="Diversity, Divergence, Dialogue",
    year="2021",
    publisher="Springer International Publishing",
    address="Cham",
    pages="514--526",
    isbn="978-3-030-71292-1",
    url={https://link.springer.com/chapter/10.1007/978-3-030-71292-1_40}
}
```


```
@inproceedings{zhukova-etal-2022-towards,
    title = "Towards Evaluation of Cross-document Coreference Resolution Models Using Datasets with Diverse Annotation Schemes",
    author = "Zhukova, Anastasia  and
      Hamborg, Felix  and
      Gipp, Bela",
    editor = "Calzolari, Nicoletta  and
      B{\'e}chet, Fr{\'e}d{\'e}ric  and
      Blache, Philippe  and
      Choukri, Khalid  and
      Cieri, Christopher  and
      Declerck, Thierry  and
      Goggi, Sara  and
      Isahara, Hitoshi  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Mazo, H{\'e}l{\`e}ne  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.522/",
    pages = "4884--4893"
}
```

```
@InProceedings{zhukova-2022-xcoref,
author="Zhukova, Anastasia
and Hamborg, Felix
and Donnay, Karsten
and Gipp, Bela",
editor="Smits, Malte",
title="{XC}oref: Cross-document Coreference Resolution in the Wild",
booktitle="Information for a Better World: Shaping the Global Future",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="272--291",
isbn="978-3-030-96957-8", 
url={https://link.springer.com/chapter/10.1007/978-3-030-96957-8\_25}
}
```

## License
Licensed under the Attribution-ShareAlike 4.0 International (the "License"); you may not use NewsWCL50r except in compliance with the License. A copy of the License is included in the project, see the file LICENSE.

Copyright 2025 The NewsWCL50r team