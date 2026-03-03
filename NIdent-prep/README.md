# Parsing NIdent 

Original repository of English NIdent dataset: https://clic.ub.edu/corpus/en/nident

### Papers
* Concept of near-identity https://aclanthology.org/L10-1103/ 
```
@inproceedings{recasens-etal-2010-typology,
    title = "A Typology of Near-Identity Relations for Coreference ({NIDENT})",
    author = "Recasens, Marta  and
      Hovy, Eduard  and
      Mart{\'\i}, M. Ant{\`o}nia",
    booktitle = "Proceedings of the Seventh International Conference on Language Resources and Evaluation ({LREC}'10)",
    month = may,
    year = "2010",
    address = "Valletta, Malta",
    publisher = "European Language Resources Association (ELRA)",
    url = "http://www.lrec-conf.org/proceedings/lrec2010/pdf/160_Paper.pdf"
}
  ```
* NIdent dataset https://aclanthology.org/L12-1391/
```
@inproceedings{recasens-etal-2012-annotating,
    title = "Annotating Near-Identity from Coreference Disagreements",
    author = "Recasens, Marta  and
      Mart{\'\i}, M. Ant{\`o}nia  and
      Orasan, Constantin",
    booktitle = "Proceedings of the Eighth International Conference on Language Resources and Evaluation ({LREC}'12)",
    month = may,
    year = "2012",
    address = "Istanbul, Turkey",
    publisher = "European Language Resources Association (ELRA)",
    url = "http://www.lrec-conf.org/proceedings/lrec2012/pdf/674_Paper.pdf",
    pages = "165--172"
}
```

### To Parse NIdent
1. request access to the original files on http://clic.ub.edu/corpus/en/. Download the dataset and make sure that the XML files are located in the following repository: 
```NIdent-prep/NIdent/english-corpus```.
2. download NP4E via ```setup.py```. NP4E is required to maintain the subtopic structure (NIdent is reannotated NP4E). 
3. execute ```python parse_nident.py``` 

NIdent annotated only entities, so ```event_mentions.json``` are saved as empty list. MMAX format didn't provide an extra tag to 
link coreference chains from the event-related documents into cross-document clusters. We performed manual annotation to merge the within-document
chains into cross-document clusters.

We propose a train-val-test split for NIdent in the ```train_val_test_split.json``` file. The split is on the subtopic level
and assigns three subtopics for training and one per validation and test. 

A mapping of the subtopic IDs to subtopic names is the following: 0) bukavu, 1) china, 2) israel, 3) peru, 4) tajikistan.

### Topic organization
News articles in the dataset are organized as following: 

```
-> topic (one topic about bomb, explosion, and kidnap)
    -> subtopic (event)
        -> documents (news articles)
   ```

The dataset contains _one topic_ about bomb, explosion, and kidnap. Subtopics report about different events within this topic.  

____________________________________________________
# NIdent
 

NIdent-EN and NIdent-CA are two English and Catalan language corpora annotated with near-identity tags. NIdent-EN contains 49,279 tokens and  has its origins in the NP4E corpus (Hasler et al., 2006) from the Reuters Agency. Near-coreferent mentions represent 12% of all coreferent mentions. NIdent-CA comes from AnCora-CA corpus (Recasens and Martí, 2010) and contains 51.622 tokens. AnCora-CA comprises newspaper and newswire articles from El Periódico newspaper, and the ACN news agency. Near-coreferent mentions represent 16% of all coreferent mentions.

The near-coreference annotation was obtained implicitly, based on the idea that different annotators would disagree in labelling a near-identity relation if the only two options they were given were “coreference” and “non-coreference”. Five annotators were asked to annotate the same Catalan and English corpora in parallel with coreference and non-coreference relations. Afterwards we relabelled as “near-identity” the relations that were annotated as coreferent by some but not all the annotators. For a more detailed description of the merging algorithm and the NIdent corpora, we refer the reader to our paper presented in LREC 2012 (Recasens et al., 2012). Please cite this paper if you use our data.


Laura Hasler, Constantin Orasan, and Karin Naumann. 2006. "NPs for Events: Experiments in coreference annotation". In Proceedings of LREC 2006, pages 1167–1172.

Marta Recasens and M. Antònia Martí. 2010. "AnCora-CO: Coreferentially annotated corpora for Spanish and Catalan". In Language Resources and Evaluation, 44(4):315–345.

Marta Recasens, M. Antònia Martí and Constantin Orasan. 2012. "Annotating Near-Identity from Coreference Disagreements". In Proceedings of LREC 2012, pag. 165-172.