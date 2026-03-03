# Parsing NP4E

Original repository of NP4E dataset: http://clg.wlv.ac.uk/projects/NP4E/

### Paper
https://aclanthology.org/L06-1325/ 
```
@inproceedings{hasler-etal-2006-nps,
    title = "{NP}s for Events: Experiments in Coreference Annotation",
    author = "Hasler, Laura  and
      Orasan, Constantin  and
      Naumann, Karin",
    booktitle = "Proceedings of the Fifth International Conference on Language Resources and Evaluation ({LREC}{'}06)",
    month = may,
    year = "2006",
    address = "Genoa, Italy",
    publisher = "European Language Resources Association (ELRA)",
    url = "http://www.lrec-conf.org/proceedings/lrec2006/pdf/539_pdf.pdf",
}
```

### To parse NP4E
1) make sure that you downloaded the dataset by running ```python setup.py``` 
2) execute ```python parse_np4e.py``` 

Since annotation of events was limited to five specific events and the authors didn't annotate event clusters for all 
topics, we parse only entity clusters and save ```event_mentions.json``` as empty list. 
Unlike most CDCR datasets, the events described as a noun phrase, e.g., an attack, is annotated as entity. 

Since the MMAX format didn't provide an extra tag to link coreference chains from the event-releted documents into 
cross-document clusters, so we applied a simple yet reliable heuristic to restore CDCR clusters. 
If at least two non-pronoun mentions or their heads are identical, we merge the chains into clusters. We make an exception 
to the chains where there is one ovelap with a proper noun, and merge such cases as well. 

We propose a train-val-test split for NP4E in the ```train_val_test_split.json``` file. The split is on the subtopic level
and assigns three subtopics for training and one per validation and test. 

A mapping of the subtopic IDs to topic names is the following: 0) bukavu, 1) china, 2) israel, 3) peru, 4) tajikistan.


### Topic organization
News articles in the dataset are organized as following:
```
-> topic (one topic about bomb, explosion, and kidnap)
    -> subtopic (event)
        -> documents (news articles)
   ```
The dataset contains _one topic_ about bomb, explosion, and kidnap. Subtopics report about different events within this topic.  


--------------------------------------------------------------------------------------------------------------------------
# Annotation of Cross-Document Coreference: A Pilot Study 

## Objective
The main objective of this project was to develop a methodology in the form of detailed annotation schemes and guidelines for marking noun phrase and event coreference within one document and across different documents. This focus meant that the different types of coreference can be fully investigated in order that the most appropriate sets of guidelines and schemes possible for the annotation could be developed. This ensures that future annotations based on this methodology capture the phenomena both reliably and in detail. The project involved extensive discussions between annotators in order to redraft and improve the guidelines as well as major changes to enable the existing annotation tool PALinkA, which was used to annotate the sample corpus, to accommodate events as well as noun phrases. The project was funded by British Academy.

## People
Prof. Ruslan Mitkov - principal investigator
Dr. Constantin Orasan - project manager
Dr. Laura Hasler - research assistant
Karin Naumann - research assistant


## Guidelines
One of the first steps of developing the annotation methodology was to decide on our definition of coreference. Following van Deemter and Kibble (1999), which is also reflected in Mitkov et al. (2000), we used a narrow definition to ensure higher quality and reliability of annotation. For NP coreference there is a sizable research on annotation guidelines. Even so, investigation of the existing guidelines revealed that quite often they are either not appropriate for the domain of security/terrorism news, or they mark too few phenomena. The annotation methodology used in this project is not limited only to IDENTITY relations between entities, but it also incorporates relations such as SYNONYMY, GENERALISATION and SPECIALISATION. In order to ensure consistency between annotators, wherever possible, the relations between entities were automatically extracted from WordNet. The annotation scheme also encodes the type of realisation of coreferential relations. A coreferential item is labelled as NP, COPULAR, APPOSITION, BRACKETED TEXT or SPEECH PRONOUN. The journalistic domain selected for investigation also required to extend the existing guidelines in order to address the problem of continuous change from direct to indirect speech.

NP coreference guidelines (pdf)
NP coreference guidelines outstanding issues (pdf)
The guidelines for event annotation developed in this project focused mainly on what constitutes an event and how to identify the appropriate arguments (participants and other slots) associated with an event. Investigation of the selected clusters revealed that each cluster has specific events and that it is very difficult to annotate all the events related to terrorism/security issues in all the five clusters. For this reason, two clusters which contained quite different types of events (the cluster about war in Zaire focused on bombing and attacks, whilst the cluster about Peru concentrated on a hostage crisis) were selected for analysis and annotation. This approach ensured comprehensive analysis of the events.

Guidelines for Annotation Event Coreference (pdf)
Outstanding issues for Event Coreference Guidelines(pdf)


## Publications
Laura Hasler, Constantin Orasan and Karin Naumann (2006) NPs for Events: Experiments in Coreference Annotation. In Proceedings of the 5th edition of the International Conference on Language Resources and Evaluation (LREC2006), 24 -- 26 May, Genoa, Italy, pp. 1167 -- 1172 (pdf:LREC poster)

## Corpus
A by-product of this project is a corpus annotated for the phenomena investigated in this project. The corpus annotated with NP coreference contains all five clusters used in the project and totals almost 55000 words. During the research, it became clear that it is not possible to produce a corpus of similar size for event coreference and for this reason this corpus contains only slightly over 12,500 words.

Cluster	NP coreference	Events
Bukavu	Annotator 1 (8917 words, 16 texts)
Annotator 2 (2900 words, 5 texts)	Annotator 1 (2720 words, 5 texts)
Annotator 2 (3046 words, 5 texts)
China	Annotator 1 (6775 words, 19 texts)	N/A
Israel	Annotator 1 (10900 words, 20 texts)	N/A
Peru	Annotator 1 (12541 words, 19 texts)	Annotator 1 (3640 words, 5 texts)
Annotator 2 (3179 words, 5 texts)
Tajikistan	Annotator 1 (10600 words, 20 texts)
Annotator 2 (2716 words, 5 texts)	N/A


## The corpus in MMAX format
The corpus annotated with NP coreference is also available in the MMAX format. The corpus was converted by Yannick Versley using the palinka2mmax.py script he developed. This version of the corpus can be downloaded from here.

## Contact details
For comments and questions please contact Constantin Orasan