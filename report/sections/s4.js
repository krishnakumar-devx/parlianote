const L = require("../lib");
const { h1, h2, h3, p, bullets, numbered, table, code } = L;

module.exports = () => [
  h1("6. Outcomes and value added"),
  p("**What was delivered.** (1) A working web application in which a postcode leads to the MP, a topic profile over time, cited search with an extractive summary, a Question Time directness estimate and a list of pledges, covering 490 MPs and 370,861 speeches. (2) A reproducible pipeline (`src/`) from the raw Parquet file to the app's data, with every evaluation script and result table in `results/`. (3) Two small labelled datasets (120 question-answer pairs, 90 sentences) and a ready-to-use sheet for independent annotation. (4) A data-quality audit that corrected an error in our own earlier analysis."),
  p("**Value for users.** Vote trackers show how an MP voted and keyword tools show where a word appears. Parlianote shows what the MP said, in their own words and in context, and makes it a one-click step to check the source. A citizen can go from a postcode to a cited answer about their MP's position on housing or climate in under a minute; an advocacy group or journalist can compare how that position was worded in different years. The deliberate choice of extractive summaries means that nothing shown is invented, which matters more for a tool about political speech than fluent prose would."),
  p("**Value as a piece of applied NLP.** The project produced a clear negative result that a less careful evaluation would have hidden: question-answer *similarity* does not measure evasiveness, and a classifier that looks accurate on all pairs mostly learns who asked the question. Reporting this, with the evidence, is a more honest contribution than a headline accuracy."),
  p("**What we do not claim.** We have not run a user study, so claims about ease of use rest on design principles and our own testing. We have not validated the directness scores against independent annotators or an external measure, and the application is not deployed publicly."),

  h1("7. Challenges faced and solutions"),
  table(["Challenge", "What happened", "Solution"], [
    ["Biased zero-shot issue classifier", "Raw cosine similarity put 30% of speeches into one category, including unrelated ones", "Z-score calibration per category plus a confidence threshold with an Other/Uncertain class; 41.8% coverage, 77.5% precision"],
    ["Silent loss of recent MPs", "The member flag is blank for MPs elected from 2022, so 8,658 speeches (69 MPs) were dropped; found when the Brisbane postcode returned the wrong MP. The same MPs also lack a uniqueID", "Filter on member != 0; use the display name as a fallback MP key; re-ran topic inference on all 370,861 speeches"],
    ["Unreliable answer flags before 2011", "Few answers were flagged before 2011, so questions had no partner", "Restrict Question Time analysis to 2011-2025 and say so in the interface"],
    ["Evaluation shortcut", "Friendly questions are nearly always answered directly, inflating accuracy; party matching also mislabelled Liberal-National pairs as hostile and distorted minister rates", "Define friendly questions by bloc, report other-bloc results separately, exclude Speaker answers, recompute all results"],
    ["No labelled ground truth", "No dataset marks answers as evasive", "Built a labelling scheme and labelled 120 pairs; report wide uncertainty; prepared an independent labelling sheet"],
    ["Overclaiming risk", "Our poster described inter-annotator agreement and a bias audit that had not been run", "This report states exactly what was done; agreement and bias audit are listed as next steps"],
    ["Compute and cost", "CPU-only machine and no API key: a dense index of about 1.7M passages needs roughly 9 hours; generative summaries were not possible", "BM25 first stage plus cross-encoder on 60 candidates; extractive summaries"],
    ["Tooling on Windows", "Gensim multicore LDA crashed under Windows process spawning; the Streamlit file watcher imported every transformers module", "Single-process LDA training with a separate parallel inference module; file watcher disabled"],
  ], [2.2, 4.1, 4], { size: 16 }),
  L.caption("Table 9. Challenges and solutions."),

  h1("8. Recommendations and next steps"),
  ...numbered([
    "**Independent annotation (highest priority).** Have at least two people who did not build the model label 300 or more pairs, report Cohen's kappa, and re-estimate every directness result. The sheet and script already exist (`team_labelling/`, `src/compute_kappa.py`).",
    "**Stronger directness models.** Fine-tune an NLI or cross-encoder model on the larger labelled set, or evaluate an LLM judge with its known biases controlled (position, length, self-preference), and compare against the classical baselines using the same protocol.",
    "**Topic modelling.** Sweep the number of topics, compare with BERTopic, and check whether the long-run shifts survive.",
    "**Generative summaries with guardrails.** Add an optional LLM summary that must quote retrieved sentences, and test faithfulness with claim-level checking and ROUGE/BERTScore against the extractive summary.",
    "**Coverage.** Include the Senate, committee evidence and Question Time before 2011 (re-parsing the XML), and update the corpus automatically as new Hansard is published.",
    "**User testing and deployment.** Run a task-based study with members of the public (for example the System Usability Scale), then deploy with monthly refreshes and an explicit correction channel.",
    "**Responsible use.** Keep the language-patterns-not-intent framing, never rank individuals on the directness estimate, and give MPs a way to flag errors.",
  ]),

  h1("9. References"),
  ...[
    "Bird, S., Klein, E., & Loper, E. (2009). *Natural language processing with Python*. O'Reilly Media.",
    "Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. *Journal of Machine Learning Research, 3*, 993-1022.",
    "Johnson, J., Douze, M., & Jegou, H. (2021). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data, 7*(3), 535-547. https://doi.org/10.1109/TBDATA.2019.2921572",
    "Katz, L., & Alexander, R. (2025). *A new, comprehensive database of all proceedings of the Australian Parliamentary Debates (1998-2025)* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17351233",
    "Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W., Rocktaschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems, 33*, 9459-9474.",
    "Nogueira, R., & Cho, K. (2019). *Passage re-ranking with BERT* (arXiv:1901.04085). arXiv. https://arxiv.org/abs/1901.04085",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830.",
    "Proctor, M. (n.d.). *Australian postcodes* [Data set]. https://www.matthewproctor.com/australian_postcodes",
    "Rehurek, R., & Sojka, P. (2010). Software framework for topic modelling with large corpora. *Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks*, 45-50.",
    "Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing*, 3982-3992. https://doi.org/10.18653/v1/D19-1410",
    "Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval, 3*(4), 333-389. https://doi.org/10.1561/1500000019",
    "Roder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures. *Proceedings of the Eighth ACM International Conference on Web Search and Data Mining*, 399-408. https://doi.org/10.1145/2684822.2685324",
    "Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. *Advances in Neural Information Processing Systems, 33*, 5776-5788.",
    "Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association, 22*(158), 209-212. https://doi.org/10.1080/01621459.1927.10502953",
  ].map((t) => new L.D.Paragraph({ spacing: { after: 70, line: 255 }, indent: { left: 567, hanging: 567 }, children: L.runs(t, { size: 19 }) })),
];
