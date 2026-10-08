const L = require("../lib");
const { h1, h2, h3, p, bullets, numbered, table, figure } = L;

module.exports = () => [
  h1("4. NLP methods and techniques, with justification"),
  p("Figure 1 shows the system. All methods were chosen from, or are direct extensions of, the techniques taught in the subject, and each choice was constrained by the same practical facts: a CPU-only laptop, no paid API access, and a requirement that every output shown to a citizen can be traced to the official record."),
  ...figure("results/fig_pipeline.png", 15.5, "Figure 1. System overview. The Hansard corpus is cleaned once; four independent components feed one web application."),

  table(["Technique", "Role in Parlianote", "Why this choice", "Alternative considered"], [
    ["Tokenisation, stop-word removal (Sessions 1-2)", "Prepare text for LDA and TF-IDF", "Standard, transparent; we added 45 parliamentary filler words (\"member\", \"minister\", \"house\") that otherwise dominate every topic", "Lemmatisation (little gain for topic words)"],
    ["LDA topic model (Session 2)", "What an MP talks about, and when", "Unsupervised, interpretable, cheap enough to run on 247,011 speeches on CPU", "BERTopic (heavier); fixed-issue zero-shot (tested, covers only 42% of speeches)"],
    ["Sentence embeddings, all-MiniLM-L6-v2 (Sessions 3, 8)", "Semantic similarity for issue tagging, answer relevance and sentence selection", "Small and fast on CPU, strong general-purpose quality (Reimers and Gurevych, 2019; Wang et al., 2020)", "Larger embedding models (too slow for 1.7M passages)"],
    ["BM25 + cross-encoder re-ranking (Session 8)", "Find the passages that answer \"what did my MP say about X?\"", "BM25 is exact and instant on one MP's text; the cross-encoder reads query and passage together and corrects its ranking on only 60 candidates", "Dense search over every passage (about 9 hours of CPU embedding for the corpus)"],
    ["Extractive summarisation (Session 8, RAG)", "Short answer made only of verbatim sentences", "Every sentence can be checked against Hansard; no hallucination; no API cost", "Generative summaries from an LLM (future work)"],
    ["Supervised classifiers (Sessions 3-4)", "Answer directness; pledge type", "Small labelled data suits linear models with explicit features; coefficients can be inspected", "Fine-tuned transformer (not enough labels); LLM judge (no API)"],
  ], [2.4, 2.5, 3.6, 2.5], { size: 16 }),
  L.caption("Table 4. Techniques, justification and alternatives."),

  h2("4.1 Pre-processing and sampling"),
  p("Text is lower-cased, tokens are alphabetic strings of three or more letters, and English stop words plus 45 parliamentary filler words are removed. For model development we drew a stratified sample of 19,600 speeches of 80 to 3,000 words (700 per year, 1998 to 2025, fixed seed) so that no year dominates. The final topic model is then applied to all 247,011 speeches of 80 words or more."),

  h2("4.2 Topic modelling and issue classification"),
  p("**LDA** (Blei et al., 2003) with 10 topics, 8 passes and a fixed seed was fitted on the 19,600-speech sample with gensim (Rehurek and Sojka, 2010). Topic quality was measured with the c_v coherence score (Roder et al., 2015). We named each topic from its ten highest-weight words (Table 5). Only k = 10 was trained; a sweep over k is future work."),
  p("**Zero-shot issue classification (phase 1).** We also tested whether speeches could be assigned to twelve fixed issues (health, economy, immigration and so on) without labelled data, by embedding a one-line description of each issue and each speech and choosing the nearest issue by cosine similarity. The naive version was biased: one category (Indigenous Affairs) absorbed 5,965 of 19,600 speeches (30%), including a speech about copyright law, because raw cosine scores are not comparable across categories. We fixed this by z-scoring each category's similarity across the corpus and requiring both z >= 1.0 and raw similarity >= 0.28; anything below is labelled Other/Uncertain. Because this labels only 41.8% of speeches, the application uses LDA, which assigns a topic to every substantive speech, and keeps the issue classifier as an analysis tool."),

  h2("4.3 Retrieval and extractive summarisation"),
  p("Each MP's speeches are split into overlapping 110-word passages (stride 90). For a query we (1) score every passage with BM25 (Robertson and Zaragoza, 2009) and keep the top 60, (2) re-rank those 60 with the ms-marco MiniLM cross-encoder (Nogueira and Cho, 2019, describe the approach), and (3) keep the top eight. The summary is the three sentences from those passages that are closest to the query in embedding space, shown in date order with a Hansard citation (date, page, speech number) under each. A year slider restricts the search, and the best early and late passages are shown side by side so a user can see how wording changed. We evaluate three retrieval variants in Section 5.4."),
  p("We deliberately do not generate text. A generative summary cannot be checked sentence by sentence, would need an API key or a GPU, and was the main source of unverifiable claims in our phase-1 pilot, where one fully hand-written summary had to be checked claim by claim. Extractive output keeps the \"every claim links to the record\" promise true by construction."),

  h2("4.4 Estimating how directly an answer responds"),
  p("**Pairing.** For each Question Time question we take the next answer row on the same sitting day before the next question, giving 16,256 pairs from 2011 to 2025. **Labels.** One annotator labelled 120 randomly sampled pairs as Direct (addresses the specific thing asked), Partial (on topic but not the specific ask) or Evasive (deflects, attacks, refers to earlier answers or ignores the question). **Features.** Lexical cues (attack phrases such as \"those opposite\", references to earlier answers, opening with thanks, yes/no openings, digits, lengths, question marks, TF-IDF overlap), bi-encoder cosine similarity between the question and the answer's opening, best and mean 100-word chunk, and cross-encoder relevance scores for the same chunks. **Models.** A majority-class baseline; TF-IDF with a linear SVM on the first 120 answer words (classical); logistic regression on lexical cues (classical), on embedding similarity, on cross-encoder scores, and on all features (semantic and combined). **Protocol.** Repeated stratified 5-fold cross-validation (10 repeats, class-weighted), reporting accuracy and macro-F1. Because friendly (same-party) questions are almost always answered directly, we report results on all pairs and on opposition questions only."),
  p("We did not use a large language model as a judge or an NLI model: neither was available without an API key or GPU, and the cross-encoder gave us a joint reading of question and answer on CPU. This is the main way a stronger system could improve."),

  h2("4.5 Pledge detection"),
  p("A sentence is a **pledge** if it matches a first-person or government future pattern (\"we will\", \"I will\", \"are committed to\", \"guarantee\") and a **specific pledge** if it also contains a number, date, dollar amount or frequency that can be checked later; hedged forms (\"will consider\", \"look at\") are general. The baseline is simply \"the sentence contains will\". We compared the rules with logistic regression on sentence embeddings using 90 labelled sentences."),

  h2("4.6 Web application design"),
  p("The application is a Streamlit app (Figures 6 and 7) reading pre-computed Parquet files, with each MP's speeches loaded on demand through column filters so memory use stays small. The flow is: enter a postcode (or pick an MP by name) -> MP card -> five views (Overview, Ask the record, Question Time, Pledges, About and limits). Design decisions follow from the evaluation:"),
  ...bullets([
    "**Every quoted sentence carries its citation** (date, page, speech number) and a link to the official Hansard site, so a user can check it themselves.",
    "**Uncertainty is shown, not hidden.** Directness is compared only on opposition questions, requires at least 10 exchanges, and is displayed with a Wilson 95% confidence interval (Wilson, 1927) next to the all-minister baseline.",
    "**Neutral wording.** The interface says the scores describe language patterns, not intent, and never ranks MPs.",
    "**Edge cases are handled in the interface:** postcodes that span several electorates, seats whose latest speaker is old, MPs with too few exchanges, and a limits page that restates the data cut-off and annotation limits.",
    "**Shareable links.** Query parameters (postcode, view, search text) preselect a screen, which also let us test each view reproducibly.",
  ]),
  /*APP_SCREENSHOTS*/
];
