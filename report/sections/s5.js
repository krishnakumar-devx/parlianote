const L = require("../lib");
const { h1, h2, h3, p, bullets, table, code, figure, pageBreak } = L;

module.exports = () => [
  pageBreak(),
  h1("Appendix A. Reproducing the results"),
  p("All code is in the project folder. Run from the project root (Python 3.13; packages: pandas, pyarrow, gensim, nltk, sentence-transformers, scikit-learn, rank-bm25, streamlit, altair, matplotlib)."),
  table(["Step", "Command", "Output"], [
    ["1. Phase-1 analysis", "Run notebooks/parlianote_analysis.ipynb", "Cleaning, postcode join, LDA, zero-shot classifier, retrieval pilot"],
    ["2. Question-answer pairs and label samples", "python -m src.build_pairs, then python -m src.make_label_samples", "data/qa_pairs.parquet (16,256 pairs); samples to label"],
    ["3. Labelled features", "python -m src.build_label_features", "data/evasiveness_label_features.parquet"],
    ["4. Evaluation tables and figures", "python -m src.make_figures", "results/model_comparison.csv, figures"],
    ["5. Score all pairs", "python -m src.score_pairs, then python -m src.rescore", "data/qa_pairs_scores.parquet"],
    ["6. Topic model", "python -m src.train_lda", "models/lda_model.gensim (c_v 0.4592)"],
    ["7. App data", "python -m src.prepare_app_data, python -m src.merge_scores", "app/data/*.parquet"],
    ["8. Retrieval evaluation", "python -m src.run_retrieval_eval pool, then score", "results/retrieval_comparison.csv"],
    ["9. Pledge evaluation", "python -m src.run_commitment_eval", "reweighted precision and recall"],
    ["10. Run the app", "streamlit run app/app.py", "http://localhost:8501/?postcode=2088"],
  ], [2.2, 4.3, 3.7], { size: 16 }),
  L.caption("Table A1. Reproduction steps. Random seeds are fixed (LDA 42; sampling 42, 11, 7; cross-validation 1)."),

  h1("Appendix B. Annotation guidelines and examples"),
  p("**Direct (D):** the answer addresses the specific thing asked, even if it adds spin. **Partial (P):** the answer stays on the topic but not on the specific ask, or answers only part. **Evasive (E):** the answer deflects, attacks the opposition, refers to earlier answers, or ignores the question. Labels were assigned from the question and the first 110 words of the answer, so late-answer content is not seen."),
  table(["Label", "Question (abridged)", "Answer (abridged)"], [
    ["D", "Is the Prime Minister aware ... that the Prime Minister's discredited forced divestment policy sets a dangerous precedent ...? Does the Prime Minister agree? (Rowland, 2018)", "No, I don't. I don't support that. What the ACCC retail electricity pricing inquiry final report found ... (Morrison)"],
    ["P", "Does the Prime Minister agree with his Treasurer? Will the Prime Minister join with Labor to ensure that teachers can't be sacked because of who they are or who they love? (Shorten, 2018)", "I'm happy to table the letter that I sent to the Leader of the Opposition today ... it's very important that we act in this parliament over the next fortnight ... (Morrison)"],
    ["E", "Senator Canavan resigned from cabinet 43 days ago. Why is the Deputy Prime Minister still sitting on your front- (Shorten, 2017)", "The honourable member's question has a distinctly familiar ring to it, and I refer him to my earlier answers. (Turnbull)"],
  ], [0.8, 4.7, 4.7], { size: 16 }),
  L.caption("Table B1. Worked examples (House of Representatives Hansard, quoted from the public record)."),
  p("**Inter-annotator agreement has not been measured.** All 120 pair labels and 90 sentence labels come from one annotator (the AI assistant, Section 2). A 45-item sheet for independent annotation (`team_labelling/labelling_sheet.xlsx`) and the agreement script (`src/compute_kappa.py`) are ready; until at least two humans complete it, the accuracy figures in Section 5 should be treated as preliminary."),

  h1("Appendix C. Key code"),
  h3("Friendly-question definition (src/blocs.py)"),
  ...code([
    'BLOC = {"ALP": "Labor", "LIB": "Coalition", "NP": "Coalition", "NAT": "Coalition", "LNP": "Coalition", "CLP": "Coalition"}',
    'def bloc(party): return BLOC.get(str(party), "Other")',
    '# friendly question  <=>  bloc(questioner party) == bloc(minister party)',
  ]),
  h3("Hybrid retrieval and extractive summary (src/retrieval.py)"),
  ...code([
    'def bm25_search(ps, bm, query, k=60, year_range=None):',
    '    scores = bm.get_scores(tokens(query))',
    '    top = [i for i in np.argsort(-scores)[:k] if scores[i] > 0]',
    '    return ps.iloc[top].assign(bm25=scores[top])',
    '',
    'def rerank(ce, query, cand, k_final=8):',
    '    cand["rerank"] = ce.predict([(query, t) for t in cand["text"]], batch_size=32)',
    '    return cand.sort_values("rerank", ascending=False).head(k_final)',
    '',
    'def extractive_summary(bi, query, hits, n=3):   # n verbatim sentences closest to the query',
    '    sents = [(i, s) for i, r in hits.iterrows() for s in split_sentences(r["text"])]',
    '    sim = bi.encode([s for _, s in sents], normalize_embeddings=True) @ bi.encode([query], normalize_embeddings=True).T',
    '    ...  # pick top n, drop near-duplicates, return in date order with citations',
  ]),
  h3("Directness model and evaluation protocol (src/make_figures.py, src/rescore.py)"),
  ...code([
    'lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))',
    'cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)',
    'for tr, te in cv.split(Xs, yt):',
    '    m = clone(clf).fit(Xs.iloc[tr][cols], yt[tr]); p = m.predict(Xs.iloc[te][cols])',
    '    acc.append(accuracy_score(yt[te], p)); mf.append(f1_score(yt[te], p, average="macro"))',
    '# features: LEX (lexical cues), SEM (bi-encoder cosine), CE (cross-encoder relevance) + same_bloc',
  ]),
  h3("Pledge detection (src/commitment.py)"),
  ...code([
    'FIRM = re.compile(r"\\b(we|i|the government|this government|our government|the coalition|labor|the minister)\\s+"',
    '                  r"(will|shall|are going to|have committed to|are committed to|commit(s)? to|guarantee|promise|pledge)\\b", re.I)',
    'SPECIFIC = re.compile(r"(\\d|\\$|\\bby (the end of )?(next|this|20\\d\\d)|\\bevery (week|year|month)|\\bper cent\\b)", re.I)',
    'def commitment_type(s):',
    '    if is_commitment(s) == "N": return "none"',
    '    return "specific pledge" if SPECIFIC.search(s) else "general pledge"',
  ]),

  /*APP_APPENDIX*/
];
