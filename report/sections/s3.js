const fs = require("fs");
const L = require("../lib");
const { h1, h2, h3, p, bullets, numbered, table, figure } = L;

const read = (f) => fs.readFileSync(f, "utf8");
const csv = (f) => read(f).trim().split(/\r?\n/).map((l) => l.split(",").map((x) => x.replace(/^"|"$/g, "")));
const topics = csv("results/topic_summary.csv").slice(1);
const cmp = csv("results/model_comparison.csv").slice(1);
const get = (subsetPrefix, task, model) => cmp.find((r) => r[0].startsWith(subsetPrefix) && r[1] === task && r[2] === model);
const f2 = (x) => Number(x).toFixed(2);
const sc = JSON.parse(read("results/scoring_summary.json"));
const pct = (x) => `${Math.round(x * 100)}%`;
const minis = csv("results/minister_directness.csv").slice(1);
let ret = null;
try { ret = csv("results/retrieval_comparison.csv").slice(1); } catch (e) { /* not yet available */ }

const modelRows = ["Majority class", "TF-IDF + SVM", "Lexical cues + LR", "Embedding similarity + LR", "Cross-encoder + LR", "All features + LR"].map((m) => {
  const a3 = get("All", "3-class", m), ab = get("All", "evasive vs not", m);
  const o3 = get("Other", "3-class", m), ob = get("Other", "evasive vs not", m);
  return [m, `${f2(a3[3])} / ${f2(a3[4])}`, `${f2(ab[3])}`, `${f2(o3[3])} / ${f2(o3[4])}`, `${f2(ob[3])}`];
});

module.exports = () => [
  h1("5. Findings"),
  p("All accuracy figures below are estimates from small, single-annotator samples and should be read with the uncertainty stated; Section 5.7 summarises what the evidence does and does not support."),

  h2("5.1 What MPs talk about, and how it has changed"),
  p("The LDA model reached a c_v coherence of **0.459**. Table 5 lists the ten topics, named from their highest-weight words, and the share of the 247,011 speeches of 80 words or more that each accounts for (the other 123,850 speeches are shorter and receive no topic). The largest single topic, at 24.6%, is partisan debate (\"labor, party, prime, opposition\"), so a quarter of substantive speech is about the contest between parties rather than a policy area."),
  table(["Topic (our name)", "Highest-weight words", "Share"], topics.map((t) => [t[1], t[2].replace(/;/g, ","), pct(Number(t[4]))]), [2.6, 6, 1]),
  L.caption("Table 5. LDA topics, top words and share of topic-labelled speeches (all MPs, 1998-2025)."),
  ...figure("results/fig_topics_over_time.png", 15.5, "Figure 2. Share of topic-labelled speeches by year, all MPs. Partisan debate fell from 36% of speeches in 1998 to 18% in 2020 (22% in 2025); community and family life rose from 5% to 19%; defence and veterans fell from 11% to 3%; legislation and amendments halved from 12% to 6%."),
  p("These shifts are descriptive. Some may reflect changes in parliamentary practice (for example more short constituency statements) rather than changes in what MPs care about, and we did not test alternative numbers of topics."),

  h2("5.2 Fixed-issue classification"),
  p("With raw cosine similarity, one category absorbed 30% of all speeches; after z-score calibration and thresholds the twelve issues were spread more evenly (Figure 3) and 8,186 of the 19,600 sampled speeches (41.8%) received a confident label. Grading 40 random confident labels by hand, 31 were correct (**77.5%**, 95% Wilson interval 62.5% to 87.7%). Five of the nine errors were the same category, Indigenous Affairs, being applied to generic community language (a sports club, a jobs summit, community radio, a bushfire response), a systematic bias rather than random noise. Because 58% of speeches fit none of the twelve issues, we did not use this classifier in the application."),
  ...figure("results/fig_issue_distribution.png", 11.5, "Figure 3. Confidently classified speeches by issue (phase-1 sample of 19,600). *Indigenous Affairs includes the systematic false positives described above."),

  h2("5.3 How directly do ministers answer?"),
  p(`The strongest pattern in our labels is not about the answer at all: of ${56} pairs where the question came from the same bloc as the minister, 45 were labelled Direct, 9 Partial and only 2 Evasive, whereas of 64 pairs from another bloc only 7 were Direct, 30 Partial and 27 Evasive (Figure 4). Friendly "Dorothy Dixer" questions are almost always answered directly. Any model trained on all pairs therefore partly learns who is asking, so we also evaluate on other-bloc questions only.`),
  ...figure("results/fig_labels_by_question_type.png", 9.5, "Figure 4. Hand labels by question type (120 pairs). Same bloc = questioner and minister are both Labor or both Coalition (Liberal and National counted together)."),
  table(["Model", "All pairs (n=120)\n3-class acc / macro-F1", "All pairs\nevasive vs not acc", "Other-bloc (n=64)\n3-class acc / macro-F1", "Other-bloc\nevasive vs not acc"], modelRows, [3, 2.3, 1.8, 2.3, 1.8]),
  L.caption("Table 6. Repeated 5-fold cross-validation (10 repeats), mean over folds. Fold standard deviations are 0.06 to 0.14, so most differences between non-baseline models are not reliable."),
  ...figure("results/fig_model_comparison.png", 15.5, "Figure 5. Classical versus semantic approaches. Left: macro-F1 for the three-class task; right: accuracy for evasive-versus-not. Grey bars use all pairs, navy bars use other-bloc questions only."),
  p(`**Reading the results.** (1) On all pairs, lexical cues with logistic regression reach ${f2(get("All", "3-class", "Lexical cues + LR")[3])} accuracy against a ${f2(get("All", "3-class", "Majority class")[3])} majority baseline. (2) On other-bloc questions, which is the case that matters, every classical or combined model reaches about ${f2(get("Other", "3-class", "Lexical cues + LR")[3])} three-class accuracy against ${f2(get("Other", "3-class", "Majority class")[3])}, and about ${f2(get("Other", "evasive vs not", "Lexical cues + LR")[3])} against ${f2(get("Other", "evasive vs not", "Majority class")[3])} for evasive versus not: roughly ten points above baseline with a standard deviation near ten points. (3) **Bi-encoder similarity between question and answer is no better than the baseline** (${f2(get("Other", "evasive vs not", "Embedding similarity + LR")[3])}): an evasive answer usually still talks about the same subject, so topical similarity cannot detect evasion. (4) The cross-encoder, which reads question and answer together, had the numerically highest two-class accuracy on other-bloc questions (${f2(get("Other", "evasive vs not", "Cross-encoder + LR")[3])}), but the gap to the lexical and TF-IDF models is far smaller than the fold-to-fold variation, and it was weaker for three classes. We deployed the all-features model because it is the most complete, not because it is demonstrably better.`),
  p(`**Applying the model to all ${sc.pairs.toLocaleString()} pairs** (excluding questions put to the Speaker) it labels ${pct(sc.pred_share_friendly.D)} of friendly questions Direct, and ${pct(sc.pred_share_opposition.E)} of other-bloc answers Evasive, ${pct(sc.pred_share_opposition.P)} Partial and ${pct(sc.pred_share_opposition.D)} Direct. The predicted evasive share is higher than the 42% in our labelled sample (27 of 64), so the model over-predicts evasion and absolute rates should not be quoted; comparisons are only meaningful relative to the all-minister baseline of ${pct(sc.baseline_evasive_opposition)}. Of the ${sc.n_ministers_ge40} ministers with at least 40 other-bloc answers, ${sc.ministers_ci_excludes_baseline_high} have an interval entirely above the baseline and ${sc.ministers_ci_excludes_baseline_low} entirely below it. We do not publish a ranking: the model has not been validated against an independent measure, and the pattern is confounded with era and role (for example the opposition-facing style of a prime minister versus a portfolio minister).`),

  h2("5.4 Retrieval: what did my MP say about X?"),
  p("We compared three retrievers on 15 queries across five MPs (Steggall, Bandt, Plibersek, Wilkie, Tink): **BM25 alone**, **BM25 followed by cross-encoder re-ranking** (the application's method) and **dense embedding search** with MiniLM over every passage. Ten queries used wording an MP would plausibly use (\"paid parental leave\", \"poker machines and gambling harm\"); five were paraphrases where the MP's own words differ from the query (\"support for mums and dads with newborn babies\", \"cutting the rubbish that ends up in landfill\"). For each query we pooled the top three passages of every method, removed duplicates, hid which method found each, and judged every pooled passage relevant or not (relevant means the MP, in their own words, substantively discusses the topic). Table 7 gives precision at 3."),
  ...(ret ? [
    table(["Method", "All 15 queries\n(45 slots)", "10 keyword queries\n(30 slots)", "5 paraphrase queries\n(15 slots)"],
      [["bm25", "BM25 only"], ["bm25+cross-encoder", "BM25 + cross-encoder (ours)"], ["dense (MiniLM)", "Dense MiniLM search"]].map(([k, label]) => {
        const r = ret.find((x) => x[0] === k);
        return [label, `${r[2]} of ${r[3]} (${f2(r[1])})`, `${r[4]} of ${r[5]} (${f2(r[6])})`, `${r[7]} of ${r[8]} (${f2(r[9])})`];
      }), [3.2, 2.4, 2.6, 2.6]),
    L.caption("Table 7. Relevant passages in the top three, and precision@3 in brackets. One annotator, blind to method."),
  ] : [p("(Retrieval comparison to be inserted.)")]),
  p("**Findings.** BM25 is perfect when the query uses the MP's words (30 of 30) but fails on paraphrases (5 of 15) because there is little word overlap to match. Dense search is the opposite: it copes with paraphrase (12 of 15) but drifts to neighbouring topics on keyword queries (26 of 30), for example returning unpaid parental leave for \"paid parental leave\" and online gambling for \"poker machines\". The hybrid combines exact matching with a model that reads query and passage together and was relevant in 43 of 45 slots (0.96). Its two misses were paraphrase queries where BM25 supplied few useful candidates; because re-ranking only reorders BM25's top 60, it cannot recover passages that share no words with the query, which is the main limit of the design. With 15 queries and one annotator, differences of one or two passages are within noise, but the pattern is consistent across queries."),
  p("In our phase-1 pilot, a dense FAISS index over one MP's speeches returned relevant passages for all 24 of the top-3 slots across 8 queries; that pilot used one MP and mostly keyword-style queries, so it is a feasibility check rather than a comparison."),

  h2("5.5 Pledge detection"),
  p("The 90 labelled sentences were drawn half from sentences flagged by the first version of our rules and half from unflagged ones (3.8% of 33,761 sentences were flagged), so every estimate below re-weights each sentence by its stratum. The naive baseline, \"the sentence contains *will*\", flags 44 of the 90 but has a population precision of only **0.51**: about half of all *will* sentences are predictions about opponents or description rather than pledges. After the final rule changes (which exclude other people's quoted words and bare nouns such as \"the superannuation guarantee\") the rules flag 43 sentences with a population precision of **0.79** and a recall of **about 0.40**; the recall estimate is imprecise because it rests on 2 commitments found among 45 unflagged sentences. Of the 43 flagged sentences, 24 were only vague or hedged intentions (\"we will continue to deliver\"), which is why a specificity cue is useful: requiring a number, date, dollar amount or frequency left 7 sentences, 6 of which were specific commitments (precision 0.86), recovering 6 of the 12 specific commitments in the sample (raw recall 0.50). Population recall for specific pledges cannot be estimated reliably because it depends on a single unflagged specific commitment. Logistic regression on sentence embeddings reached 0.73 three-class accuracy and 0.62 macro-F1 in cross-validation, against 0.60 and 0.44 for the rules, but the sample is small and we kept the rules in the application because their output can be explained to a user."),

  h2("5.6 The application"),
  p("We exercised all five views in a browser on a CPU-only Windows laptop. **Responsiveness:** the language models load once per server start (about 14 s); the first search for an MP builds a BM25 index in 0.7 s (Steggall, 611 speeches, 5,547 passages) to 2.4 s (Albanese, 8,377 speeches, 28,846 passages); each search then takes 1.2 to 1.5 s including re-ranking and sentence selection. **Postcode lookup:** Table 8 shows nine postcodes that resolved to the expected seat and MP, including two seats that changed hands at the May 2025 election (Melbourne and Brisbane). Brisbane returned the previous MP before we fixed the member-flag problem. The postcode dataset is approximate: it places postcode 2060 (North Sydney) in Warringah, which we have not been able to confirm and which is why the app asks users to confirm their electorate with the AEC when a postcode spans seats."),
  table(["Postcode", "Electorate", "MP returned", "Note"], [
    ["2088", "Warringah", "Zali Steggall (IND)", ""], ["2000", "Sydney", "Tanya Plibersek (ALP)", ""],
    ["2040", "Grayndler", "Anthony Albanese (ALP)", ""], ["3000", "Melbourne", "Sarah Witty (ALP)", "Changed hands, 2025"],
    ["3141", "Macnamara", "Josh Burns (ALP)", ""], ["4000", "Brisbane", "Madonna Jarrett (ALP)", "Changed hands, 2025"],
    ["6000", "Perth", "Patrick Gorman (ALP)", ""], ["2600", "Canberra", "Alicia Payne (ALP)", ""],
    ["7000", "Clark", "Andrew Wilkie (IND)", "Seat formerly Denison"],
  ], [1.2, 1.8, 3, 2.4]),
  L.caption("Table 8. Postcode-to-MP spot checks (nine capital-city postcodes). This is a spot check, not a full validation against the AEC."),
  ...figure("results/screens/fig_app_both.png", 16.6, "Figure 6. The application. Left: Ask the record returns a verbatim extractive summary, each sentence with its Hansard date and page, then the most relevant passages. Right: Question Time shows the share of answers estimated evasive on other-bloc questions beside the all-minister baseline, with a 95% interval and the highest-scoring exchanges for inspection."),
  p("We did not run a user study. The interface choices that matter for usability and trust are therefore justified by design reasoning (Section 4.6): citations beside every quotation, plain-language labels, uncertainty shown with every estimate, and an explicit limits page. A task-based test with members of the public is the most important missing evaluation."),
  h2("5.7 What the evidence supports"),
  table(["Component", "Supported by evidence", "Not supported / caution"], [
    ["Postcode to MP", "Correct on all spot-checked postcodes after the member-flag fix; 86% of Hansard electorates match current postcode data", "Seats that changed hands after the 3 May 2025 election may show the previous MP; postcodes can span electorates"],
    ["Topics over time", "Coherent, interpretable topics (c_v 0.459); clear long-run shifts", "No sweep over k; topic labels are ours; 33% of speeches too short for a topic"],
    ["Search and extractive summary", "Every sentence is verbatim and cited; hybrid retrieval relevant in 43 of 45 blind-judged slots vs 38 (dense) and 35 (BM25)", "Fifteen queries, one annotator; cannot recover passages BM25 misses; no user study"],
    ["Directness estimate", "Roughly 10 points above baseline on other-bloc questions", "Single annotator, n=64 for the hard case, over-predicts evasion, not validated externally; not suitable for ranking people"],
    ["Pledges", "Rules beat the \"contains will\" baseline (precision 0.79 vs 0.51); specificity cue 6 of 7 correct", "Recall about 0.40; 90 labelled sentences; embeddings scored higher in cross-validation"],
  ], [2, 4, 4.2], { size: 16 }),
  L.caption("Table 8. Summary of evidence by component."),
];
