const L = require("../lib");
const { h1, h2, h3, p, bullets, numbered, table, spacer } = L;

module.exports = () => [
  h1("Executive summary"),
  p("Everything an MP says in the House of Representatives is on the public record, but 27 years of Hansard transcripts are too long and too dense for most voters to use, so MPs tend to be judged on headlines and advertising. **Parlianote** is a web application that lets a user type a postcode and see their MP, the issues that MP speaks about and how that mix has changed, a verbatim summary of what the MP said on any issue (every sentence carries a Hansard citation), an estimate of how directly ministers answer Question Time questions, and the pledges they made."),
  p("The system is built on 370,861 MP speeches (1998 to July 2025) and combines an LDA topic model (coherence 0.459), BM25 retrieval re-ranked by a cross-encoder, extractive summarisation, a directness classifier trained on 120 hand-labelled question-answer pairs, and a rule-based pledge detector. Topic discovery and retrieval work well. The directness estimate is much weaker than the poster we presented earlier implied: among opposition questions it separates evasive from non-evasive answers with about 68% accuracy against a 58% majority baseline on 64 pairs, so the app reports it as a screening signal with confidence intervals, never as a verdict on an individual. We also found and fixed a data-validity problem that had silently dropped 69 recently elected MPs. Section 8 lists what we would do next."),

  h1("1. Project objectives and scope"),
  h2("1.1 Problem and users"),
  p("Hansard is public, but it is decades of dense, unsearchable text. Most voters therefore judge their MP on headlines or advertising rather than on the record, and there is no easy way to track which MPs are raising the issues a voter cares about. Existing tools count votes or find keyword mentions; neither shows what an MP actually said, in context, with a way to check it. Our primary users are citizens who want to understand their own representative. Advocacy groups and journalists are secondary users who want to find and cite what MPs have said on an issue."),
  h2("1.2 Objectives"),
  ...bullets([
    "**O1.** Resolve a postcode to the current MP for that electorate.",
    "**O2.** Show what an MP talks about and how their focus has shifted over time.",
    "**O3.** Answer \"what has my MP said about X?\" with verbatim passages and a citation for every sentence shown.",
    "**O4.** Estimate how directly answers address the question asked at Question Time, and surface the pledges ministers make.",
    "**O5.** Evaluate every component against labelled data, report results honestly, and state limits in the interface.",
    "**O6.** Deliver the above as a usable web application.",
  ]),
  h2("1.3 Scope and relationship to earlier work"),
  p("**In scope:** House of Representatives only; 1998 to 31 July 2025 (Question Time analysis 2011 to 2025, see Section 3.4); English text; current-electorate postcode lookup; extractive (not generated) summaries. **Out of scope:** the Senate, voting records, committee evidence, live updates, a user study with members of the public, and public deployment."),
  p("This report builds on the Idea Napkin proposal and the Part A poster (group mark 84/100). The poster pitched a \"dodge index\" and listed evaluation methods such as inter-annotator agreement and a cross-party bias audit. In this report we state which of those were actually carried out, and where the evidence is weaker than the poster suggested we say so."),

  h1("2. Project phases and contributions"),
  table(["Phase", "What was done", "Main outputs"], [
    ["1. Ideation", "Problem framing, dataset choice, Idea Napkin on Miro, instructor feedback", "Napkin; project scope"],
    ["2. Exploration and phase-1 analysis", "Cleaning, postcode join, LDA, zero-shot issue classification, RAG retrieval pilot for one MP", "Notebook (parlianote_analysis.ipynb); insights document"],
    ["3. Poster (AT2 Part A)", "Communicating goals, techniques and findings", "Group poster (84/100)"],
    ["4. Question-answer analysis", "Pairing questions with answers, labelling, directness models, pledge detection", "src/ pipeline; labelled sets; results tables"],
    ["5. Web application", "Hybrid retrieval, topic views, directness and pledge views, citation design", "app/app.py (Streamlit)"],
    ["6. Evaluation and report", "Cross-validation, retrieval comparison, data-quality audit, this report", "results/ folder; this document"],
  ], [2.2, 5.2, 3]),
  L.caption("Table 1. Project phases."),
  p("**Contributions by team member.** Each member's contribution is summarised in Table 2; the team's self and peer ratings are submitted separately on SparkPlus."),
  table(["Team member", "Contribution"], [
    ["Krishna Kumar Thirugnanavel", "[to be completed by the team]"],
    ["Sidharth Joly", "[to be completed by the team]"],
    ["Brian Shimmer", "[to be completed by the team]"],
    ["Malavika Rajesh", "[to be completed by the team]"],
    ["Prathamesh Nemade", "[to be completed by the team]"],
    ["Milad Kianfar", "[to be completed by the team]"],
  ], [3, 7.2]),
  L.caption("Table 2. Individual contributions (to be completed by the team before submission)."),
  p("**Declaration of AI assistance.** We used an AI coding assistant (Claude Code, Anthropic) to help write and debug the code in this project, and the same assistant produced the single-annotator labels for the 120 question-answer pairs and 90 sentences described in Section 4. The team is responsible for the content of this report. Because those labels were not produced by independent human annotators, Section 5 treats the evaluation as preliminary."),

  h1("3. Data sources"),
  h2("3.1 Hansard corpus"),
  p("Katz and Alexander (2025) parsed every sitting day of the House of Representatives from 2 March 1998 to 31 July 2025 from the official XML into a single Parquet file (579.6 MB, 647,852 rows). Each row is one utterance with the speaker, electorate, party, date, page number, speech number and body text, plus flags for questions, answers, interjections and member validity. We keep rows that have an electorate, are not interjections, are not flagged as a non-member, and contain more than 30 characters, giving **370,861 speeches** from 1998 to 2025."),
  h2("3.2 Postcode lookup"),
  p("The Hansard data records electorate names but not postcodes. We joined it to the open Australian Postcodes dataset (Proctor, n.d.), which maps 3,165 postcodes to federal electorates. The current MP for an electorate is taken to be its most recent speaker in the corpus."),
  h2("3.3 Annotated data created by the team"),
  ...bullets([
    "**Directness labels:** 120 Question Time pairs sampled at random, about 10 per year from 2011 to 2025, labelled Direct (52), Partial (39) or Evasive (29) from the visible question and the first 110 words of the answer.",
    "**Pledge labels:** 90 sentences from ministerial answers (45 flagged by our rules, 45 not), labelled Specific commitment (12), Vague or hedged intention (25) or Not a commitment (53).",
    "**Topic and issue checks:** 40 random speeches graded for correct issue assignment (phase 1) and a set of retrieval queries judged for relevance (Section 5.4).",
  ]),
  h2("3.4 Data quality issues found and how we handled them"),
  table(["Issue", "Evidence", "Handling"], [
    ["Answer flags unreliable before 2011", "In 2010 only 287 answers are flagged against 1,208 questions; from 2011 the counts match (1,276 vs 1,496 in 2011)", "Question-answer analysis restricted to 2011-2025 (16,256 pairs); earlier years excluded"],
    ["Member flag blank for MPs elected from 2022", "8,658 genuine speeches from 69 MPs (2022-2025) were removed by our first filter; the Brisbane postcode returned the previous MP", "Filter changed to exclude only member = 0; corpus grew from 362,203 to 370,861 speeches"],
    ["Electorate redistributions", "Only 143 of 166 Hansard electorates (86%) appear in the current postcode data (e.g. Batman is now Cooper)", "Old seats stay searchable by MP name, not by postcode"],
    ["Postcodes spanning electorates", "A postcode can map to several electorates", "App asks the user to choose and links to the AEC for confirmation"],
    ["Corpus ends 31 July 2025", "Newest speaker may not be the current MP", "App warns when an electorate's latest speaker spoke before 2025"],
  ], [2.6, 4.6, 3.6]),
  L.caption("Table 3. Data quality issues, evidence and handling."),
];
