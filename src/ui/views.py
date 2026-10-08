"""The app's screens. Each function draws one screen; app/app.py decides which to show.

Screens for a single MP take the MP's key (`mp_key`), a one-row `info` table row (name, party, ...) and/or their speeches `sp`.
"""

import altair as alt
import pandas as pd
import streamlit as st

from src import global_search as G
from src import retrieval as R
from src.commitment import commitment_type, split_sentences
from src.topics import TOPIC_NAMES
from src.ui import components as C
from src.ui.config import EXAMPLE_TOPICS, HANSARD_HOME, NO_MATCH_TIPS
from src.ui.data import cite, load_global, load_models, load_pairs, wilson
from src.ui.search import clean_query, key_sentences, retrieve, tidy_query, usable_query

esc = C.esc


# --------------------------------------------------------------------------------------------- search across everyone
def search_all():
    """'Search all MPs': type a topic or question, see who spoke about it and exactly what they said."""
    st.subheader("Search all MPs")
    st.caption(
        "Ask a question or type a topic. We find the passages that best match, across every MP since 1998, and show who said them. "
        "Every sentence is copied word for word from the official record."
    )
    q = tidy_query(
        st.text_input(
            "Your question or topic", placeholder="e.g. what has been said about housing affordability?", key="global_q", max_chars=300
        )
    )
    years = st.slider("Years", 1998, 2025, (1998, 2025), key="global_years")
    if not q:
        C.how_it_works()
        C.example_buttons("Try one of these:", EXAMPLE_TOPICS, "ex_topic", "global_q", fmt=lambda t: t.capitalize())
        return
    if not usable_query(q):
        st.warning("Please type a few words, for example a topic like *aged care* or a question like *who has spoken about drought?*")
        return

    topic = clean_query(q)
    with st.spinner("Searching every speech..."):
        hits, candidates = load_global().search(topic, load_models()[1], year_range=years, k_speeches=40, k_final=8)
    quality = "none" if hits.empty else G.match_quality(float(candidates["tfidf"].max()), float(hits["rerank"].head(3).mean()))
    if quality == "none":
        # Absence from this record does not prove nobody spoke about it, so we say so instead of "nobody did".
        st.warning(
            f'We couldn\'t find anything that clearly matches "{q}". That does not prove nobody has spoken about it. ' + NO_MATCH_TIPS
        )
        return
    st.query_params.update({"mode": "Search all MPs", "q": q})  # makes the address bar a shareable link
    if quality == "weak":
        st.warning("These are the closest matches we found, but none of them answers your question directly. Treat them with caution.")

    C.ai_answer_box("all", q, hits, speaker_of=lambda r: r["displayName"])
    st.markdown("#### Who spoke about this")
    for r in G.who_spoke(hits).itertuples():
        st.markdown(f"**{r.displayName}** ({r.party}, {r.electorate}): {r.passages} of the {min(20, len(hits))} best-matching passages")
    st.markdown("#### What was said")
    st.caption("The sentences closest to your question, each with who said it. Nothing is written by the computer.")
    for i, sentence in R.extractive_summary(load_models()[0], topic, hits.head(10), n=3):
        r = hits.loc[i]
        C.quote_card(sentence, f"<b>{esc(r['displayName'])}</b> ({esc(r['party'])}, {esc(r['electorate'])}) &middot; {esc(cite(r))}")
    st.markdown("#### The passages behind it")
    for _, r in hits.head(8).iterrows():
        with st.expander(f"{r['displayName']} ({r['party']}, {r['electorate']})  |  {pd.Timestamp(r['date']):%d %b %Y}"):
            st.write(r["text"])
            st.markdown(C.cite_line(r), unsafe_allow_html=True)
    st.download_button(
        "Download these passages (CSV)",
        C.results_csv(hits.head(8), speaker_col="displayName"),
        file_name="parlianote_search.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------------------------- one MP
def overview(sp: pd.DataFrame):
    """Headline numbers, speeches per year, subjects talked about, and how the subject mix changed over time."""
    topics = sp[sp["topic"] >= 0].copy()
    topics["Topic"] = topics["topic"].map(TOPIC_NAMES)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Speeches recorded", f"{len(sp):,}")
    k2.metric("Years in Parliament (in our data)", f"{sp['year'].nunique()}")
    k3.metric("Busiest year", f"{int(sp.groupby('year').size().idxmax())}")
    top_topic = esc(topics["Topic"].mode().iloc[0]) if not topics.empty else "Not enough data"
    k4.markdown(  # st.metric cuts long text off, so draw this one by hand
        f"<div style='font-size:0.85rem;color:#5b6b7b'>Most common subject</div>"
        f"<div style='font-size:1.15rem;font-weight:600;color:#16304f;line-height:1.3'>{top_topic}</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Speeches per year")
        per_year = sp.groupby("year").size().reset_index(name="speeches")
        st.altair_chart(
            alt.Chart(per_year)
            .mark_bar(color="#16304f")
            .encode(x=alt.X("year:O", title=None), y=alt.Y("speeches:Q", title="Speeches"), tooltip=["year", "speeches"])
            .properties(height=300),
            use_container_width=True,
        )
    with right:
        st.subheader("What they talk about")
        if topics.empty:
            st.write("This MP has too few longer speeches for us to work out subjects.")
        else:
            mix = topics.groupby("Topic").size().reset_index(name="speeches").sort_values("speeches", ascending=False)
            mix["share"] = mix["speeches"] / mix["speeches"].sum()
            st.altair_chart(
                alt.Chart(mix)
                .mark_bar(color="#16304f")
                .encode(
                    y=alt.Y("Topic:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300, labelOverlap=False)),
                    x=alt.X("share:Q", axis=alt.Axis(format="%"), title="Share of speeches"),
                    tooltip=["Topic", alt.Tooltip("share:Q", format=".0%"), "speeches"],
                )
                .properties(height=300),
                use_container_width=True,
            )
    if topics.empty:
        return
    st.subheader("How their focus has changed over time")
    by_year = topics.groupby(["year", "Topic"]).size().reset_index(name="speeches")
    by_year["share"] = by_year["speeches"] / by_year.groupby("year")["speeches"].transform("sum")
    st.altair_chart(
        alt.Chart(by_year)
        .mark_area()
        .encode(
            x=alt.X("year:O", title=None),
            y=alt.Y("share:Q", stack="normalize", axis=alt.Axis(format="%"), title="Share of speeches"),
            color=alt.Color("Topic:N", legend=alt.Legend(title=None, orient="bottom", columns=2)),
            tooltip=["year", "Topic", alt.Tooltip("share:Q", format=".0%")],
        )
        .properties(height=320),
        use_container_width=True,
    )
    st.caption(
        "Speeches are sorted into 10 subjects automatically by a computer model, and we named each subject from its most common words. "
        "The names are our interpretation and can be imperfect. Speeches under 80 words are not given a subject."
    )


def ask_the_record(mp_key: str, info, sp: pd.DataFrame):
    """Search one MP's speeches: key sentences, closest passages, early-versus-late wording, optional AI answer."""
    st.subheader("Ask the record")
    q = tidy_query(
        st.text_input(
            "What do you want to know about this MP's views?",
            placeholder="e.g. housing affordability, climate change, gambling ads",
            key="mp_q",
            max_chars=300,
        )
    )
    years = sorted(sp["year"].unique())
    year_range = None
    if len(years) > 1:
        year_range = st.slider("Years", int(min(years)), int(max(years)), (int(min(years)), int(max(years))), key=f"years_{mp_key}")
    if not q:
        C.example_buttons("Try one of these:", EXAMPLE_TOPICS[:5], "ex_mp_topic", "mp_q", fmt=lambda t: t.capitalize())
        return
    if not usable_query(q):
        st.warning("Please type a few words, for example a topic like *housing* or a question like *what do they think about tax?*")
        return

    topic = clean_query(q)
    with st.spinner("Searching their speeches..."):
        hits = retrieve(mp_key, topic, year_range=year_range)
    quality = "none" if hits.empty else R.match_quality(float(hits["rerank"].iloc[0]))
    if quality == "none":
        st.warning(
            f'We couldn\'t find anything {info["name"]} has said that clearly matches "{q}". That does not prove they never spoke about it. '
            + NO_MATCH_TIPS
            + ' You can also try "Search all MPs" to see whether anyone else has.'
        )
        return
    if quality == "weak":
        st.warning("These are the closest matches we found, but none of them clearly answers your question. Treat them with caution.")

    C.ai_answer_box(f"mp-{mp_key}", q, hits, speaker_of=lambda r: info["name"])
    st.markdown("#### Key sentences")
    st.caption("Each sentence is copied word for word from the official record. Nothing is written by the computer.")
    for i, sentence in key_sentences(topic, hits):
        C.quote_card(sentence, esc(cite(hits.loc[i])))
    st.markdown("#### Closest matching passages")
    for _, r in hits.head(5).iterrows():
        with st.expander(f"{pd.Timestamp(r['date']):%d %b %Y}  |  match score {r['rerank']:.1f}"):
            st.write(r["text"])
            st.markdown(C.cite_line(r), unsafe_allow_html=True)

    st.markdown("#### How the wording changed")
    hits = hits.assign(year=pd.to_datetime(hits["date"]).dt.year)
    strong = hits[hits["rerank"] >= hits["rerank"].median()].sort_values("year")  # compare only strong matches
    first, last = strong.iloc[0], strong.iloc[-1]
    if first["year"] == last["year"]:
        st.write("The best matches all come from the same year. Widen the years to compare how the wording changed.")
    else:
        a, b = st.columns(2)
        a.markdown(f"**{first['year']}**")
        a.write(first["text"])
        a.caption(cite(first))
        b.markdown(f"**{last['year']}**")
        b.write(last["text"])
        b.caption(cite(last))
    st.download_button(
        "Download these passages (CSV)",
        C.results_csv(hits.head(8), fixed_speaker=info["name"]),
        file_name="parlianote_mp_search.csv",
        mime="text/csv",
    )


def question_time(mp_key: str):
    """How directly this MP's answers (or the answers to their questions) address the question asked. Estimates only."""
    st.subheader("Question Time: do the answers address the question?")
    st.caption(
        "Question Time is the daily live session where MPs ask government ministers questions. This is a statistical estimate of how directly "
        "an answer responds to the question asked (2011 to 2025). It measures how the answer is worded, not what anyone intended, "
        "and it is never a verdict on a single answer."
    )
    pairs = load_pairs()
    pairs = pairs[~pairs["a_is_chair"]]  # questions put to the Speaker are rulings, not ministerial answers
    from_other_side = pairs[~pairs["same_bloc"]]  # friendly questions are almost always answered directly, so we compare only these
    average = (from_other_side["pred"] == "E").mean()
    answered = pairs[pairs["a_key"] == mp_key]
    asked = pairs[pairs["q_key"] == mp_key]

    if len(answered) >= 10:
        theirs = answered[~answered["same_bloc"]]
        st.markdown(
            f"**When answering:** {len(answered)} Question Time answers, {len(theirs)} of them to questions from the other side "
            "(opposition or crossbench MPs)."
        )
        if len(theirs) >= 10:
            k = int((theirs["pred"] == "E").sum())
            low, high = wilson(k, len(theirs))
            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Answers that don't directly address the question",
                f"{k / len(theirs):.0%}",
                help=f"95% range {low:.0%} to {high:.0%}. Questions from the other side only.",
            )
            m2.metric("Average for all ministers", f"{average:.0%}")
            m3.metric("Average score (0 = direct, 1 = not direct)", f"{theirs['p_E'].mean():.2f}")
            st.caption(
                f"We are 95% confident the true figure for this MP is between {low:.0%} and {high:.0%} (based on {len(theirs)} answers). "
                "Differences of a few points are within the model's margin of error."
            )
            st.markdown("**Exchanges scored as least direct** (read them yourself):")
            for _, r in theirs.sort_values("p_E", ascending=False).head(3).iterrows():
                with st.expander(f"{pd.Timestamp(r['date']):%d %b %Y}: question from {r['q_name']}  (score {r['p_E']:.2f})"):
                    st.markdown("**Question**")
                    st.write(r["q_text"])
                    st.markdown("**Answer**")
                    st.write(r["a_text"][:1500] + (" ..." if len(r["a_text"]) > 1500 else ""))
                    page = int(r["a_page"]) if pd.notna(r["a_page"]) else "-"
                    st.caption(f"House of Representatives Hansard, {pd.Timestamp(r['date']):%d %b %Y}, p. {page}")

    if len(asked) >= 10:
        theirs = asked[~asked["same_bloc"]]
        st.markdown(f"**When asking:** {len(asked)} questions, {len(theirs)} of them to ministers from the other side.")
        if len(theirs) >= 10:
            k = int((theirs["pred"] == "E").sum())
            low, high = wilson(k, len(theirs))
            st.metric(
                "Answers to their questions that don't directly address them",
                f"{k / len(theirs):.0%}",
                help=f"95% range {low:.0%} to {high:.0%}",
            )
            st.caption(f"Average across all questions from the other side: {average:.0%}. 95% range for this MP: {low:.0%} to {high:.0%}.")

    if len(answered) < 10 and len(asked) < 10:
        st.info(
            "This MP has fewer than 10 recorded Question Time exchanges, which is too few for a reliable estimate. "
            "Question and answer labels in the source data are only reliable from 2011."
        )


def promises(mp_key: str):
    """Sentences in this MP's Question Time answers where they promise something, split into specific and general."""
    st.subheader("Promises made in Question Time answers")
    st.caption(
        "Sentences where the speaker says they or their government will do something. "
        "A promise is 'specific' when it also contains a number, date or dollar amount that can be checked later."
    )
    answered = load_pairs()
    answered = answered[answered["a_key"] == mp_key]
    if len(answered) < 10:
        st.info("We look for promises in the answers ministers give, and this MP has too few recorded answers.")
        return
    rows = []
    for r in answered.itertuples():
        for sentence in split_sentences(r.a_text):
            kind = commitment_type(sentence)
            if kind != "none":
                rows.append({"date": r.date, "type": kind, "sentence": sentence, "page": r.a_page})
    found = pd.DataFrame(rows)
    if found.empty:
        st.write("We did not find any sentences that look like promises.")
        return
    c1, c2 = st.columns(2)
    c1.metric("Specific promises (with a number, date or amount)", int((found["type"] == "specific pledge").sum()))
    c2.metric("General promises", int((found["type"] == "general pledge").sum()))
    labels = {"specific pledge": "Specific promises", "general pledge": "General promises"}
    chosen = st.radio("Show", list(labels), format_func=labels.get, horizontal=True)
    for _, r in found[found["type"] == chosen].sort_values("date", ascending=False).head(15).iterrows():
        page = int(r["page"]) if pd.notna(r["page"]) else "-"
        C.quote_card(r["sentence"], f"House of Representatives Hansard, {pd.Timestamp(r['date']):%d %b %Y}, p. {page}")
    st.caption(
        'Promises are found with simple pattern rules (for example "we will ..."), so some sentences will be missed or wrongly included.'
    )


def about():
    """What the tool is, how it works, and its limits, in plain language."""
    st.subheader("What this tool is, and is not")
    st.markdown(
        f"""
**What it is.** A way to see what Australian MPs have said in the House of Representatives, the main chamber of federal Parliament, from March 1998 to July 2025.
The words come from Hansard, the official written record of Parliament, as organised by Katz & Alexander (2025). Please check any quote against the official record at [aph.gov.au]({HANSARD_HOME}).

**How your MP is found.** Your postcode is matched to an electorate (the local area an MP represents) using an open postcode list (Matthew Proctor).
The most recent speaker recorded for an area is treated as its MP, so check areas where the MP may have changed.

**How it works, in brief.**
- *Topics:* a statistical model sorts speeches into 10 subjects automatically.
- *Search:* keyword matching finds candidate passages, then a second model re-reads the best ones and scores how well each answers your question.
- *Key sentences:* only sentences copied from the record, never written by the computer.
- *AI short answers (optional):* if switched on, an AI model reads only the best-matching passages and writes a short answer that cites them. Citations and figures are checked automatically, but the AI can still make mistakes.
- *Directness of answers:* a statistical estimate built from 120 question-and-answer pairs that we labelled by hand.
- *Promises:* sentences matched by simple rules such as "we will ...".

**What to keep in mind.**
- If a search finds nothing relevant, that does not prove nobody spoke about it. Only the House of Representatives is covered, only up to July 2025, and people may have used different words.
- The directness estimate is rough: among questions from the other side it is right about 68% of the time when separating "not direct" from "direct or partly direct" (guessing would give 58%). It was labelled by one person, so treat it as a screening signal, not a verdict.
- Questions from MPs on the same side as the minister are almost always answered directly, so scores only compare questions from the other side. The Liberal and National parties count as one side. Answers by the Speaker are excluded.
- Question and answer labels in the source data are only reliable from 2011, so earlier years are not included in the Question Time and Promises views.
- Subject names can be wrong, and speeches under 80 words are not given a subject.
- This describes the wording of the public record. It says nothing about what anyone intended or believes.
"""
    )
