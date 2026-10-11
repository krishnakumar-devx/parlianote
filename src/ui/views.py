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
from src.blocs import bloc
from src.ui.data import cite, load_global, load_minister_directness, load_models, load_pairs, load_speeches, load_topic_counts, wilson
from src.ui.search import clean_query, key_sentences, retrieve, tidy_query, usable_query

esc = C.esc
INK, LINE, MUTED = "#1f5c3f", "#d3ddd1", "#5e7266"
TOPIC_COLOURS = ["#1f5c3f", "#b8860b", "#3a6ea5", "#c0504d", "#8fb48a", "#8e6c8a", "#d4a373", "#6b9ac4", "#9c9c9c", "#2a9d8f"]
TOPIC_SCALE = alt.Scale(domain=sorted(TOPIC_NAMES.values()), range=TOPIC_COLOURS)  # each subject keeps its colour on every chart
GOLD = "#b8860b"


def _style(chart):
    """The app's quiet chart style: no frame, no gridlines, muted axes."""
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(grid=False, domainColor=LINE, labelColor=MUTED, tickColor=LINE, titleColor=MUTED)
        .configure_legend(labelColor="#1f2a24", labelLimit=260)
    )


def _topic_area(by_year: pd.DataFrame, height: int = 340):
    """Stacked area of each subject's share of speeches per year. `by_year` has year, Topic and speeches columns."""
    by_year = by_year.assign(share=by_year["speeches"] / by_year.groupby("year")["speeches"].transform("sum"))
    st.altair_chart(
        _style(
            alt.Chart(by_year)
            .mark_area()
            .encode(
                x=alt.X("year:O", title=None),
                y=alt.Y("share:Q", stack="normalize", axis=alt.Axis(format="%"), title="Share of speeches"),
                color=alt.Color("Topic:N", scale=TOPIC_SCALE, legend=alt.Legend(title=None, orient="right", labelLimit=240)),
                tooltip=["year", "Topic", alt.Tooltip("share:Q", format=".0%"), "speeches"],
            )
            .properties(height=height, padding={"left": 8, "top": 8, "right": 8, "bottom": 8})
        ),
        use_container_width=True,
    )


# --------------------------------------------------------------------------------------------- search across everyone
def search_all():
    """'Search all MPs': type a topic or question, see who spoke about it and exactly what they said."""
    with st.container(border=True):
        q = tidy_query(
            st.text_input(
                "Ask a question or type a topic",
                placeholder="e.g. what has been said about housing affordability?",
                key="global_q",
                max_chars=300,
            )
        )
        with st.expander("Filter by year"):
            years = st.slider("Years", 1998, 2025, (1998, 2025), key="global_years")
    if not q:
        C.how_it_works(C.SEARCH_STEPS)
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
    C.section("What was said", "The three sentences closest to your question, copied word for word from the record.")
    for i, sentence in R.extractive_summary(load_models()[0], topic, hits.head(10), n=3):
        r = hits.loc[i]
        C.quote_card(
            sentence,
            f"<b>{esc(C.display_name(r['displayName']))}</b> ({esc(r['party'])}, {esc(r['electorate'])}) &middot; {esc(cite(r))}",
        )
    total = min(20, len(hits))
    C.section("Who spoke about this", f"How many of the {total} best-matching passages came from each MP.")
    C.speaker_rows(G.who_spoke(hits), total)
    C.section("The passages behind it", "Open one to read the full passage and its source.")
    for _, r in hits.head(8).iterrows():
        with st.expander(f"{C.display_name(r['displayName'])} · {C.party_label(r['party'])} · {pd.Timestamp(r['date']):%d %b %Y}"):
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

    C.stat_cards(
        [
            ("Speeches recorded", f"{len(sp):,}"),
            ("Years active", sp["year"].nunique()),
            ("Busiest year", int(sp.groupby("year").size().idxmax())),
            ("Most common subject", topics["Topic"].mode().iloc[0] if not topics.empty else "Not enough data"),
        ]
    )

    left, right = st.columns(2, gap="large")
    with left:
        C.section("Speeches per year")
        per_year = sp.groupby("year").size().reset_index(name="speeches")
        st.altair_chart(
            alt.Chart(per_year)
            .mark_bar(color=INK, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(x=alt.X("year:O", title=None), y=alt.Y("speeches:Q", title=None), tooltip=["year", "speeches"])
            .properties(height=280)
            .configure_view(strokeWidth=0)
            .configure_axis(grid=False, domainColor=LINE, labelColor=MUTED, tickColor=LINE),
            use_container_width=True,
        )
    with right:
        C.section("What they talk about")
        if topics.empty:
            st.write("This MP has too few longer speeches for us to work out subjects.")
        else:
            mix = topics.groupby("Topic").size().reset_index(name="speeches").sort_values("speeches", ascending=False)
            mix["share"] = mix["speeches"] / mix["speeches"].sum()
            st.altair_chart(
                alt.Chart(mix)
                .mark_bar(color=INK, cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
                .encode(
                    y=alt.Y("Topic:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300, labelOverlap=False)),
                    x=alt.X("share:Q", axis=alt.Axis(format="%"), title=None),
                    tooltip=["Topic", alt.Tooltip("share:Q", format=".0%"), "speeches"],
                )
                .properties(height=280)
                .configure_view(strokeWidth=0)
                .configure_axis(grid=False, domainColor=LINE, labelColor=MUTED, tickColor=LINE),
                use_container_width=True,
            )
    if topics.empty:
        return
    C.section("How their focus has changed", "Each band is one subject's share of their speeches in that year.")
    _topic_area(topics.groupby(["year", "Topic"]).size().reset_index(name="speeches"))
    st.caption(
        "Subjects are found automatically by a computer model and named by us from their most common words, so the names can be imperfect. "
        "Speeches under 80 words are not given a subject."
    )


def ask_the_record(mp_key: str, info, sp: pd.DataFrame):
    """Search one MP's speeches: key sentences, closest passages, early-versus-late wording, optional AI answer."""
    years = sorted(sp["year"].unique())
    year_range = None
    with st.container(border=True):
        q = tidy_query(
            st.text_input(
                f"What has {C.display_name(info['name'])} said about...",
                placeholder="e.g. housing affordability, climate change, gambling ads",
                key="mp_q",
                max_chars=300,
            )
        )
        if len(years) > 1:
            with st.expander("Filter by year"):
                year_range = st.slider(
                    "Years", int(min(years)), int(max(years)), (int(min(years)), int(max(years))), key=f"years_{mp_key}"
                )
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
    C.section("Key sentences", "Copied word for word from the official record. Nothing here is written by a computer.")
    for i, sentence in key_sentences(topic, hits):
        C.quote_card(sentence, esc(cite(hits.loc[i])))
    C.section("Closest matching passages", "Open one to read the full passage and its source.")
    for _, r in hits.head(5).iterrows():
        with st.expander(f"{pd.Timestamp(r['date']):%d %b %Y} · match score {r['rerank']:.1f}"):
            st.write(r["text"])
            st.markdown(C.cite_line(r), unsafe_allow_html=True)

    C.section("How the wording changed", "Their earliest and latest strong match on this topic, side by side.")
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
    C.section(
        "Do the answers address the question?",
        "Question Time is the daily session where MPs question ministers. This is a rough statistical estimate (2011 to 2025) "
        "of how directly answers respond, based on wording only. It is never a verdict on a single answer.",
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
    C.section(
        "Promises made in Question Time answers",
        "Sentences like &ldquo;we will ...&rdquo;. A promise is <b>specific</b> when it includes a number, date or dollar amount "
        "you could check later.",
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
    n_specific = int((found["type"] == "specific pledge").sum())
    n_general = int((found["type"] == "general pledge").sum())
    labels = {"specific pledge": f"Specific ({n_specific})", "general pledge": f"General ({n_general})"}
    chosen = st.radio("Show", list(labels), format_func=labels.get, horizontal=True, label_visibility="collapsed")
    for _, r in found[found["type"] == chosen].sort_values("date", ascending=False).head(15).iterrows():
        page = int(r["page"]) if pd.notna(r["page"]) else "-"
        C.quote_card(r["sentence"], f"House of Representatives Hansard, {pd.Timestamp(r['date']):%d %b %Y}, p. {page}")
    st.caption(
        'Promises are found with simple pattern rules (for example "we will ..."), so some sentences will be missed or wrongly included.'
    )


def about():
    """What the tool is, how it works, and its limits, in plain language."""
    C.section("What this tool is, and is not")
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


# --------------------------------------------------------------------------------------------- the whole House
TREND_VIEWS = ["Topics over time", "Question Time across ministers"]
BLOCS = ["Everyone", "Labor", "Coalition", "Other"]


def parliament_trends():
    """'Parliament trends': what the whole House talks about over time, and Question Time directness across ministers."""
    st.query_params.update({"mode": "Parliament trends"})
    st.radio("Trend", TREND_VIEWS, horizontal=True, label_visibility="collapsed", key="trend_view")
    if st.session_state["trend_view"] == TREND_VIEWS[0]:
        _topic_trends()
    else:
        _minister_comparison()


def _topic_trends():
    counts = load_topic_counts().assign(Topic=lambda d: d["topic"].map(TOPIC_NAMES), Bloc=lambda d: d["party"].map(bloc))
    C.section(
        "What the House of Representatives talks about",
        "Each band is one subject's share of all speeches in that year. Choose a side of politics to see only their MPs.",
    )
    who = st.radio("Whose speeches", BLOCS, horizontal=True, label_visibility="collapsed", key="trend_bloc")
    if who != "Everyone":
        counts = counts[counts["Bloc"] == who]
    if counts.empty:
        st.info("No speeches with a subject for this group.")
        return
    by_year = counts.groupby(["year", "Topic"])["speeches"].sum().reset_index()

    # biggest changes between the first and last five years of the record
    def share(years):
        part = by_year[by_year["year"].isin(years)].groupby("Topic")["speeches"].sum()
        return part / part.sum()

    change = (share(range(2021, 2026)).sub(share(range(1998, 2003)), fill_value=0) * 100).sort_values()
    C.stat_cards(
        [
            (f"Grew the most · {change.iloc[-1]:+.1f} pts", change.index[-1]),
            (f"Grew second most · {change.iloc[-2]:+.1f} pts", change.index[-2]),
            (f"Shrank the most · {change.iloc[0]:+.1f} pts", change.index[0]),
            (f"Shrank second most · {change.iloc[1]:+.1f} pts", change.index[1]),
        ]
    )
    C.hint("Change in share of speeches, 1998&ndash;2002 compared with 2021&ndash;2025 (pts = percentage points).")
    _topic_area(by_year, height=380)
    st.caption(
        f"Based on {int(by_year['speeches'].sum()):,} speeches of 80 words or more. Subjects are found automatically by a computer model "
        "and named by us from their most common words, so the names can be imperfect. 2025 covers January to July only."
    )


def _minister_comparison():
    table, average = load_minister_directness()
    C.section(
        "How directly ministers' answers address the question",
        "For each MP with at least 40 Question Time answers to the other side: the share of those answers our model scored as "
        "<b>not directly addressing the question</b>, with a 95% range.",
    )
    C.notice(
        "Read this as a rough screening signal, not a ranking of honesty. The score is based on wording only, is right about 68% of the "
        "time, and was trained on 120 hand-labelled answers. Ministers whose ranges overlap cannot be told apart, and the kind of "
        "questions a portfolio attracts affects the score."
    )
    if table.empty:
        st.info("Not enough Question Time answers to compare ministers.")
        return
    order = st.radio("Order", ["Sort by name", "Sort by estimate"], horizontal=True, label_visibility="collapsed", key="minister_sort")
    table = table.assign(
        label=table["name"].map(C.display_name) + " (" + table["party"] + ")",
        bloc=table["party"].map(bloc),
    )
    table = table.sort_values("share" if order == "Sort by estimate" else "name", ascending=order != "Sort by estimate")
    sort = table["label"].tolist()
    base = alt.Chart(table).encode(y=alt.Y("label:N", sort=sort, title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False)))
    tooltip = [
        alt.Tooltip("label:N", title="MP"),
        alt.Tooltip("answers:Q", title="Answers to the other side"),
        alt.Tooltip("share:Q", title="Not direct", format=".0%"),
        alt.Tooltip("low:Q", title="95% range from", format=".0%"),
        alt.Tooltip("high:Q", title="to", format=".0%"),
        alt.Tooltip("first:Q", title="First year", format="d"),
        alt.Tooltip("last:Q", title="Last year", format="d"),
    ]
    colours = alt.Scale(domain=["Labor", "Coalition", "Other"], range=[INK, GOLD, "#6b9ac4"])
    ranges = base.mark_rule(strokeWidth=2.5, opacity=0.45).encode(
        x=alt.X("low:Q", axis=alt.Axis(format="%"), title="Answers scored as not directly addressing the question", scale=alt.Scale(domain=[0, 1])),
        x2="high:Q",
        color=alt.Color("bloc:N", scale=colours, legend=alt.Legend(title=None, orient="top")),
        tooltip=tooltip,
    )
    points = base.mark_circle(size=70, opacity=1).encode(x="share:Q", color=alt.Color("bloc:N", scale=colours), tooltip=tooltip)
    avg = alt.Chart(pd.DataFrame({"x": [average]})).mark_rule(strokeDash=[4, 4], color=MUTED).encode(x="x:Q")
    st.altair_chart(_style((ranges + points + avg).properties(height=max(260, 26 * len(table)))), use_container_width=True)

    above = int((table["low"] > average).sum())
    below = int((table["high"] < average).sum())
    st.caption(
        f"Dashed line: the average for all ministers ({average:.0%}). Of these {len(table)} MPs, the whole range of {above} sits above "
        f"the average and of {below} below it; the other {len(table) - above - below} cannot be told apart from the average. "
        "Questions from the minister's own side and questions to the Speaker are left out. Question Time data covers 2011 to 2025."
    )
    with st.expander("Show as a table"):
        st.dataframe(
            table[["label", "answers", "share", "low", "high", "first", "last"]].rename(
                columns={"label": "MP", "answers": "Answers", "share": "Not direct", "low": "95% from", "high": "95% to",
                         "first": "From", "last": "To"}
            ),
            hide_index=True,
            use_container_width=True,
            column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ["Not direct", "95% from", "95% to"]},
        )


# --------------------------------------------------------------------------------------------- two MPs side by side
CHOOSE = "(choose an MP)"
COMPARE_STEPS = [
    ("Choose two MPs", "For example your MP and someone from another party or area."),
    ("Compare their subjects", "See which topics each of them speaks about most."),
    ("Ask the same question", "Read what each of them said on it, in their own words."),
]


def compare_mps(mps: pd.DataFrame):
    """'Compare MPs': two MPs side by side, by subject mix and by what each said on the same question."""
    names = [CHOOSE] + mps["name"].tolist()
    with st.container(border=True):
        col_a, col_b = st.columns(2, gap="large")
        pick_a = col_a.selectbox("First MP", names, key="cmp_a")
        pick_b = col_b.selectbox("Second MP", names, key="cmp_b")
    if CHOOSE in (pick_a, pick_b):
        C.how_it_works(COMPARE_STEPS)
        return
    if pick_a == pick_b:
        st.warning("You picked the same MP twice. Choose two different MPs to compare.")
        return
    st.query_params.update({"mode": "Compare MPs", "a": pick_a, "b": pick_b})

    people = [mps[mps["name"] == pick].iloc[0] for pick in (pick_a, pick_b)]
    for col, info in zip(st.columns(2, gap="medium"), people):
        with col:
            C.mp_header(info)

    # subject mix
    C.section("What they talk about", "Each subject's share of their speeches, for the years each MP is in our data.")
    rows = []
    for info in people:
        sp = load_speeches(info["mp_key"])
        topics = sp[sp["topic"] >= 0]
        if len(topics):
            share = topics["topic"].map(TOPIC_NAMES).value_counts(normalize=True)
            rows += [{"MP": C.display_name(info["name"]), "Topic": t, "share": v} for t, v in share.items()]
    if rows:
        mix = pd.DataFrame(rows)
        order = mix.groupby("Topic")["share"].sum().sort_values(ascending=False).index.tolist()
        st.altair_chart(
            _style(
                alt.Chart(mix)
                .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
                .encode(
                    y=alt.Y("Topic:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
                    yOffset=alt.YOffset("MP:N", sort=[C.display_name(i["name"]) for i in people]),
                    x=alt.X("share:Q", axis=alt.Axis(format="%"), title=None),
                    color=alt.Color(
                        "MP:N", scale=alt.Scale(domain=[C.display_name(i["name"]) for i in people], range=[INK, GOLD]),
                        legend=alt.Legend(title=None, orient="top"),
                    ),
                    tooltip=["MP", "Topic", alt.Tooltip("share:Q", format=".0%")],
                )
                .properties(height=440)
            ),
            use_container_width=True,
        )
    else:
        st.write("Neither MP has enough longer speeches for us to work out subjects.")

    # the same question to both
    C.section("What did they each say about...", "Type a topic; we search each MP's speeches separately and show their closest sentences.")
    q = tidy_query(st.text_input("Topic or question for both MPs", placeholder="e.g. climate change", key="cmp_q", max_chars=300))
    if not q:
        C.example_buttons("Try one of these:", EXAMPLE_TOPICS[:5], "ex_cmp_topic", "cmp_q", fmt=lambda t: t.capitalize())
        return
    if not usable_query(q):
        st.warning("Please type a few words, for example a topic like *housing* or *climate change*.")
        return
    topic = clean_query(q)
    for col, info in zip(st.columns(2, gap="large"), people):
        with col:
            name = C.display_name(info["name"])
            st.markdown(f"**{esc(name)}**")
            with st.spinner(f"Searching {name}'s speeches..."):
                hits = retrieve(info["mp_key"], topic)
            quality = "none" if hits.empty else R.match_quality(float(hits["rerank"].iloc[0]))
            if quality == "none":
                st.info(f'Nothing {name} said clearly matches "{q}". That does not prove they never spoke about it.')
                continue
            if quality == "weak":
                st.caption("Closest matches only; none clearly answers the question.")
            for i, sentence in key_sentences(topic, hits)[:2]:
                C.quote_card(sentence, esc(cite(hits.loc[i])))
