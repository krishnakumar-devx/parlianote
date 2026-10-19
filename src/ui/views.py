"""The app's screens. Each function draws one screen; app/app.py decides which to show.

Screens for a single MP take the MP's key (`mp_key`), a one-row `info` table row (name, party, ...) and/or their speeches `sp`.
"""

import json
import random
import re

import altair as alt
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

from src import global_search as G
from src import retrieval as R
from src.blocs import bloc
from src.commitment import FIRM, GUARANTEE, commitment_type, split_sentences
from src.topics import TOPIC_NAMES
from src.ui import components as C
from src.ui import quotes
from src.ui.config import BLOC_COLOURS, DATA, EXAMPLE_TOPICS, HANSARD_HOME, NO_MATCH_TIPS, ROOT
from src.ui.data import cite, load_global, load_minister_directness, load_models, load_pairs, load_speeches, load_topic_counts, wilson
from src.ui.search import clean_query, key_sentences, retrieve, tidy_query, usable_query
from src.ui.seatmap import seat_map_pair

esc = C.esc
INK, LINE, MUTED = "#16140f", "#d9d4c7", "#6b665c"
HL_SOFT = "#fff1a8"  # the soft highlighter (--hl-soft in components.py), for areas
# Muted, print-like colours, one per subject (in alphabetical order of subject name), so each keeps its colour on every chart.
# The largest subject, Partisan debate, gets a quiet stone grey so it does not dominate the stacked charts.
TOPIC_COLOURS = ["#d8a47f", "#556b3f", "#c9a227", "#5c8a99", "#b5523b", "#8fb08f", "#8c6d9e", "#b8b2a6", "#34587a", "#6e3b4e"]
TOPIC_SCALE = alt.Scale(domain=sorted(TOPIC_NAMES.values()), range=TOPIC_COLOURS)
OCHRE = "#c9a227"  # second series when two MPs are compared (not a party colour)


def _style(chart):
    """The app's quiet, print-like chart style: Inter type, no frame, no gridlines, an ink baseline."""
    return (
        chart.configure(font="Inter")
        .configure_view(strokeWidth=0)
        .configure_axis(
            grid=False, domainColor=INK, labelColor=MUTED, tickColor=LINE, titleColor=MUTED, labelFontSize=11, titleFontWeight=500
        )
        .configure_legend(labelColor=INK, labelLimit=260, labelFontSize=12, symbolType="square")
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


MIN_YEAR_SPEECHES = 10  # a year with fewer speeches that have a subject is too thin for shares (one speech would be 10% or more)


def focus_rows(topics: pd.DataFrame) -> pd.DataFrame | None:
    """Each subject's share of an MP's speeches per year, for "How their focus has changed"; None if under 2 usable years.

    `topics` has one row per speech with year and Topic. Every subject they spoke on gets every year, so a year with nothing on
    a subject counts as 0% (rows missing for those years are what broke the old stacked chart). Years with fewer than
    MIN_YEAR_SPEECHES speeches are marked thin and get no share. Rows are numbered from the bottom: the biggest subject is on top.
    """
    counts = topics.groupby(["year", "Topic"]).size().unstack(fill_value=0)
    counts = counts.reindex(range(int(counts.index.min()), int(counts.index.max()) + 1), fill_value=0)
    counts.index.name = "year"
    per_year = counts.sum(axis=1)
    if (per_year >= MIN_YEAR_SPEECHES).sum() < 2:
        return None
    overall = (counts.sum() / counts.values.sum()).sort_values(kind="stable")  # smallest first, so it gets the bottom row
    df = counts.stack().rename("speeches").reset_index()
    df["n"] = df["year"].map(per_year)
    df["thin"] = df["n"] < MIN_YEAR_SPEECHES
    df["share"] = (df["speeches"] / df["n"]).where(~df["thin"])
    df["row"] = df["Topic"].map({topic: i for i, topic in enumerate(overall.index)})
    df["label"] = df["Topic"].map(lambda topic: f"{topic}  {overall[topic]:.0%}")
    usable = df[~df["thin"] & (df["share"] > 0)]
    df["peak"] = df.index.isin(usable.groupby("Topic")["share"].idxmax())
    return df


def _focus_chart(df: pd.DataFrame):
    """One slim row per subject on a shared timeline, drawn as one chart so it fits any width: each subject's share of the
    MP's speeches every year (the largest share anywhere fills 82% of a row), with a dot on its peak year. Thin years are
    shaded and left out."""
    k, top = int(df["row"].max()) + 1, float(df["share"].max())
    first, last = int(df["year"].min()), int(df["year"].max())
    df = df.assign(
        base=df["row"],
        y=df["row"] + 0.82 * df["share"].fillna(0) / top,
        x0=df["year"] - 0.5,
        x1=df["year"] + 0.5,
        late=df["year"] > last - (last - first) / 4,  # peak labels near the right edge sit to the left of their dot
        peak_label=[f"{s:.0%} in {y}" if p else "" for s, y, p in zip(df["share"], df["year"], df["peak"])],
    )
    names = df.drop_duplicates("row").sort_values("row")["label"].tolist()
    year_scale = alt.Scale(domain=[first - 0.5, last + 0.5], nice=False, zero=False)
    row_scale = alt.Scale(domain=[0, k], nice=False)
    # Every layer gets these same two axes: layers' axes are merged, so one layer's default (or no) axis would override them.
    x_axis = alt.Axis(format="d", tickMinStep=1, labelAngle=0, grid=False)
    y_axis = alt.Axis(values=[i + 0.45 for i in range(k)], labelExpr=f"{json.dumps(names)}[floor(datum.value)]", domain=False,
                      ticks=False, grid=False, labelFontSize=12, labelColor=INK, labelLimit=260, labelPadding=8,
                      labelOverlap=False)  # the subject names, one per row, however close the rows are
    x = alt.X("year:Q", scale=year_scale, title=None, axis=x_axis)
    y = alt.Y("y:Q", scale=row_scale, title=None, axis=y_axis)
    tip = [
        alt.Tooltip("Topic:N", title="Subject"),
        alt.Tooltip("year:Q", title="Year"),
        alt.Tooltip("share:Q", format=".0%", title="Share of their speeches"),
        alt.Tooltip("speeches:Q", title="Speeches on it"),
        alt.Tooltip("n:Q", title="Speeches that year"),
    ]
    base = alt.Chart(df)
    thin = base.mark_rect(color="#efebe2").encode(
        x=alt.X("x0:Q", scale=year_scale, title=None, axis=x_axis),
        x2="x1:Q",
        y=alt.Y(datum=k, scale=row_scale, title=None, axis=y_axis),
        y2=alt.datum(0),
    ).transform_filter("datum.thin && datum.row == 0")
    rules = base.mark_rule(color=LINE).encode(y=alt.Y("base:Q", scale=row_scale, title=None, axis=y_axis)).transform_filter(
        f"datum.year == {first}")
    area = base.mark_area(color=HL_SOFT, line={"color": INK, "strokeWidth": 1.2}, interpolate="monotone").encode(
        x=x, y=y, y2="base:Q", detail="Topic:N", tooltip=tip
    ).transform_filter("!datum.thin")
    dot = base.mark_circle(color=INK, size=24, opacity=1).encode(x=x, y=y, tooltip=tip).transform_filter("datum.peak")
    # peak labels only where the rows are wide enough to hold them (on a phone the dot stays; a tap shows the numbers)
    text = dict(fontSize=10.5, color=MUTED, baseline="middle", text=alt.ExprRef(expr="width >= 420 ? datum.peak_label : ''"))
    label_r = base.mark_text(align="left", dx=6, **text).encode(x=x, y=y).transform_filter("datum.peak && !datum.late")
    label_l = base.mark_text(align="right", dx=-6, **text).encode(x=x, y=y).transform_filter("datum.peak && datum.late")
    st.altair_chart(
        _style(
            alt.layer(thin, rules, area, dot, label_r, label_l).properties(
                height=40 * k, padding={"left": 8, "top": 8, "right": 8, "bottom": 8}
            )
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
    marks = C.query_marks(topic)
    C.section("What was said", "The three sentences closest to your question, copied word for word from the record.")
    for i, sentence in R.extractive_summary(load_models()[0], topic, hits.head(10), n=3):
        r = hits.loc[i]
        C.quote_card(
            sentence,
            f"<b>{esc(C.display_name(r['displayName']))}</b> ({esc(r['party'])}, {esc(r['electorate'])}) &middot; {esc(cite(r))}",
            marks=marks,
        )
    total = min(20, len(hits))
    C.section("Who spoke about this", f"How many of the {total} best-matching passages came from each MP.")
    C.speaker_rows(G.who_spoke(hits), total)
    C.section("The passages behind it", "Open one to read the full passage and its source.")
    for _, r in hits.head(8).iterrows():
        with st.expander(f"{C.display_name(r['displayName'])} · {C.party_label(r['party'])} · {pd.Timestamp(r['date']):%d %b %Y}"):
            C.passage(r["text"], marks)
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
            _style(
                alt.Chart(per_year)
                .mark_bar(color=INK, width={"band": 0.6})
                .encode(x=alt.X("year:O", title=None), y=alt.Y("speeches:Q", title=None), tooltip=["year", "speeches"])
                .properties(height=280)
            ),
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
                _style(
                    alt.Chart(mix)
                    .mark_bar(color=INK, height={"band": 0.6})
                    .encode(
                        y=alt.Y("Topic:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300, labelOverlap=False)),
                        x=alt.X("share:Q", axis=alt.Axis(format="%"), title=None),
                        tooltip=["Topic", alt.Tooltip("share:Q", format=".0%"), "speeches"],
                    )
                    .properties(height=280)
                ),
                use_container_width=True,
            )
    if topics.empty:
        return
    C.section(
        "How their focus has changed",
        "Each row is one subject: its share of their speeches each year, biggest subject first. The dot marks the year it peaked.",
    )
    rows = focus_rows(topics)
    if rows is None:
        st.write(
            f"Not enough of their record yet to show a change: this needs at least two years with {MIN_YEAR_SPEECHES} or more "
            "speeches that have a subject."
        )
    else:
        _focus_chart(rows)
    st.caption(
        "Subjects are found automatically by a computer model and named by us from their most common words, so the names can be imperfect. "
        f"Speeches under 80 words are not given a subject, and years with fewer than {MIN_YEAR_SPEECHES} speeches that have one are "
        "shaded and left out."
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
    marks = C.query_marks(topic)
    C.section("Key sentences", "Copied word for word from the official record. Nothing here is written by a computer.")
    for i, sentence in key_sentences(topic, hits):
        C.quote_card(sentence, esc(cite(hits.loc[i])), marks=marks)
    C.section("Closest matching passages", "Open one to read the full passage and its source.")
    for _, r in hits.head(5).iterrows():
        with st.expander(f"{pd.Timestamp(r['date']):%d %b %Y} · match score {r['rerank']:.1f}"):
            C.passage(r["text"], marks)
            st.markdown(C.cite_line(r), unsafe_allow_html=True)

    C.section("How the wording changed", "Their earliest and latest strong match on this topic, side by side.")
    hits = hits.assign(year=pd.to_datetime(hits["date"]).dt.year)
    strong = hits[hits["rerank"] >= hits["rerank"].median()].sort_values("year")  # compare only strong matches
    first, last = strong.iloc[0], strong.iloc[-1]
    if first["year"] == last["year"]:
        st.write("The best matches all come from the same year. Widen the years to compare how the wording changed.")
    else:
        for col, r, label in zip(st.columns(2, gap="large"), (first, last), ("Earliest", "Latest")):
            col.markdown(
                f"<div class='then'><div class='yr'><span>{label}</span>{r['year']}</div><p>&hellip;{C.marked(r['text'], marks)}&hellip;</p>"
                f"<div class='cite'>{esc(cite(r))}</div></div>",
                unsafe_allow_html=True,
            )
    st.download_button(
        "Download these passages (CSV)",
        C.results_csv(hits.head(8), fixed_speaker=info["name"]),
        file_name="parlianote_mp_search.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------------------------- shared by "At a glance" and the tabs
# A checkable detail in a promise: an amount, a percentage, a number, or a deadline.
DETAILS = re.compile(
    r"\$?\d[\d,.]*(?:\s?(?:per cent|%|billion|million|thousand))?|\bby (?:the end of )?(?:next|this) (?:year|month|week)\b|"
    r"\bfrom (?:1 )?(?:january|july|october)\b|\bevery (?:week|month|year)\b",
    re.I,
)
# What the highlighter marks in a promise: the pledge itself ("we will ...") and its checkable details.
PLEDGE_MARKS = re.compile("|".join(f"(?:{p.pattern})" for p in (FIRM, GUARANTEE, DETAILS)), re.I)


def _go(view: str):
    """Button callback: switch this MP's page to another tab."""
    st.session_state["view"] = view


def _directness(mp_key: str) -> dict:
    """Question Time numbers for one MP, when answering and when asking. Only questions from the other side count, and answers by
    the Speaker are left out (as on the ministers chart). A role with 10 or more such exchanges gets its share and 95% range."""
    pairs = load_pairs()
    pairs = pairs[~pairs["a_is_chair"]]  # questions put to the Speaker are rulings, not ministerial answers
    out = {"average": float((pairs[~pairs["same_bloc"]]["pred"] == "E").mean())}  # friendly questions are almost always answered directly
    for role, col in (("answering", "a_key"), ("asking", "q_key")):
        mine = pairs[pairs[col] == mp_key]
        other = mine[~mine["same_bloc"]]
        stats = {"n": len(mine), "other": other}
        if len(other) >= 10:
            k = int((other["pred"] == "E").sum())
            low, high = wilson(k, len(other))
            stats.update(share=k / len(other), low=low, high=high)
        out[role] = stats
    return out


def _compared(stats: dict, average: float) -> str:
    """How an MP's 95% range sits against the average, in plain words (overlapping ranges cannot be told apart)."""
    if stats["high"] < average:
        return "fewer than"
    if stats["low"] > average:
        return "more than"
    return "about the same as"


def _range_chart(stats: dict, average: float, who: str = "this MP"):
    """This MP's share with its 95% range, against the average for every minister."""
    C.range_bar(stats["share"], stats["low"], stats["high"], average, who)


def _presentable(sentence: str) -> bool:
    """Whether a flagged sentence reads as a promise on its own: not a question, not cut off mid-way, not a fragment.

    Display only: the evaluated rules in src/commitment.py are unchanged."""
    s = sentence.strip()
    return "?" not in s and bool(re.search(r"[.!]['\"\u2019\u201d)]*$", s)) and len(s.split()) >= 8


def _promises(mp_key: str):
    """Promises in this MP's Question Time answers worth showing, most checkable first (then newest). None if too few answers."""
    answered = load_pairs()
    answered = answered[answered["a_key"] == mp_key]
    if len(answered) < 10:
        return None
    rows = []
    for r in answered.itertuples():
        for sentence in split_sentences(r.a_text):
            kind = commitment_type(sentence)
            if kind != "none" and _presentable(sentence):
                rows.append({"date": r.date, "type": kind, "sentence": sentence, "page": r.a_page, "details": len(DETAILS.findall(sentence))})
    found = pd.DataFrame(rows, columns=["date", "type", "sentence", "page", "details"])
    return found.sort_values(["details", "date"], ascending=False)


def _source(r) -> str:
    page = int(r["page"]) if pd.notna(r["page"]) else "-"
    return f"House of Representatives Hansard, {pd.Timestamp(r['date']):%d %b %Y}, p. {page}"


# --------------------------------------------------------------------------------------------- on this day (home screen)
@st.cache_data(show_spinner=False, ttl=3600)
def _on_this_day(today: str):
    """A short, self-contained sentence spoken on today's date in an earlier year (newest year first), or on the nearest earlier
    date that Parliament sat. Same rules as the idle tour's quotes: no parties or sides, and it must make sense on its own."""
    dates = pd.Series(pq.read_table(DATA / "speeches.parquet", columns=["date"]).column("date").to_pandas().unique())
    day = pd.Timestamp(today)
    for back in range(30):
        wanted = (day - pd.Timedelta(days=back)).strftime("%m-%d")
        sittings = sorted(dates[dates.dt.strftime("%m-%d") == wanted], reverse=True)
        for date in sittings:
            cols = ["date", "displayName", "party", "electorate", "page", "body", "words"]
            t = pq.read_table(DATA / "speeches.parquet", columns=cols, filters=[("date", "=", date)]).to_pandas()
            found = [(quotes.score(s, r.electorate), s, r) for r in t[t["words"] >= 60].itertuples() for s in split_sentences(r.body)]
            found = [f for f in found if quotes.usable(f[1])]
            if found:
                top = max(f[0] for f in found)
                # among the best, a pick seeded by the date: fair across MPs (the file is in name order) and steady all day
                _, sentence, r = random.Random(today).choice([f for f in found if f[0] == top])
                return {"text": sentence, "date": pd.Timestamp(r.date), "page": r.page, "who": C.display_name(r.displayName),
                        "party": C.party_label(r.party), "electorate": r.electorate, "exact": back == 0}
    return None


def on_this_day():
    """A sentence from the record spoken on this date in an earlier year (for the empty home screen)."""
    from zoneinfo import ZoneInfo
    from datetime import datetime

    q = _on_this_day(datetime.now(ZoneInfo("Australia/Sydney")).strftime("%Y-%m-%d"))
    if not q:
        return
    when = f"{q['date'].day} {q['date']:%B %Y}"  # not %-d, which Windows does not support
    C.section("On this day in Parliament" if q["exact"] else "From the record", f"Said on {when}." if q["exact"] else f"The nearest sitting day: {when}.")
    C.quote_card(
        q["text"],
        f"<b>{esc(q['who'])}</b> ({esc(q['party'])}, {esc(q['electorate'])}) &middot; "
        f"{esc(cite({'date': q['date'], 'page': q['page']}))}",
    )


# --------------------------------------------------------------------------------------------- at a glance (the first tab)
def at_a_glance(mp_key: str, info, sp: pd.DataFrame):
    """The poster's questions about one MP, answered in plain words, each with its evidence one click away."""
    name = esc(C.display_name(info["name"]))
    C.section("The record at a glance", f"How {name} uses Parliament, answered from the record. Each answer links to its evidence.")
    col_qt, col_pr, col_tp = st.columns(3, gap="large")
    with col_qt:
        _glance_answers(mp_key)
    with col_pr:
        _glance_promises(mp_key)
    with col_tp:
        _glance_subjects(sp)
    st.markdown(
        "<div class='glance' style='margin-top:2.2rem'><div class='q'>Has their position shifted?</div><div class='s'>"
        "Pick any topic in <b>Ask the record</b> to read their earliest and latest words on it, side by side.</div></div>",
        unsafe_allow_html=True,
    )
    st.button("Ask the record →", key="go_ask", on_click=_go, args=("Ask the record",))


def _glance_answers(mp_key: str):
    d = _directness(mp_key)
    a, q, avg = d["answering"], d["asking"], d["average"]
    fine = "A rough estimate from the wording, right about 68% of the time. Never a verdict on a single answer."
    if "share" in a:
        C.glance(
            "Do they answer the question?",
            f"{a['share']:.0%}",
            f"of their answers to the other side's questions did not directly address them: {_compared(a, avg)} the average minister.",
            f"95% range {a['low']:.0%} to {a['high']:.0%}, from {len(a['other'])} answers since 2011. {fine}",
        )
        _range_chart(a, avg)
        st.button("Read the exchanges →", key="go_qt", on_click=_go, args=("Question Time",))
    elif "share" in q:
        C.glance(
            "Do their questions get answered?",
            f"{q['share']:.0%}",
            f"of ministers' answers to their questions did not directly address them: {_compared(q, avg)} the average.",
            f"95% range {q['low']:.0%} to {q['high']:.0%}, from {len(q['other'])} questions since 2011. {fine}",
        )
        _range_chart(q, avg)
        st.button("See the numbers →", key="go_qt", on_click=_go, args=("Question Time",))
    else:
        C.glance(
            "Do they answer the question?",
            "Too few exchanges",
            "They have fewer than 10 Question Time exchanges with the other side since 2011, too few for an estimate.",
            words=True,
        )


def _glance_promises(mp_key: str):
    found = _promises(mp_key)
    if found is None:
        C.glance(
            "What have they promised?",
            "Too few answers",
            "We look for promises in the answers ministers give in Question Time, and they have too few recorded answers.",
            words=True,
        )
        return
    specific = found[found["type"] == "specific pledge"]
    n_general = int((found["type"] == "general pledge").sum())
    C.glance(
        "What have they promised?",
        f"{len(specific)}",
        f"specific promises, with a number, date or amount you could check, and {n_general} general ones, in their Question Time answers.",
        "Found with simple pattern rules, so some are missed or wrongly included.",
    )
    if len(specific):
        r = specific.iloc[0]
        C.quote_card(r["sentence"], esc(_source(r)), marks=PLEDGE_MARKS)
    st.button("All their promises →", key="go_pr", on_click=_go, args=("Promises",))


def _glance_subjects(sp: pd.DataFrame):
    topics = sp[sp["topic"] >= 0]
    if topics.empty:
        C.glance("What do they talk about?", "Not enough to say", "They have too few longer speeches for us to work out subjects.", words=True)
        return
    share = topics["topic"].map(TOPIC_NAMES).value_counts(normalize=True).head(3)
    years = f"{int(sp['year'].min())}&ndash;{int(sp['year'].max())}"
    C.glance(
        "What do they talk about?",
        esc(share.index[0]),
        f"is their most common subject, across {len(sp):,} speeches from {years}.",
        "Subjects are found by a topic model and named by us, so the names can be imperfect.",
        words=True,
    )
    C.share_bars(list(share.items()))
    st.button("Speeches and subjects →", key="go_ov", on_click=_go, args=("Overview",))


# --------------------------------------------------------------------------------------------- question time and promises tabs
def _sentences(text: str):
    """Every sentence of a passage, short ones included (for shading an answer sentence by sentence)."""
    return [p for p in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'\u2018\u201c(])", " ".join(str(text).split())) if p.strip()]


@st.cache_data(show_spinner=False, max_entries=128)
def _relevance(question: str, answer: str):
    """How closely each sentence of an answer (the first 40) responds to the question: [(sentence, score), ...].

    Scored by the cross-encoder, which reads the question (its first 120 words, as in the directness features) and the sentence together."""
    sentences = _sentences(answer)[:40]
    q = " ".join(str(question).split()[:120])
    scores = load_models()[1].predict([(q, s) for s in sentences], batch_size=32, show_progress_bar=False)
    return [(s, float(x)) for s, x in zip(sentences, scores)]


def _exchange(r):
    """One Question Time exchange: the question, then the answer shaded by how closely each sentence responds to it."""
    with st.expander(f"{pd.Timestamp(r['date']):%d %b %Y}: question from {C.display_name(r['q_name'])}"):
        st.markdown("**Question**")
        C.passage(r["q_text"])
        st.markdown("**Answer**")
        C.shaded_answer(_relevance(r["q_text"], r["a_text"]))
        page = int(r["a_page"]) if pd.notna(r["a_page"]) else "-"
        st.caption(
            f"House of Representatives Hansard, {pd.Timestamp(r['date']):%d %b %Y}, p. {page}. "
            f"The model's score for the whole answer: {r['p_E']:.2f} (0 = direct, 1 = does not address the question)."
        )


def question_time(mp_key: str):
    """How directly this MP's answers (or the answers to their questions) address the question asked. Estimates only."""
    C.section(
        "Do the answers address the question?",
        "Question Time is the daily session where MPs question ministers. This is a rough statistical estimate (2011 to 2025) "
        "of how directly answers respond, based on wording only. It is never a verdict on a single answer.",
    )
    d = _directness(mp_key)
    a, q, average = d["answering"], d["asking"], d["average"]

    if a["n"] >= 10:
        st.markdown(
            f"**When answering:** {a['n']} Question Time answers, {len(a['other'])} of them to questions from the other side "
            "(opposition or crossbench MPs)."
        )
        if "share" in a:
            C.glance(
                "When they answer",
                f"{a['share']:.0%}",
                f"of their answers to the other side did not directly address the question: {_compared(a, average)} "
                f"the average for all ministers ({average:.0%}).",
                f"We are 95% confident the true figure for this MP is between {a['low']:.0%} and {a['high']:.0%} "
                f"(based on {len(a['other'])} answers). Differences of a few points are within the model's margin of error.",
            )
            _range_chart(a, average)
            st.markdown("**Exchanges scored as least direct** (read them yourself; each answer is shaded sentence by sentence):")
            for _, r in a["other"].sort_values("p_E", ascending=False).head(3).iterrows():
                _exchange(r)
            st.markdown("**For comparison, exchanges scored as most direct:**")
            for _, r in a["other"].sort_values("p_E").head(2).iterrows():
                _exchange(r)
            st.caption(
                "The shading comes from a model that reads the question and each sentence together (the same one the directness "
                "estimate uses). It shows how closely each sentence responds to the question, not whether it is true or a good answer."
            )

    if q["n"] >= 10:
        st.markdown(f"**When asking:** {q['n']} questions, {len(q['other'])} of them to ministers from the other side.")
        if "share" in q:
            C.glance(
                "When they ask",
                f"{q['share']:.0%}",
                f"of ministers' answers to their questions did not directly address them: {_compared(q, average)} "
                f"the average ({average:.0%}).",
                f"95% range for this MP: {q['low']:.0%} to {q['high']:.0%}.",
            )
            _range_chart(q, average)

    if a["n"] < 10 and q["n"] < 10:
        st.info(
            "This MP has fewer than 10 recorded Question Time exchanges, which is too few for a reliable estimate. "
            "Question and answer labels in the source data are only reliable from 2011."
        )


def promises(mp_key: str):
    """Sentences in this MP's Question Time answers where they promise something, split into specific and general."""
    C.section(
        "Promises made in Question Time answers",
        "Sentences like &ldquo;we will ...&rdquo;. A promise is <b>specific</b> when it includes a number, date or dollar amount "
        "you could check later; the most checkable come first.",
    )
    found = _promises(mp_key)
    if found is None:
        st.info("We look for promises in the answers ministers give, and this MP has too few recorded answers.")
        return
    if found.empty:
        st.write("We did not find any sentences that look like promises.")
        return
    n_specific = int((found["type"] == "specific pledge").sum())
    n_general = int((found["type"] == "general pledge").sum())
    labels = {"specific pledge": f"Specific ({n_specific})", "general pledge": f"General ({n_general})"}
    chosen = st.radio("Show", list(labels), format_func=labels.get, horizontal=True, label_visibility="collapsed")
    for _, r in found[found["type"] == chosen].head(15).iterrows():
        C.quote_card(r["sentence"], esc(_source(r)), marks=PLEDGE_MARKS)
    st.caption(
        'Promises are found with simple pattern rules (for example "we will ..."), so some sentences will be missed or wrongly included. '
        "Questions and sentences cut off mid-way are left out."
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

**Photos.** Each MP's official portrait comes from the Parliament of Australia website (CC BY-NC-ND 4.0) and is shown as published, trimmed at the edges to fit its frame.
Your browser loads it from aph.gov.au when you open an MP; if it cannot, their initials are shown instead.

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
    table = table.assign(
        label=table["name"].map(C.display_name) + " (" + table["party"] + ")",
        bloc=table["party"].map(bloc),
    )
    table = table.sort_values("name")  # always alphabetical: this screen must not rank people
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
    colours = alt.Scale(domain=list(BLOC_COLOURS), range=list(BLOC_COLOURS.values()))
    ranges = base.mark_rule(strokeWidth=2.5, opacity=0.45).encode(
        x=alt.X(
            "low:Q",
            axis=alt.Axis(format="%"),
            title="Answers scored as not directly addressing the question",
            scale=alt.Scale(domain=[0, 1]),
        ),
        x2="high:Q",
        color=alt.Color("bloc:N", scale=colours, legend=alt.Legend(title=None, orient="top")),
        tooltip=tooltip,
    )
    points = base.mark_circle(size=70, opacity=1).encode(x="share:Q", color=alt.Color("bloc:N", scale=colours), tooltip=tooltip)
    avg = alt.Chart(pd.DataFrame({"x": [average]})).mark_rule(strokeDash=[4, 4], color=INK).encode(x="x:Q")
    st.altair_chart(_style((ranges + points + avg).properties(height=max(260, 26 * len(table)))), use_container_width=True)

    above = int((table["low"] > average).sum())
    below = int((table["high"] < average).sum())
    st.caption(
        f"Dashed line: the average for all ministers ({average:.0%}). Of these {len(table)} MPs, the whole range of {above} sits above "
        f"the average and of {below} below it; the other {len(table) - above - below} cannot be told apart from the average. "
        "Questions from the minister's own side and questions to the Speaker are left out. Question Time data covers 2011 to 2025."
    )
    with st.expander("Show as a table"):
        # a plain table (not st.dataframe) so viewers cannot click a column header to rank the MPs
        shown = table[["label", "answers", "share", "low", "high", "first", "last"]].rename(
            columns={
                "label": "MP",
                "answers": "Answers",
                "share": "Not direct",
                "low": "95% from",
                "high": "95% to",
                "first": "From",
                "last": "To",
            }
        )
        for col in ["Not direct", "95% from", "95% to"]:
            shown[col] = shown[col].map("{:.2f}".format)
        st.table(shown.set_index("MP"))
    _fairness(average)


def _years(years) -> str:
    """Runs of years in words: [2011, 2012, 2013, 2022, 2023] -> "2011–13 and 2022–23"."""
    years, runs = sorted(set(int(y) for y in years)), []
    for y in years:
        if runs and y == runs[-1][1] + 1:
            runs[-1][1] = y
        else:
            runs.append([y, y])
    return " and ".join(f"{a}&ndash;{str(b)[2:]}" if b > a else str(a) for a, b in runs)


def _fairness(average: float):
    """Three checks a sceptical reader should ask for: does it favour a side, does it agree with people, how often is it right."""
    C.section(
        "Is the model fair?",
        "Three checks on whether this estimate treats the sides of politics differently, and on how far to trust it.",
    )
    pairs = load_pairs()
    opp = pairs[~pairs["a_is_chair"] & ~pairs["same_bloc"]].assign(side=lambda d: d["a_party"].map(bloc))
    agreement = json.loads((ROOT / "results" / "agreement.json").read_text())
    comparison = pd.read_csv(ROOT / "results" / "model_comparison.csv")
    col_side, col_people, col_right = st.columns(3, gap="large")

    with col_side:
        shares = {}
        for side in ("Labor", "Coalition"):
            g = opp[opp["side"] == side]
            k = int((g["pred"] == "E").sum())
            per_year = g["year"].value_counts()
            years = _years(per_year[per_year >= 20].index)  # the years in government (a stray answer or two in other years is ignored)
            shares[side] = (k / len(g), *wilson(k, len(g)), years, len(g))
        C.glance(
            "Does it favour a side?",
            f"{shares['Labor'][0]:.0%} vs {shares['Coalition'][0]:.0%}",
            f"of answers to the other side scored as not direct: Labor ministers ({shares['Labor'][3]}) against Coalition ministers "
            f"({shares['Coalition'][3]}). The sides governed in different years, against different oppositions and questions, so "
            "this gap cannot tell model bias apart from real differences.",
            "The model is never told anyone's party: only whether a question came from their own side.",
        )
        for side in ("Labor", "Coalition"):
            share, low, high, years, n = shares[side]
            C.range_bar(share, low, high, average, who=f"{side} ministers, {n:,} answers")

    with col_people:
        model = agreement["model_vs_human"]
        humans = next(v for k, v in agreement["pairwise"].items() if "annotator_1" not in k)  # the two people who labelled independently
        m = [v["agree_evasive_or_not"] for v in model.values()]
        C.glance(
            "Does it agree with people?",
            f"{min(m):.0%}&ndash;{max(m):.0%}",
            f"agreement with each of two team members who judged {humans['n']} answers on their own (dodged the question or not), "
            f"at least as often as the two people agreed with each other ({humans['agree_evasive_or_not']:.0%}).",
            f"Cohen's kappa, which discounts agreement by chance: model and people "
            f"{min(v['kappa_evasive_or_not'] for v in model.values()):.2f}&ndash;{max(v['kappa_evasive_or_not'] for v in model.values()):.2f}, "
            f"the two people {humans['kappa_evasive_or_not']:.2f}: moderate. Judging evasiveness is subjective even for people.",
        )

    with col_right:
        other = comparison[comparison["subset"].str.startswith("Other") & (comparison["task"] == "evasive vs not")].set_index("model")
        C.glance(
            "How often is it right?",
            f"{other.loc['All features + LR', 'accuracy']:.0%}",
            "of hand-labelled answers to the other side sorted correctly into &ldquo;not direct&rdquo; or not, against "
            f"{other.loc['Majority class', 'accuracy']:.0%} by always guessing the most common label.",
            "Cross-validated on the 120 answers labelled for training. Good enough to screen many answers, never to judge one.",
        )

    st.markdown(
        "**What it cannot tell you.** Whether an answer was true, what anyone intended, or whether a single answer dodged the "
        "question. It also cannot see tone, time limits or rulings from the Speaker; it reads only the words in the record."
    )
    by_year = opp.groupby("year")["pred"].apply(lambda s: (s == "E").mean()).reset_index(name="share")
    terms = pd.DataFrame(
        {"start": [2010.5, 2013.7, 2022.4], "end": [2013.7, 2022.4, 2025.5], "Government": ["Labor", "Coalition", "Labor"]}
    )
    bands = alt.Chart(terms).mark_rect(opacity=0.12).encode(
        x=alt.X("start:Q", scale=alt.Scale(domain=[2010.5, 2025.5]), axis=alt.Axis(format="d", title=None, tickCount=8)),
        x2="end:Q",
        color=alt.Color("Government:N", scale=alt.Scale(domain=list(BLOC_COLOURS)[:2], range=list(BLOC_COLOURS.values())[:2]),
                        legend=alt.Legend(title="In government", orient="top")),
    )
    line = alt.Chart(by_year).mark_line(color=INK, point=alt.OverlayMarkDef(color=INK, size=40)).encode(
        x="year:Q",
        y=alt.Y("share:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title=None)),
        tooltip=[alt.Tooltip("year:Q", format="d"), alt.Tooltip("share:Q", title="Not direct", format=".0%")],
    )
    C.hint("Answers to the other side scored as not direct, by year; the shading shows which side was in government.")
    st.altair_chart(_style((bands + line).properties(height=220)), use_container_width=True)


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
    seat_map_pair(*[(info["electorate"], C.display_name(info["name"])) for info in people])

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
                .mark_bar()
                .encode(
                    y=alt.Y("Topic:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
                    yOffset=alt.YOffset("MP:N", sort=[C.display_name(i["name"]) for i in people]),
                    x=alt.X("share:Q", axis=alt.Axis(format="%"), title=None),
                    color=alt.Color(
                        "MP:N",
                        scale=alt.Scale(domain=[C.display_name(i["name"]) for i in people], range=[INK, OCHRE]),
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
                C.quote_card(sentence, esc(cite(hits.loc[i])), marks=C.query_marks(topic))
