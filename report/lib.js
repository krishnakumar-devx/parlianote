// Formatting helpers for the report (docx-js).
const fs = require("fs");
const D = require("docx");
const { Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType, ShadingType,
        BorderStyle, ImageRun, LevelFormat, PageBreak } = D;

const NAVY = "16304F", GOLD = "9C6F0A", GREY = "5B6B7B", INK = "1C2530";
const FONT = "Calibri";
const CONTENT_W = 9638; // A4 width 11906 minus 2 x 1134 margins (DXA)

// "text with **bold** and *italic*" -> runs
function runs(text, base = {}) {
  text = String(text).replace(/`/g, "");
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, size: 21, color: INK, ...base }));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(new TextRun({ text: tok.slice(2, -2), bold: true, font: FONT, size: 21, color: INK, ...base }));
    else out.push(new TextRun({ text: tok.slice(1, -1), italics: true, font: FONT, size: 21, color: INK, ...base }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, size: 21, color: INK, ...base }));
  return out;
}

const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, keepNext: true, spacing: { before: 280, after: 120 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: NAVY, space: 3 } },
  children: [new TextRun({ text: t, bold: true, font: FONT, size: 30, color: NAVY })] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, spacing: { before: 200, after: 80 },
  children: [new TextRun({ text: t, bold: true, font: FONT, size: 24, color: NAVY })] });
const h3 = (t) => new Paragraph({ keepNext: true, spacing: { before: 120, after: 60 },
  children: [new TextRun({ text: t, bold: true, italics: true, font: FONT, size: 21, color: NAVY })] });
const p = (t, opts = {}) => new Paragraph({ spacing: { after: 110, line: 270 }, alignment: AlignmentType.JUSTIFIED, ...opts, children: runs(t) });
const bullets = (items, ref = "bul") => items.map((t) => new Paragraph({ numbering: { reference: ref, level: 0 }, spacing: { after: 60, line: 262 }, children: runs(t) }));
const numbered = (items) => bullets(items, "num");

function caption(text) {
  return new Paragraph({ spacing: { before: 60, after: 160 }, alignment: AlignmentType.LEFT,
    children: runs(text, { size: 18, color: GREY }) });
}

// Table: header row + body rows; widths are proportions that are scaled to CONTENT_W
function table(header, rows, widths, opts = {}) {
  const sum = widths.reduce((a, b) => a + b, 0);
  const w = widths.map((x) => Math.round((x / sum) * CONTENT_W));
  w[w.length - 1] += CONTENT_W - w.reduce((a, b) => a + b, 0);
  const cell = (t, i, head, shade) => new TableCell({
    width: { size: w[i], type: WidthType.DXA },
    shading: head ? { type: ShadingType.CLEAR, fill: NAVY } : shade ? { type: ShadingType.CLEAR, fill: "F3EFE4" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: String(t).split("\n").map((line) => new Paragraph({ spacing: { after: 0, line: 245 },
      children: runs(line, { size: opts.size || 18, color: head ? "FFFFFF" : INK, bold: head ? true : undefined }) })),
  });
  const border = { style: BorderStyle.SINGLE, size: 4, color: "D8D2BF" };
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: w,
    borders: { top: border, bottom: border, left: border, right: border, insideHorizontal: border, insideVertical: border },
    rows: [new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, true)) }),
           ...rows.map((r, k) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, false, k % 2 === 1)) }))],
  });
}

function figure(path, widthCm, caps) {
  const buf = fs.readFileSync(path);
  // read PNG dimensions from the IHDR chunk
  const wpx = buf.readUInt32BE(16), hpx = buf.readUInt32BE(20);
  const wEmu = widthCm; const hCm = widthCm * hpx / wpx;
  const px = (cm) => Math.round(cm / 2.54 * 96);
  return [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100, after: 40 }, keepNext: true,
    children: [new ImageRun({ type: "png", data: buf, transformation: { width: px(wEmu), height: px(hCm) },
      altText: { title: caps, description: caps, name: caps.slice(0, 40) } })] }), caption(caps)];
}

const spacer = (n = 80) => new Paragraph({ spacing: { after: n }, children: [] });
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });
const code = (lines) => lines.map((l) => new Paragraph({ spacing: { after: 0, line: 220 }, shading: { type: ShadingType.CLEAR, fill: "F3F3F3" },
  children: [new TextRun({ text: l || " ", font: "Consolas", size: 16, color: "222222" })] }));

module.exports = { D, runs, h1, h2, h3, p, bullets, numbered, caption, table, figure, spacer, pageBreak, code, NAVY, GOLD, GREY, INK, FONT, CONTENT_W };
