// Builds report/Parlianote_Technical_Report.docx   (run from the project root: node report/build_report.js)
const fs = require("fs");
const L = require("./lib");
const { D } = L;

const body = [
  // title block
  new D.Paragraph({ spacing: { after: 40 }, children: [new D.TextRun({ text: "Parlianote", bold: true, font: L.FONT, size: 56, color: L.NAVY })] }),
  new D.Paragraph({ spacing: { after: 60 }, children: [new D.TextRun({ text: "Making the parliamentary record searchable, citable and checkable by postcode", font: L.FONT, size: 28, color: L.GREY })] }),
  new D.Paragraph({ spacing: { after: 20 }, children: [new D.TextRun({ text: "36118 Applied Natural Language Processing, UTS | Technical report | Group 11 (Team Phoenix)", font: L.FONT, size: 20, color: L.INK })] }),
  new D.Paragraph({ spacing: { after: 20 }, children: [new D.TextRun({ text: "Krishna Kumar Thirugnanavel, Sidharth Joly, Brian Shimmer, Malavika Rajesh, Prathamesh Nemade, Milad Kianfar", font: L.FONT, size: 20, color: L.INK })] }),
  new D.Paragraph({ spacing: { after: 160 }, border: { bottom: { style: D.BorderStyle.SINGLE, size: 8, color: L.GOLD, space: 6 } }, children: [new D.TextRun({ text: "October 2026", font: L.FONT, size: 20, color: L.GREY })] }),
  ...require("./sections/s1")(),
  ...require("./sections/s2")(),
  ...require("./sections/s3")(),
  ...require("./sections/s4")(),
  ...require("./sections/s5")(),
];

const num = (ref) => ({ reference: ref, levels: [{ level: 0, format: ref === "bul" ? D.LevelFormat.BULLET : D.LevelFormat.DECIMAL,
  text: ref === "bul" ? "•" : "%1.", alignment: D.AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 340 } } } }] });

const doc = new D.Document({
  creator: "Group 11 (Team Phoenix)", title: "Parlianote technical report", description: "36118 Applied NLP",
  styles: { default: { document: { run: { font: L.FONT, size: 21 } } } },
  numbering: { config: [num("bul"), num("num")] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    footers: { default: new D.Footer({ children: [new D.Paragraph({ alignment: D.AlignmentType.CENTER, children: [
      new D.TextRun({ text: "Parlianote | Group 11 (Team Phoenix) | page ", font: L.FONT, size: 16, color: L.GREY }),
      new D.TextRun({ children: [D.PageNumber.CURRENT], font: L.FONT, size: 16, color: L.GREY })] })] }) },
    children: body,
  }],
});

D.Packer.toBuffer(doc).then((buf) => { fs.writeFileSync("report/Parlianote_Technical_Report.docx", buf); console.log("written", buf.length, "bytes"); });
