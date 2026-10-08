"""Convert the report to PDF with Word (Windows) and render each page to results/pages/ for visual checks."""

import glob, os
import win32com.client
import pypdf, pypdfium2 as pdfium

root = os.path.abspath(".")
docx, pdf_path = (
    os.path.join(root, "report", "Parlianote_Technical_Report.docx"),
    os.path.join(root, "report", "Parlianote_Technical_Report.pdf"),
)
word = win32com.client.DispatchEx("Word.Application")
word.Visible = False
try:
    doc = word.Documents.Open(docx, ReadOnly=True)
    doc.SaveAs(pdf_path, FileFormat=17)
    doc.Close(False)
finally:
    word.Quit()
r = pypdf.PdfReader(pdf_path)
print("pages:", len(r.pages))
os.makedirs("results/pages", exist_ok=True)
for f in glob.glob("results/pages/p*.png"):
    os.remove(f)
pdf = pdfium.PdfDocument(pdf_path)
for i in range(len(pdf)):
    pdf[i].render(scale=1.25).to_pil().save(f"results/pages/p{i + 1:02d}.png")
for i, pg in enumerate(r.pages):
    print(i + 1, (pg.extract_text() or "")[60:120].replace("\n", " | "))
