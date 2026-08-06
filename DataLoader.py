import io
import pandas as pd
from pypdf import PdfReader

def load_csv(file):
    return pd.read_csv(file)

def load_excel(file):
    return pd.read_excel(file)

def load_pdf(file):

    pdf_bytes = file.getvalue()
    pdf_stream = io.BytesIO(pdf_bytes)
    reader = PdfReader(pdf_stream)

    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    if not text.strip():
        raise ValueError(
            "Error"
        )
    return text

def load_txt(file):
    return file.getvalue().decode(
        "utf-8",
        errors="ignore"
    )

def load_file(file):
    file_type = file.name.split(".")[-1].lower()

    if file_type == "csv":
        return load_csv(file)

    elif file_type in ["xlsx", "xls"]:
        return load_excel(file)

    elif file_type == "pdf":
        return load_pdf(file)

    elif file_type == "txt":
        return load_txt(file)

    else:
        raise ValueError(
            f"Unsupported file type: {file_type}"
        )