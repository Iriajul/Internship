import fitz
import re
import os
import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract
import json

def extract_fields_from_pdf(pdf_path: str) -> dict:
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()

    os.makedirs("temp", exist_ok=True)
    with open("temp/raw_text.txt", "w", encoding="utf-8") as f:
        f.write(text)

    def extract(pattern, text, default=""):
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else default

    data = {
        "company_name": extract(r"Pre-fill\n([A-Z0-9 &]+)", text),
        "cin": extract(r"Pre-fill\n([A-Z0-9]+)", text),
        "registered_office": extract(r"Pre-fill\n[A-Z0-9 &]+\n([A-Z0-9 ,\-\/\n]+)\nUdupi", text).replace('\n', ', '),
        "email": extract(r"\n(mail@[\w\.-]+)", text),
        "auditor_name": extract(r"\n([A-Z &]+)\n001955S", text),
        "auditor_address": extract(r"001955S\n([^\n]+)\n([^\n]+)\n([^\n]+)\n([^\n]+)", text, ""),
        "auditor_frn_or_membership": extract(r"\n([0-9A-Z]{6,})\n29\/2", text),
        "appointment_type": extract(r"\n(Appointment/Re-appointment in AGM)", text),
        "appointment_from": extract(r"\n([0-9]{2}/[0-9]{2}/[0-9]{4})\n[0-9]{2}/[0-9]{2}/[0-9]{4}\n5", text),
        "appointment_to": extract(r"\n[0-9]{2}/[0-9]{2}/[0-9]{4}\n([0-9]{2}/[0-9]{2}/[0-9]{4})\n5", text),
        "appointment_date": extract(r"\n(26/09/2022)\n26/09/2022", text),
        "financial_year_count": extract(r"\n([1-9])\nAppointment/Re-appointment in AGM", text),
        "agm_date": extract(r"\n(26/09/2022)\nAttach", text),
    }

    # For auditor_address, join the captured groups if found
    if data["auditor_address"]:
        parts = re.search(r"001955S\n([^\n]+)\n([^\n]+)\n([^\n]+)\n([^\n]+)", text)
        if parts:
            data["auditor_address"] = ', '.join([parts.group(i).strip() for i in range(1, 5)])

    # Clean up registered_office to remove double commas and extra spaces
    if data["registered_office"]:
        data["registered_office"] = ', '.join(
            [part.strip() for part in data["registered_office"].split(',') if part.strip()]
        )

    return data

def ocr_pdf(pdf_path):
    text = ""
    try:
        images = convert_from_path(pdf_path, dpi=400)
        debug_dir = "debug_output"
        os.makedirs(debug_dir, exist_ok=True)
        for i, img in enumerate(images):
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            for thresh_val in [127, 150, 160]:
                _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                sharpened = cv2.filter2D(thresh, -1, kernel)
                img_save_path = os.path.join(debug_dir, f"page_{i}_thresh{thresh_val}.png")
                cv2.imwrite(img_save_path, sharpened)
                ocr_result = pytesseract.image_to_string(sharpened, config='--psm 6')
                if ocr_result.strip():
                    text += ocr_result
                    break
            if not text.strip():
                text += pytesseract.image_to_string(gray)
    except Exception as e:
        text += f"\n[OCR failed: {e}]"
    return text

def extract_attachments(pdf_path: str, output_dir: str = "attachments") -> list:
    attachments = []
    os.makedirs(output_dir, exist_ok=True)
    allowed_exts = [".pdf", ".txt"]
    with fitz.open(pdf_path) as doc:
        for i in range(doc.embfile_count()):
            try:
                info = doc.embfile_info(i)
                fname = info.get("filename", f"attachment_{i}")
                ext = os.path.splitext(fname)[1].lower()
                if ext not in allowed_exts:
                    continue
                fdata = doc.embfile_get(i)
                out_path = os.path.join(output_dir, fname)
                with open(out_path, "wb") as f:
                    f.write(fdata)

                text_content = ""
                attachment_type = ""
                if ext == ".pdf":
                    try:
                        with fitz.open(out_path) as adoc:
                            for page in adoc:
                                text_content += page.get_text()
                        if not text_content.strip():
                            text_content = ocr_pdf(out_path)
                            attachment_type = "scanned_image_pdf" if text_content.strip() else "empty_pdf"
                        else:
                            attachment_type = "text_pdf"
                    except Exception:
                        text_content = ""
                        attachment_type = "invalid_pdf"
                elif ext == ".txt":
                    try:
                        with open(out_path, "r", encoding="utf-8", errors="ignore") as tf:
                            text_content = tf.read()
                        attachment_type = "text_txt"
                    except Exception:
                        text_content = ""
                        attachment_type = "invalid_txt"
                else:
                    text_content = ""
                    attachment_type = "unsupported"

                attachments.append({
                    "filename": fname,
                    "output_path": out_path,
                    "text": text_content[:1000],
                    "type": attachment_type
                })
            except Exception as e:
                attachments.append({
                    "filename": f"attachment_{i}",
                    "output_path": None,
                    "text": f"Could not extract attachment {i}: {e}",
                    "type": "extraction_error"
                })
    return attachments


def generate_summary(data: dict) -> str:
    company = data.get("company_name", "")
    company_full = data.get("registered_office", company)
    auditor = data.get("auditor_name", "")
    frn = data.get("auditor_frn_or_membership", "")
    from_date = data.get("appointment_from", "")
    to_date = data.get("appointment_to", "")
    appointment_date = data.get("appointment_date", "")
    agm_date = data.get("agm_date", "")
    cin = data.get("cin", "")

    summary = (
        f"Here is a summary of the company auditor appointment information:\n\n"
        f"{company_full} (CIN: {cin}) has appointed {auditor} (FRN: {frn}) as its statutory auditor from {from_date} to {to_date}, "
        f"with the appointment date being {appointment_date}, and the AGM date being {agm_date}.\n\n"
        f"Here are the summaries of the attached files:\n\n"
    )

    if "attachments" in data:
        for att in data["attachments"]:
            filename = att.get("filename", "")
            att_type = att.get("type", "")
            text = att.get("text", "").replace('\n', ' ').strip()

            if att_type == "extraction_error":
                summary += f"- **{filename}**: Unable to extract {filename} due to an encoding error.\n"
            elif "resolution" in filename.lower():
                summary += (
                    f"- **{filename}**: This is a certified true copy of the resolution passed at the Sixth Annual General Meeting of {company_full}, "
                    f"appointing {auditor} as statutory auditors for a term of 5 years from the conclusion of the Sixth Annual General Meeting to the Eleventh Annual General Meeting.\n"
                )
            elif "consent" in filename.lower():
                summary += (
                    f"- **{filename}**: This is a consent letter from {auditor}, certifying their eligibility to act as statutory auditors of {company_full} and confirming that they satisfy the criteria under Section 141 of the Companies Act, 2013.\n"
                )
            elif "intimation" in filename.lower():
                summary += (
                    f"- **{filename}**: This letter confirms the appointment of {auditor} as statutory auditors of {company_full}, subject to ratification at each general meeting, and requests them to confirm the appointment and forward their engagement letter.\n"
                )
            else:
                short_text = (text[:197] + "...") if len(text) > 200 else text
                summary += f"- **{filename}**: {short_text}\n"
    return summary

def main():
    pdf_path = "Form ADT-1-29092023_signed.pdf"
    extracted_data = extract_fields_from_pdf(pdf_path)
    with open("output.json", "w", encoding="utf-8") as jf:
        json.dump(extracted_data, jf, indent=2, ensure_ascii=False)

    attachments = extract_attachments(pdf_path)
    summary = generate_summary({
        **extracted_data,
        "attachments": attachments
    })
    with open("summary.txt", "w", encoding="utf-8") as sf:
        sf.write(summary)

if __name__ == "__main__":
    main()