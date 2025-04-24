import re

def extract_metadata(text):
    """
    Extracts metadata from the given text, including names, dates, IDs, phone numbers,
    email addresses, monetary amounts, and more.
    """
    metadata = {}

    # Name
    name_match = re.search(r"(Name|Student|Employee|Customer)\s*[:\-]?\s*([A-Z][a-z]+\s[A-Z][a-z]+)", text)
    if name_match:
        metadata["Name"] = name_match.group(2)

    # Date
    date_match = re.search(r"(Date of Issue|Date|Due Date|Enrollment Date)\s*[:\-]?\s*(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})", text)
    if date_match:
        metadata["Date"] = date_match.group(2)

    # ID
    id_match = re.search(r"(Student ID|Account Number|Employee ID|Transaction ID|Reference Number)\s*[:\-]?\s*(\w+)", text)
    if id_match:
        metadata["ID"] = id_match.group(2)

    # Phone Number
    phone_match = re.search(r"(Phone|Contact)\s*[:\-]?\s*(\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9})", text)
    if phone_match:
        metadata["Phone"] = phone_match.group(2)

    # Email Address
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    if email_match:
        metadata["Email"] = email_match.group(0)

    # Monetary Amount
    amount_match = re.search(r"(Total Due|Amount|Balance|Net Pay|Gross Pay)\s*[:\-]?\s*\$?(\d{1,3}(,\d{3})*(\.\d{2})?)", text)
    if amount_match:
        metadata["Amount"] = amount_match.group(2)

    # Address
    address_match = re.search(r"(Address|Location)\s*[:\-]?\s*([\w\s,]+(?:\d{5}|\d{4}))", text)
    if address_match:
        metadata["Address"] = address_match.group(2)

    return metadata