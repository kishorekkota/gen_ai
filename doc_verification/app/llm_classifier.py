import vertexai
from vertexai.generative_models import GenerativeModel

# Initialize Vertex AI
PROJECT_ID = "ai-experimentation-428115"  # Replace with your GCP project ID
REGION = "us-central1"                   # Replace with your desired region
vertexai.init(project=PROJECT_ID, location=REGION)

# Define the candidate labels
CANDIDATE_LABELS = [
    "bank account statement",
    "W-2 tax form",
    "cell-phone bill",
    "school-enrollment certificate",
    "employee payslip",
    "1099 tax form",
    "utility bill - water",
    "utility bill - electricity",
    "utility bill - gas",
]

# Few-shot examples
FEW_SHOT_EXAMPLES = [
    {
        "document": """ACME BANK
Account Statement – January 2025
Beginning balance: 7,500.00
Deposits & Credits: 2,000.50
Withdrawals: 1,250.25
Ending balance: 8,250.25
""",
        "label": "bank account statement",
    },
    {
        "document": """Form W-2, Wage and Tax Statement
Box 1: Wages, tips, other comp
Box 2: Federal income tax withheld
Employer's name, address, and ZIP code
""",
        "label": "W-2 tax form",
    },
    {
        "document": """PAY SLIP
Employee: John Smith
Pay Date: 07/15/2023
Gross Pay: 2,400.00
Deductions: 500.00
Net Pay: 1,900.00
""",
        "label": "employee payslip",
    },
    {
        "document": """Verizon Wireless
Cell Phone Bill – March 2025
Account Number: 123456789
Total Due: $85.50
Due Date: 03/25/2025
""",
        "label": "cell-phone bill",
    },
    {
        "document": """School Enrollment Certificate
Student Name: Jane Doe
School Name: Springfield High School
Enrollment Date: 08/15/2024
""",
        "label": "school-enrollment certificate",
    },
    {
        "document": """Form 1099-MISC
Payer's Name: ABC Corporation
Recipient's Name: John Doe
Nonemployee Compensation: $5,000.00
""",
        "label": "1099 tax form",
    },
    {
        "document": """Utility Bill
Provider: ACME Water Services
Account Number: 987654321
Total Due: $45.75
Due Date: 04/15/2025
""",
        "label": "utility bill - water",
    },
    {
        "document": """Utility Bill
Provider: ACME Electricity
Account Number: 123456789
Total Due: $120.50
Due Date: 04/20/2025
""",
        "label": "utility bill - electricity",
    },
    {
        "document": """Utility Bill
Provider: ACME Gas Services
Account Number: 567890123
Total Due: $75.25
Due Date: 04/25/2025
""",
        "label": "utility bill - gas",
    },
]

def build_few_shot_prompt(ocr_text: str) -> str:
    """
    Builds a prompt with few-shot examples to classify the OCR text.
    """
    print("[DEBUG] Building few-shot prompt...")
    examples_block = []
    for example in FEW_SHOT_EXAMPLES:
        examples_block.append(
            f"""Document:
{example['document']}
Label: {example['label']}

"""
        )
    classification_labels = ", ".join(CANDIDATE_LABELS)

    prompt = f"""You are a helpful AI document classifier. Your task is to classify documents into one of the following categories:
{classification_labels}

Below are some examples:

{''.join(examples_block)}

Now classify the following document:
Document:
{ocr_text}

Please respond with the single best label from the categories: {classification_labels}.
"""
    print("[DEBUG] Prompt built successfully.")
    return prompt

def classify_document_with_gemini(ocr_text: str) -> str:
    """
    Classifies an OCR text into one of the predefined categories using Vertex AI's Gemini model.
    """
    print("[DEBUG] Starting document classification...")
    print("[DEBUG] OCR text length:", len(ocr_text))
    print("[DEBUG] OCR text snippet:", ocr_text[:100])  # Print first 100 characters for brevity
    prompt_text = build_few_shot_prompt(ocr_text)

    print("[DEBUG] Prompt text for classification:")
    print(prompt_text)
    classifier = GenerativeModel(model_name="gemini-2.0-flash-001")

    print("[DEBUG] Sending prompt to Gemini model...")
    response = classifier.generate_content(prompt_text)

    # Extract the predicted label
    predicted_label = response.text.strip()
    print(f"[DEBUG] Model response: {predicted_label}")

    # Validate the predicted label
    if predicted_label not in CANDIDATE_LABELS:
        print(f"[WARNING] Unexpected label: {predicted_label}")
        return "Unknown"

    print(f"[DEBUG] Classification successful. Predicted label: {predicted_label}")
    return predicted_label

# Example usage
if __name__ == "__main__":
    # Sample OCR text
    sample_ocr_text = """
    Utility Bill
    Provider: ACME Electricity
    Account Number: 123456789
    Total Due: $120.50
    Due Date: 04/20/2025
    """

    print("[DEBUG] Running example classification...")
    predicted_label = classify_document_with_gemini(sample_ocr_text)
    print("Predicted Label:", predicted_label)