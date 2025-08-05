import re

def extract_fields(text: str) -> dict:
    vendor = text.split('\n')[0] if text else "Unknown Vendor"
    total = re.search(r'(Total|TOTAL|Amount).*?(\d+\.\d{2})', text, re.IGNORECASE)
    date = re.search(r'\b(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})\b', text)
    return {
        'vendor': vendor.strip(),
        'total': float(total.group(2)) if total else None,
        'date': date.group(1) if date else None
    }
