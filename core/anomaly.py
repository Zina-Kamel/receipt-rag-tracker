def detect_anomaly(new_receipt, all_receipts, threshold=1.5):
    vendor = new_receipt["vendor"]
    vendor_totals = [r["total"] for r in all_receipts if r["vendor"] == vendor and r["total"]]

    if not vendor_totals or not new_receipt["total"]:
        return False

    avg = sum(vendor_totals) / len(vendor_totals)
    return new_receipt["total"] > threshold * avg
