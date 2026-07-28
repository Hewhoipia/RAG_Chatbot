"""
FILE: domain_banking.py
DESCRIPTION: Banking domain keyword dictionary for query classification.
    Maps 15 business domains to their characteristic keywords (Vietnamese + English)
    and provides keyword matching functions with word-boundary safety for short terms.

HOW IT WORKS:
1. BANKING_DOMAINS dict defines keywords per domain (lowercase, Vietnamese → English)
2. BANKING_ACTIONS list captures common banking action verbs
3. ALL_BANKING_TERMS aggregates everything into a flat set for quick membership checks
4. Short keywords (<=4 chars) use compiled regex with word-boundary anchors to avoid
   false-positive substring matches (e.g. "ai" in "tổng thống Mỹ là AI?")

IMPLEMENTED CLASSES & FUNCTIONS:
- _keyword_in_query(kw, query_lower): Check if a keyword exists in query with boundary safety
- count_domain_matches(query_lower): Count how many domains match and return {domain: hit_count}
"""

# ===========================================================================
# 15 mảng nghiệp vụ ngân hàng
# Mỗi key là tên mảng, value là danh sách từ khoá đặc trưng (lowercase)
# Cấu trúc: [tiếng Việt] → [Tiếng Anh / thuật ngữ quốc tế]
#
# QUY TẮC THÊM TỪ KHOÁ:
# 1. Lowercase toàn bộ
# 2. Ưu tiên cụm từ >= 5 ký tự để tránh false-positive
# 3. Từ ngắn (<= 4 ký tự) sẽ dùng word-boundary regex (ví dụ: "lcr", "npl")
# 4. Không thêm từ quá chung chung ("tiền", "ngân hàng", "bank")
#    vì chúng xuất hiện ở mọi domain → gây multi-domain false alarm
# ===========================================================================

BANKING_DOMAINS = {
    # -----------------------------------------------------------------
    # 1. BANCASSURANCE - Bảo hiểm qua kênh ngân hàng
    # -----------------------------------------------------------------
    "bancassurance": [
        # tiếng Việt
        "bảo hiểm", "bảo hiểm nhân thọ", "bảo hiểm phi nhân thọ",
        "bảo hiểm khoản vay", "bảo hiểm tín dụng", "bảo hiểm cháy nổ",
        "bảo hiểm tài sản", "bảo hiểm sức khỏe", "bảo hiểm tai nạn",
        "phí bảo hiểm", "quyền lợi bảo hiểm", "bồi thường bảo hiểm",
        "hợp đồng bảo hiểm", "đại lý bảo hiểm", "hoa hồng bảo hiểm",
        "bancassurance",
        # tiếng Anh
        "insurance", "life insurance", "non-life insurance",
        "loan insurance", "credit insurance", "fire insurance",
        "insurance premium", "insurance claim", "insurance policy",
        "underwriting",
    ],

    # -----------------------------------------------------------------
    # 2. CAPITAL MARKET - Thị trường vốn & đầu tư
    # -----------------------------------------------------------------
    "capital_market": [
        # tiếng Việt
        "trái phiếu", "cổ phiếu", "chứng khoán", "quỹ đầu tư",
        "trái phiếu doanh nghiệp", "trái phiếu chính phủ",
        "thị trường vốn", "thị trường tiền tệ",
        "chứng chỉ tiền gửi", "chứng chỉ quỹ",
        "repo", "mua bán lại", "giao dịch ký quỹ",
        "danh mục đầu tư", "phái sinh", "quyền chọn",
        "hợp đồng tương lai", "lãi coupon", "mệnh giá",
        "niêm yết", "phát hành", "ipo",
        # tiếng Anh
        "bond", "equity", "securities", "mutual fund",
        "corporate bond", "government bond", "treasury bill",
        "capital market", "money market",
        "certificate of deposit", "fund certificate",
        "margin trading", "portfolio", "derivative",
        "option", "futures", "coupon rate", "par value",
        "listing", "issuance",
    ],

    # -----------------------------------------------------------------
    # 3. CARD - Thẻ ngân hàng
    # -----------------------------------------------------------------
    "card": [
        # tiếng Việt
        "thẻ tín dụng", "thẻ ghi nợ", "thẻ trả trước",
        "thẻ quốc tế", "thẻ nội địa", "thẻ phụ",
        "hạn mức thẻ", "phí thường niên", "phí thẻ",
        "rút tiền mặt", "thanh toán thẻ", "sao kê thẻ",
        "khóa thẻ", "mở thẻ", "gia hạn thẻ",
        "số thẻ", "mã pin", "mã cvv",
        "hoàn tiền", "cashback", "trả góp qua thẻ",
        "chương trình điểm", "điểm thưởng", "tích điểm",
        "contactless", "thẻ chip",
        # tiếng Anh
        "credit card", "debit card", "prepaid card",
        "visa", "mastercard", "jcb",
        "card limit", "annual fee", "card statement",
        "card activation", "card renewal",
        "cash advance", "installment",
        "reward points", "loyalty program",
    ],

    # -----------------------------------------------------------------
    # 4. CUSTOMER SERVICE - Dịch vụ khách hàng
    # -----------------------------------------------------------------
    "customer_service": [
        # tiếng Việt
        "khiếu nại", "phản hồi", "khách hàng",
        "chăm sóc khách hàng", "tổng đài", "hotline",
        "dịch vụ khách hàng", "trải nghiệm khách hàng",
        "hài lòng khách hàng", "survey khách hàng",
        "giải quyết khiếu nại", "phản ánh",
        "thời gian xử lý", "cam kết dịch vụ",
        "sla", "nps", "csat",
        # tiếng Anh
        "complaint", "customer service", "call center",
        "customer experience", "customer satisfaction",
        "service level agreement", "feedback",
        "ticket", "escalation",
    ],

    # -----------------------------------------------------------------
    # 5. DEPOSIT - Tiền gửi
    # -----------------------------------------------------------------
    "deposit": [
        # tiếng Việt
        "tiền gửi", "tiết kiệm", "tiền gửi có kỳ hạn",
        "tiền gửi không kỳ hạn", "tiền gửi thanh toán",
        "sổ tiết kiệm", "lãi suất tiền gửi", "lãi suất tiết kiệm",
        "kỳ hạn gửi", "đáo hạn", "tái tục",
        "tự động gia hạn", "lãi nhập gốc", "lãi trả trước",
        "lãi trả sau", "lãi trả định kỳ",
        "tài khoản tiết kiệm", "tài khoản thanh toán",
        "tài khoản tiền gửi", "tài khoản vãng lai",
        "số dư tối thiểu", "phí duy trì tài khoản",
        "mở tài khoản", "đóng tài khoản",
        "casa",
        # tiếng Anh
        "deposit", "savings", "savings account",
        "term deposit", "fixed deposit",
        "demand deposit", "current account",
        "deposit rate", "maturity", "rollover",
        "auto renewal", "compound interest",
        "minimum balance", "account maintenance fee",
    ],

    # -----------------------------------------------------------------
    # 6. DIGITAL MARKETING - Tiếp thị số
    # -----------------------------------------------------------------
    "digital_marketing": [
        # tiếng Việt
        "tiếp thị số", "marketing online", "chiến dịch marketing",
        "quảng cáo", "khuyến mãi", "chương trình ưu đãi",
        "email marketing", "sms marketing",
        "social media", "mạng xã hội",
        "landing page", "conversion rate",
        "kênh phân phối số", "ngân hàng số",
        "mobile banking", "internet banking",
        "ứng dụng ngân hàng", "app ngân hàng",
        "ebanking", "e-banking", "digital banking",
        "onboarding", "ekyc",
        # tiếng Anh
        "digital marketing", "online marketing",
        "campaign", "promotion", "advertisement",
        "digital channel", "digital banking",
        "mobile app", "online banking",
    ],

    # -----------------------------------------------------------------
    # 7. FINANCE & ACCOUNTING - Tài chính kế toán
    # -----------------------------------------------------------------
    "finance_accounting": [
        # tiếng Việt
        "kế toán", "tài chính", "báo cáo tài chính",
        "bảng cân đối kế toán", "kết quả kinh doanh",
        "lưu chuyển tiền tệ", "thuyết minh tài chính",
        "thuế", "thuế thu nhập", "thuế giá trị gia tăng",
        "hạch toán", "bút toán", "sổ cái",
        "chi phí dự phòng", "dự phòng rủi ro",
        "trích lập dự phòng", "khấu hao",
        "thu nhập lãi thuần", "chi phí hoạt động",
        "tỷ lệ chi phí trên thu nhập",
        "lợi nhuận trước thuế", "lợi nhuận sau thuế",
        "vốn chủ sở hữu", "vốn điều lệ",
        "nii", "nim", "cir", "roe", "roa",
        # tiếng Anh
        "accounting", "financial report", "balance sheet",
        "income statement", "cash flow statement",
        "tax", "income tax", "vat",
        "general ledger", "provision",
        "loan loss provision", "depreciation",
        "net interest income", "operating expense",
        "cost to income ratio", "profit before tax",
        "return on equity", "return on assets",
        "ifrs", "gaap",
    ],

    # -----------------------------------------------------------------
    # 8. HR & GENERAL AFFAIR - Nhân sự & Tổng vụ
    # -----------------------------------------------------------------
    "hr_general_affair": [
        # tiếng Việt
        "nhân sự", "tuyển dụng", "hợp đồng lao động",
        "lương", "thưởng", "phúc lợi",
        "bảo hiểm xã hội", "bảo hiểm y tế",
        "nghỉ phép", "ngày phép", "chấm công",
        "đào tạo", "đánh giá nhân sự", "kpi",
        "bổ nhiệm", "thăng chức", "luân chuyển",
        "thử việc", "sa thải", "nghỉ việc",
        "quản lý nhân sự", "biên chế",
        "tổng vụ", "hành chính", "văn phòng phẩm",
        "quản lý tài sản cố định", "mua sắm",
        # tiếng Anh
        "human resources", "recruitment", "labor contract",
        "salary", "bonus", "benefits",
        "social insurance", "health insurance",
        "annual leave", "attendance", "training",
        "performance review", "promotion", "rotation",
        "probation", "termination", "headcount",
        "general affair", "office management",
    ],

    # -----------------------------------------------------------------
    # 9. INTERNAL CONTROL - Kiểm soát nội bộ
    # -----------------------------------------------------------------
    "internal_control": [
        # tiếng Việt
        "kiểm soát nội bộ", "kiểm toán nội bộ",
        "tuân thủ", "rủi ro hoạt động",
        "phòng chống rửa tiền", "chống rửa tiền",
        "chống tài trợ khủng bố",
        "kyc", "cdd", "ekyc",
        "gian lận", "phòng chống gian lận",
        "báo cáo nghi ngờ", "giao dịch đáng ngờ",
        "hạn mức rủi ro", "rủi ro tín dụng",
        "rủi ro thị trường", "rủi ro thanh khoản",
        "quản lý rủi ro", "khẩu vị rủi ro",
        "kiểm tra", "giám sát",
        "npl", "car", "lcr", "nsfr",
        # tiếng Anh
        "internal control", "internal audit",
        "compliance", "operational risk",
        "anti money laundering", "aml",
        "counter terrorism financing", "cft",
        "know your customer", "customer due diligence",
        "fraud", "fraud prevention",
        "suspicious transaction report", "str",
        "risk management", "risk appetite",
        "credit risk", "market risk", "liquidity risk",
        "non-performing loan", "capital adequacy ratio",
        "liquidity coverage ratio",
    ],

    # -----------------------------------------------------------------
    # 10. IT - Công nghệ thông tin
    # -----------------------------------------------------------------
    "it": [
        # tiếng Việt
        "công nghệ thông tin", "hệ thống", "phần mềm",
        "cơ sở dữ liệu", "máy chủ", "mạng",
        "bảo mật", "an ninh thông tin", "an toàn thông tin",
        "mật khẩu", "xác thực", "otp",
        "lỗi hệ thống", "sự cố", "downtime",
        "core banking", "t24", "temenos",
        "api", "tích hợp hệ thống", "middleware",
        "sao lưu", "khôi phục", "disaster recovery",
        "chuyển đổi số", "dự án it",
        # tiếng Anh
        "information technology", "system", "software",
        "database", "server", "network",
        "cybersecurity", "information security",
        "password", "authentication", "two-factor",
        "system error", "incident", "uptime",
        "core banking system", "integration",
        "backup", "recovery", "firewall",
        "digital transformation",
    ],

    # -----------------------------------------------------------------
    # 11. LOAN - Tín dụng / Cho vay
    # -----------------------------------------------------------------
    "loan": [
        # tiếng Việt
        "cho vay", "khoản vay", "vay vốn",
        "vay tiêu dùng", "vay mua nhà", "vay mua ô tô",
        "vay mua xe", "vay thế chấp", "vay tín chấp",
        "vay doanh nghiệp", "vay sản xuất kinh doanh",
        "hạn mức tín dụng", "giải ngân", "thu nợ",
        "lãi suất vay", "lãi suất cho vay",
        "kỳ hạn vay", "thời hạn vay",
        "trả góp", "trả nợ", "lịch trả nợ",
        "nợ xấu", "nợ quá hạn", "gia hạn nợ",
        "cơ cấu lại nợ", "xử lý nợ",
        "tài sản bảo đảm", "tài sản thế chấp",
        "định giá tài sản", "thẩm định",
        "hồ sơ vay", "phê duyệt tín dụng",
        "hợp đồng tín dụng", "khế ước nhận nợ",
        "tỷ lệ cho vay", "ltv", "dscr",
        "consumer loan", "equipment term loan",
        # tiếng Anh
        "loan", "lending", "credit",
        "mortgage", "auto loan", "car loan",
        "consumer loan", "corporate loan",
        "credit limit", "disbursement", "repayment",
        "loan interest rate", "loan tenor",
        "installment", "overdue", "restructuring",
        "collateral", "appraisal", "credit approval",
        "loan to value", "debt service coverage",
    ],

    # -----------------------------------------------------------------
    # 12. ORGANIZATION & MANAGEMENT - Tổ chức & Quản lý
    # -----------------------------------------------------------------
    "organization_management": [
        # tiếng Việt
        "sơ đồ tổ chức", "cơ cấu tổ chức",
        "phòng ban", "chi nhánh", "hội sở",
        "ban giám đốc", "hội đồng quản trị",
        "ban điều hành", "tổng giám đốc",
        "quy chế", "nội quy", "điều lệ",
        "ủy ban", "ban kiểm soát",
        "r&r", "r & r", "vai trò", "trách nhiệm",
        "phân quyền", "ủy quyền",
        "mô hình tổ chức", "restructuring",
        # tiếng Anh
        "organization", "organizational structure",
        "department", "branch", "head office",
        "board of directors", "management board",
        "ceo", "cfo", "coo", "cro", "cto",
        "charter", "regulation",
        "committee", "supervisory board",
        "role", "responsibility", "delegation",
        "governance",
    ],

    # -----------------------------------------------------------------
    # 13. REMITTANCE - Chuyển tiền
    # -----------------------------------------------------------------
    "remittance": [
        # tiếng Việt
        "chuyển tiền", "chuyển khoản", "chuyển tiền quốc tế",
        "chuyển tiền trong nước", "chuyển tiền nội địa",
        "phí chuyển tiền", "phí chuyển khoản",
        "người gửi", "người nhận", "người thụ hưởng",
        "số tài khoản", "mã ngân hàng",
        "thanh toán liên ngân hàng", "napas",
        "thanh toán quốc tế",
        "tỷ giá", "ngoại tệ", "quy đổi ngoại tệ",
        "mua ngoại tệ", "bán ngoại tệ",
        "swift", "swift code",
        # tiếng Anh
        "remittance", "transfer", "wire transfer",
        "international transfer", "domestic transfer",
        "transfer fee", "beneficiary",
        "account number", "bank code",
        "interbank payment", "cross-border payment",
        "exchange rate", "foreign currency", "forex",
        "fx rate", "spot rate",
    ],

    # -----------------------------------------------------------------
    # 14. TRADE FINANCE - Tài trợ thương mại
    # -----------------------------------------------------------------
    "trade_finance": [
        # tiếng Việt
        "tài trợ thương mại", "thư tín dụng",
        "l/c", "letter of credit",
        "bảo lãnh ngân hàng", "bảo lãnh thanh toán",
        "bảo lãnh thực hiện hợp đồng", "bảo lãnh dự thầu",
        "nhờ thu", "nhờ thu kèm chứng từ", "nhờ thu trơn",
        "chiết khấu", "chiết khấu bộ chứng từ",
        "xuất khẩu", "nhập khẩu",
        "chứng từ xuất nhập khẩu",
        "vận đơn", "hóa đơn thương mại",
        "giấy chứng nhận xuất xứ",
        "thanh toán xuất nhập khẩu",
        "incoterms", "ucp 600",
        # tiếng Anh
        "trade finance", "letter of credit",
        "bank guarantee", "performance guarantee",
        "bid bond", "advance payment guarantee",
        "documentary collection", "clean collection",
        "negotiation", "discounting",
        "export", "import", "shipping document",
        "bill of lading", "commercial invoice",
        "certificate of origin",
        "standby letter of credit", "sblc",
    ],

    # -----------------------------------------------------------------
    # 15. LEGAL & PROPERTY - Pháp lý & Tài sản
    # -----------------------------------------------------------------
    "legal_property": [
        # tiếng Việt
        "pháp lý", "pháp chế", "hợp đồng",
        "văn bản pháp luật", "quy định pháp luật",
        "luật ngân hàng", "luật tín dụng",
        "luật doanh nghiệp", "luật đất đai",
        "tranh chấp", "kiện tụng", "tố tụng",
        "giấy chứng nhận quyền sử dụng đất",
        "sổ đỏ", "sổ hồng",
        "quyền sở hữu", "đăng ký giao dịch bảo đảm",
        "tài sản đảm bảo", "công chứng",
        "hợp đồng thế chấp", "hợp đồng cầm cố",
        "giải chấp", "phát mãi tài sản",
        "định giá tài sản", "thẩm định giá",
        # tiếng Anh
        "legal", "legal department", "contract",
        "regulation", "banking law", "credit law",
        "enterprise law", "land law",
        "dispute", "litigation", "lawsuit",
        "land use right certificate",
        "property ownership", "secured transaction",
        "collateral", "notarization",
        "mortgage contract", "pledge contract",
        "collateral release", "foreclosure",
        "property valuation", "appraisal",
    ],
}

# ===========================================================================
# BANKING ACTIONS - Động từ hành động liên quan ngân hàng
# Khi xuất hiện kèm domain keyword → query có intent rõ ràng
# ===========================================================================
BANKING_ACTIONS = [
    # Tra cứu / Tìm kiếm
    "tra cứu", "kiểm tra", "tìm hiểu", "xem",
    "tra soát", "đối chiếu",
    # Thao tác tài khoản
    "mở tài khoản", "đóng tài khoản", "khóa tài khoản",
    "kích hoạt", "hủy", "đăng ký",
    # Giao dịch
    "chuyển tiền", "rút tiền", "nộp tiền", "gửi tiền",
    "thanh toán", "nạp tiền",
    # Quản lý sản phẩm
    "đăng ký vay", "tất toán", "gia hạn",
    "cập nhật", "thay đổi", "điều chỉnh",
    # Yêu cầu hỗ trợ
    "yêu cầu", "đề nghị", "xin cấp",
    "xác nhận", "phê duyệt",
]

# ===========================================================================
# AGGREGATE SET - Tổng hợp tất cả banking terms vào 1 flat set
# ===========================================================================
ALL_BANKING_TERMS = set()
for domain_terms in BANKING_DOMAINS.values():
    ALL_BANKING_TERMS.update(domain_terms)
ALL_BANKING_TERMS.update(BANKING_ACTIONS)

# ===========================================================================
# WORD-BOUNDARY MATCHING - Giải quyết lỗi false-positive cho từ ngắn
# ===========================================================================
# Vấn đề: substring matching đơn giản ('kw in query') gây false-positive:
#   - "ai" match trong "tổng thống Mỹ là AI?"
#   - "nim" match trong "minimum balance"
#   - "car" match trong "cartoon"
# Giải pháp: từ khoá ngắn (<= 4 ký tự) dùng regex \\b word-boundary

import re

SHORT_KW_THRESHOLD = 4
_short_kw_patterns = {}
for kw in ALL_BANKING_TERMS:
    if len(kw) <= SHORT_KW_THRESHOLD:
        _short_kw_patterns[kw] = re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE)


def _keyword_in_query(kw: str, query_lower: str) -> bool:
    """
    Check if a keyword exists in the query string with word-boundary safety for short terms.

    Short keywords (<=4 chars) use compiled regex with word-boundary anchors to prevent
    false-positive substring matches. Longer keywords use simple 'in' operator.

    Args:
        kw (str): Banking keyword to search for (lowercase).
        query_lower (str): User query string already lowered.

    Returns:
        bool: True if the keyword is found in the query with appropriate matching.
    """
    if len(kw) <= SHORT_KW_THRESHOLD:
        pattern = _short_kw_patterns.get(kw)
        if pattern:
            return bool(pattern.search(query_lower))
        return False
    else:
        return kw in query_lower


def count_domain_matches(query_lower: str) -> dict:
    """
    Count how many keywords match per domain for a given query.

    Iterates over all domains and their keywords, returning a dict of
    {domain_name: hit_count} for domains that have at least one keyword match.

    Args:
        query_lower (str): User query string already lowered.

    Returns:
        dict: Mapping of matched domain names to their keyword hit counts.
              Empty dict if no banking domain keywords are found.
    """
    matched = {}
    for domain, kws in BANKING_DOMAINS.items():
        for kw in kws:
            if _keyword_in_query(kw, query_lower):
                matched[domain] = matched.get(domain, 0) + 1
    return matched