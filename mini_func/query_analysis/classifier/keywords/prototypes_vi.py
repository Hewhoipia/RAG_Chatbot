# =============================================================================
# FILE: prototypes_vi.py
# =============================================================================
#
# GIỚI THIỆU (Introduction):
# --------------------------
# Chứa danh sách câu mẫu (Anchor Queries / Prototype Sentences) cho
# 4 nhóm phân loại chính: Chit-chat, Simple, Complex, Clarify.
# Các câu này được dùng để tính Centroid Embedding vectors cho
# Semantic Router (Qwen3-Embedding-4B).
#
# MỤC ĐÍCH (Purpose):
# -------------------
# 1. Làm dữ liệu gốc cho script build_router_cache.py
# 2. Qwen3-Embedding-4B sẽ embedding các câu này → tính trung bình → 4 Centroids
# 3. Khi user hỏi, so sánh Cosine Similarity với 4 Centroids để phân loại
#
# CÁCH SỬ DỤNG (Usage):
# ----------------------
# from classifier.keywords.prototypes_vi import (
#     CHITCHAT_SENTENCES, SIMPLE_SENTENCES, COMPLEX_SENTENCES,
#     CLARIFY_SENTENCES
# )
#
# LƯU Ý (Notes):
# ---------------
# - Mỗi nhóm nên có 10-15 câu đa dạng, phủ đều các pattern phổ biến
# - Câu mẫu nên sát thực tế nhất có thể (domain Ngân hàng Shinhan)
# - Khi thêm/sửa câu mẫu, cần chạy lại build_router_cache.py
# - [V5] Thêm CLARIFY_SENTENCES cho centroid thứ 4 (bare noun, mơ hồ)
# - [V5] Trộn 30% câu chat-style (không dấu câu) vào SIMPLE + COMPLEX
# =============================================================================


# =============================================================================
# NHÓM 1: CHIT-CHAT (Câu xã giao, cảm xúc, ngoài phạm vi ngân hàng)
# =============================================================================
# Bao gồm: Chào hỏi, cảm ơn, cảm xúc cá nhân, câu hỏi ngoài domain
# (chính trị, thời tiết, cá nhân...). Những câu này KHÔNG CẦN tra cứu CSDL.
CHITCHAT_SENTENCES = [
    # --- Chào hỏi & Xã giao ---
    "Chào bạn",
    "Xin cảm ơn",
    "Tôi nhớ bạn",
    "Mình yêu bạn",
    "Xin chào, bạn có thể giúp mình không?",
    "Hello, ai đang trực vậy?",
    "Bạn khỏe không?",
    # --- Cảm xúc & Phản hồi ---
    "Cảm ơn bạn nhiều nhé!",
    "Haha hay quá bạn ơi!",
    "Trời ơi tuyệt vời luôn",
    "Ok mình hiểu rồi, cảm ơn",
    "Hôm nay em buồn lắm, công việc tệ quá",
    # --- Ngoài phạm vi (Off-domain) ---
    "Tổng thống Mỹ là ai?",
    "Hôm nay thời tiết thế nào?",
    "Bạn có người yêu chưa?",
    "Ad rảnh không nói chuyện chơi?",
    "Bạn biết tôi là ai không?",
    "Bạn có thể cho tôi mượn tiền không?",
    "Anh Nguyễn Văn Hoàng là ai ?", 
    "Anh Nguyễn Văn Hoàng đang làm ở phòng ban nào ?",
    "Bạn là ai thế",
    "Đang mấy giờ thế",
    "1 + 1 bằng bao nhiêu",
    "Top các ngân hàng mạnh nhất Việt Nam hiện nay?",
    "Hôm nay có tin tức gì hot không?",
    "Hôm nay nên ăn gì"
]

# =============================================================================
# NHÓM 2: SIMPLE (Câu hỏi đơn giản, Factoid/Definition, 1 chủ đề)
# =============================================================================
# ĐỊNH NGHĨA KỸ:
# - Là các truy vấn Tra cứu (Lookup) chỉ cần 1 tài liệu/bảng biểu là ra ngay.
# - Bao gồm: Nghĩa là gì (Definition), Bao nhiêu (Fact), Những gì (Checklist).
# - KIÊNG KỴ: KHÔNG chứa các từ nối phức tạp (Quy trình, Hướng dẫn, Tại sao,
#   Nếu thì, So sánh) vì những từ này thuộc phạm vi phức tạp (COMPLEX).
#
# --- UPDATE: 
# Mỗi domain có đủ 4 loại pattern:
# Definition: "X là gì?", "Khái niệm Y?"
# Factoid/Lookup: "Phí X là bao nhiêu?", "Hạn mức Y tối đa?"
# Yes/No: "Có thể… không?", "X có được… không?"
# Checklist: "Hồ sơ X cần những gì?"
SIMPLE_SENTENCES = [
    ## Loan
    # --- Definition ---
    "DTI (Debt-to-Income ratio) là gì và ngưỡng tối đa cho vay tiêu dùng Shinhan là bao nhiêu?",
    "Ân hạn nợ gốc (grace period) trong khoản vay mua nhà là gì?",
    "DSCR (Debt Service Coverage Ratio) được tính theo công thức nào?",
    "Khái niệm LTV (Loan-to-Value) trong thẩm định bất động sản là gì?",
    "Nợ nhóm 3 theo phân loại CIC có nghĩa là gì?",
    # --- Factoid / Lookup ---
    "Hạn mức vay tín chấp tối đa dành cho cá nhân tại Shinhan là bao nhiêu?",
    "Phí phạt tất toán khoản vay trước hạn hiện tại là bao nhiêu phần trăm?",
    "Lãi suất vay mua nhà cố định năm đầu tiên của Shinhan là bao nhiêu?",
    "Thời hạn vay mua xe tối đa tại Shinhan là bao nhiêu tháng?",
    "Hồ sơ vay tiêu dùng tín chấp cần những giấy tờ gì?",
    "Khoản vay Nhanh Shinhan yêu cầu thu nhập tối thiểu là bao nhiêu?",
    "Tỷ lệ tài trợ (LTV) tối đa khi thế chấp chung cư tại Hà Nội là bao nhiêu?",
    "Vay thấu chi (overdraft) tài khoản thanh toán hạn mức tối đa bao nhiêu?",
    "Bảng lãi suất cho vay doanh nghiệp SME ngắn hạn hiện hành xem ở đâu?",
    # --- Yes / No ---
    "Người nước ngoài đang làm việc tại Việt Nam có được vay tiêu dùng tín chấp tại Shinhan không?",
    "Khách hàng đang có khoản vay tại ngân hàng khác có được vay thêm tại Shinhan không?",
    "Vay mua xe có cần công chứng hợp đồng thế chấp hay không?",
    "Có thể gia hạn kỳ hạn vay sau khi hợp đồng tín dụng đã ký không?",
    # --- Extend
    "Tỷ lệ LTV tối đa cho vay mua nhà là bao nhiêu?",
    "DSCR tối thiểu để duyệt khoản vay doanh nghiệp là bao nhiêu?",
    "Khoản vay có thể ân hạn gốc tối đa bao lâu?",
    "Phí tất toán trước hạn khoản vay là bao nhiêu?",
    "Nợ xấu nhóm 5 có được vay tiếp không?",

    ## Card
    # --- Definition ---
    "Chargeback là gì và khách hàng có quyền yêu cầu trong bao nhiêu ngày?",
    "EMV 3DS (3D Secure) có tác dụng bảo mật gì khi thanh toán online?",
    "Tokenization thẻ (Card Tokenization) trong Google Pay là gì?",
    "Billing cycle của thẻ tín dụng Shinhan được tính như thế nào?",
    "Phòng chờ sân bay (DragonPass) đi kèm thẻ tín dụng hạng nào của Shinhan?",
    # --- Factoid / Lookup ---
    "Hạn mức tín dụng tối đa của thẻ Visa Platinum Shinhan là bao nhiêu?",
    "Phí rút tiền mặt từ thẻ tín dụng tại ATM nội mạng là bao nhiêu phần trăm?",
    "Ngày chốt sao kê và ngày đến hạn thanh toán thẻ tín dụng Shinhan cách nhau bao nhiêu ngày?",
    "Tỷ lệ thanh toán tối thiểu (minimum payment) hàng tháng của thẻ tín dụng Shinhan là bao nhiêu?",
    "Hạn mức thanh toán không tiếp xúc (contactless) tối đa mỗi giao dịch là bao nhiêu?",
    "Cashback tối đa mỗi tháng khi dùng thẻ Shinhan Zone là bao nhiêu?",
    "Thẻ JCB của Shinhan có hỗ trợ Garmin Pay không?",
    "Điểm thưởng Shinhan Reward được quy đổi theo tỷ lệ nào?",
    "Phí phạt trả chậm (late payment fee) tối thiểu là bao nhiêu?",
    # --- Yes / No ---
    "Có thể phát hành thẻ phụ (supplementary card) cho người thân dưới 18 tuổi không?",
    "Thẻ ghi nợ Shinhan có hỗ trợ thanh toán Samsung Pay không?",
    "Thẻ tín dụng đã quá hạn gia hạn có tự động được phát hành lại không?",
    "Thẻ tín dụng có thể đăng ký trả góp 0% sau khi đã thanh toán toàn bộ dư nợ không?",
    # --- Extend
    "Ngày chốt sao kê thẻ tín dụng là ngày nào?",
    "Minimum payment của thẻ tín dụng là bao nhiêu phần trăm?",
    "Phí overlimit fee được tính như thế nào?",
    "Có thể khóa thẻ tạm thời trên app không?",
    "Thẻ virtual card có dùng được POS không?",

    ## Deposit
    # --- Definition ---
    "CASA ratio là gì và tại sao chỉ số này quan trọng với ngân hàng?",
    "Lãi kép (compound interest) trong tiết kiệm tự động tái tục được tính như thế nào?",
    "Tài khoản không hoạt động (dormant account) bị xử lý như thế nào sau bao lâu?",
    "Chứng chỉ tiền gửi (NCD – Negotiable Certificate of Deposit) là gì?",
    # --- Factoid / Lookup ---
    "Lãi suất tiết kiệm kỳ hạn 12 tháng gửi tại quầy hiện tại là bao nhiêu?",
    "Số dư tối thiểu để duy trì tài khoản thanh toán Shinhan là bao nhiêu?",
    "Phí rút tiền trước hạn sổ tiết kiệm kỳ hạn 6 tháng là bao nhiêu?",
    "Lãi suất bậc thang (step-up interest) áp dụng cho kỳ hạn nào?",
    "Mức bảo hiểm tiền gửi tối đa hiện nay theo quy định NHNN là bao nhiêu?",
    "Tài khoản S-Payroll có hưởng lãi suất không kỳ hạn ưu đãi không?",
    "Giới hạn số tiền gửi tiết kiệm online tối đa mỗi lần giao dịch trên SOL là bao nhiêu?",
    # --- Yes / No ---
    "Có thể tất toán sổ tiết kiệm trước hạn qua ứng dụng SOL không?",
    "Tài khoản đồng sở hữu có thể được mở online hay phải ra quầy?",
    "Người nước ngoài có thể mở tài khoản tiết kiệm ngoại tệ USD tại Shinhan không?",
    # --- Extend
    "Số dư tối thiểu để duy trì tài khoản là bao nhiêu?",
    "Có thể tất toán sổ tiết kiệm trước hạn không?",
    "Lãi suất bậc thang là gì?",
    "Tài khoản dormant là gì?",

    ## Capital market
    # --- Definition ---
    "Trái phiếu xanh (Green Bond) là gì và tiêu chuẩn phát hành như thế nào?",
    "Đường cong lợi suất (Yield Curve) phẳng (flat yield curve) có ý nghĩa kinh tế gì?",
    "ESG Investment là gì và Shinhan áp dụng tiêu chí nào trong danh mục tự doanh?",
    "Nhà tạo lập thị trường sơ cấp (Primary Dealer) có nghĩa vụ gì đối với NHNN?",
    "NAV (Net Asset Value) của chứng chỉ quỹ được tính như thế nào?",
 
    # --- Factoid / Lookup ---
    "Hạn mức tự doanh (proprietary trading limit) trái phiếu chính phủ hiện tại là bao nhiêu?",
    "Lãi suất trái phiếu chính phủ kỳ hạn 5 năm phiên đấu thầu gần nhất là bao nhiêu?",
    "Tỷ lệ ký quỹ (margin) tối thiểu khi giao dịch phái sinh lãi suất tại Shinhan là bao nhiêu?",
    "Chứng khoán hóa tài sản (Securitization) cần loại tài sản cơ sở nào mới đủ điều kiện?",
 
    # --- Yes / No ---
    "Shinhan có được phép giao dịch phái sinh tiền tệ cho khách hàng doanh nghiệp không?",
    "Trái phiếu doanh nghiệp phát hành riêng lẻ có thể giao dịch trên thị trường thứ cấp ngay sau khi phát hành không?",

    ## Financial accouting
    # --- Definition ---
    "CIR (Cost-to-Income Ratio) được tính theo công thức nào và mức chuẩn ngành là bao nhiêu?",
    "Accrual basis (cơ sở dồn tích) trong kế toán ngân hàng có nghĩa là gì?",
    "IFRS 9 thay thế IAS 39 với những thay đổi chính nào về phân loại tài sản tài chính?",
    "LCR (Liquidity Coverage Ratio) được tính theo công thức nào?",
    # --- Factoid / Lookup ---
    "Mức trích lập dự phòng rủi ro tín dụng cho nợ nhóm 5 (nợ có khả năng mất vốn) là bao nhiêu phần trăm?",
    "Kỳ báo cáo tài chính quý gửi NHNN phải nộp trong bao nhiêu ngày sau khi kết thúc quý?",
    "Thuế suất thuế thu nhập doanh nghiệp (TNDN) áp dụng cho ngân hàng hiện tại là bao nhiêu?",
    "Hệ số CAR (Capital Adequacy Ratio) tối thiểu theo chuẩn Basel II tại Việt Nam là bao nhiêu?",
    # --- Yes / No ---
    "Chi phí mua sắm laptop cho nhân viên có được ghi nhận là CAPEX hay OPEX?",
    "Khoản dự phòng trích lập theo IFRS 9 có được khấu trừ khi tính thuế TNDN không?",
    # --- Extend ---
    "NIM là gì?",
    "CIR là chỉ số gì?",
    "Provisioning được hạch toán như thế nào?",
    "Deferred tax assets nằm ở báo cáo nào?",

    ## HR
    # --- Definition ---
    "Khung năng lực (Competency Framework) của Shinhan gồm những năng lực cốt lõi nào?",
    "Well-being program là gì và Shinhan đang triển khai những chương trình nào?",
    # --- Factoid / Lookup ---
    "Số ngày nghỉ phép năm tối đa của nhân viên cấp chuyên viên cao cấp (Senior) là bao nhiêu ngày?",
    "Phụ cấp điện thoại hàng tháng cho vị trí cấp quản lý (Manager) là bao nhiêu?",
    "Định mức chi phí đào tạo nội bộ (training budget) theo đầu nhân viên là bao nhiêu?",
    "Thủ tục xin nghỉ phép đặc biệt (hiếu/hỷ) cần nộp trước bao nhiêu ngày?",
    "Hồ sơ onboarding nhân viên mới bộ phận Tín dụng cần bao gồm những gì?",
    "Deadline nộp timesheet hàng tháng là ngày bao nhiêu?",
    "Quy định về làm thêm giờ (overtime) và mức phụ cấp tương ứng là bao nhiêu?",
    "Mẫu đơn xin nghỉ phép không lương tải ở cổng nội bộ nào?",
    # --- Yes / No ---
    "Nhân viên thử việc có được tính phụ cấp đi lại không?",
    "Có thể chuyển ngày nghỉ phép năm sang năm sau không nếu chưa dùng hết?",
    "Nhân viên đang trong thời gian thử việc có được tham gia BHXH không?",
    # --- Extend ---
    "Timesheet nộp ở hệ thống nào?",
    "Chính sách nghỉ phép năm là bao nhiêu ngày?",
    "Expense claim cần những chứng từ gì?",

    ## Internal control
    # --- Definition ---
    "PEP (Politically Exposed Person) là ai và mức độ thẩm định tăng cường (Enhanced Due Diligence) gồm những bước nào?",
    "STR (Suspicious Transaction Report) cần được báo cáo cho cơ quan nào và trong bao nhiêu giờ?",
    "Sanction screening là gì và hệ thống nào đang được sử dụng tại Shinhan?",
    "Segregation of Duties (SoD) trong kiểm soát nội bộ có ý nghĩa là gì?",
    # --- Factoid / Lookup ---
    "Ngưỡng giao dịch tiền mặt phải báo cáo bắt buộc (CTR – Currency Transaction Report) là bao nhiêu?",
    "Tần suất kiểm toán nội bộ đối với chi nhánh hạng A là bao nhiêu lần mỗi năm?",
    "Thời hạn lưu trữ hồ sơ KYC của khách hàng theo quy định AML là bao nhiêu năm?",
    "Ai là người có thẩm quyền phê duyệt miễn trừ kiểm soát nội bộ trong trường hợp khẩn cấp?",
    # --- Yes / No ---
    "Nhân viên phát hiện giao dịch đáng ngờ có bắt buộc phải báo cáo dù không chắc chắn không?",
    "FATCA có áp dụng cho khách hàng cá nhân người Việt Nam mang thẻ xanh (Green Card) Mỹ không?",
    # --- Extend ---
    "PEP là gì?",
    "FATCA áp dụng cho đối tượng nào?",
    "Transaction monitoring là gì?",

    ## BANCASSURANCE
    # --- Definition ---
    "Persistency Rate (tỷ lệ duy trì hợp đồng) là gì và tại sao chỉ số này quan trọng với kênh Banca?",
    "Surrender Value (giá trị hoàn lại) được tính như thế nào nếu khách hàng muốn hủy hợp đồng sớm?",
    "Medical underwriting trong quy trình phát hành hợp đồng bảo hiểm nhân thọ gồm những bước gì?",
    "Người thụ hưởng (Beneficiary) trong hợp đồng bảo hiểm có thể là ai và thay đổi được không?",
 
    # --- Factoid / Lookup ---
    "Hoa hồng đại lý bảo hiểm (bancassurance commission) cho sản phẩm bảo hiểm nhân thọ năm đầu là bao nhiêu phần trăm?",
    "Thời gian cân nhắc (free-look period) của hợp đồng bảo hiểm nhân thọ là bao nhiêu ngày?",
    "Chỉ tiêu KPI tỷ lệ duy trì hợp đồng năm 13 (13-month persistency) mức đạt chuẩn là bao nhiêu phần trăm?",
    "Sản phẩm bảo hiểm nào đang được cross-sell kèm với sản phẩm vay mua nhà của Shinhan?",
 
    # --- Yes / No ---
    "Khách hàng có tiền sử bệnh tim mạch có được mua bảo hiểm nhân thọ qua kênh Banca không?",
    "Hợp đồng bảo hiểm liên kết đầu tư có được bảo vệ bởi bảo hiểm tiền gửi không?",

    ## ORGANIZATION
    # --- Definition ---
    "Business Process Reengineering (BPR) là gì và khi nào ngân hàng nên áp dụng?",
    "Corporate Governance (quản trị công ty) trong ngân hàng gồm những nguyên tắc nào theo Basel?",
    "Agile Organization là gì và Shinhan đang áp dụng mô hình này ở những bộ phận nào?",
 
    # --- Factoid / Lookup ---
    "Hội đồng quản trị Shinhan Bank Việt Nam hiện có bao nhiêu thành viên và cơ cấu như thế nào?",
    "Số lượng chi nhánh và phòng giao dịch của Shinhan tại Việt Nam hiện tại là bao nhiêu?",
    "Phân khúc khách hàng nào thuộc khối Bán lẻ (Retail Banking) và khối Doanh nghiệp (Wholesale Banking)?",
    "Ngưỡng phán quyết tín dụng của Giám đốc Chi nhánh hạng A là bao nhiêu?",
 
    # --- Yes / No ---
    "Giám đốc phòng giao dịch có thẩm quyền ký kết hợp đồng mở tài khoản cho doanh nghiệp không?",
    "Có thể chuyển giao khách hàng giữa các chi nhánh Shinhan khác nhau không?",

    ## REMITTANCE
    # --- Definition ---
    "SWIFT GPI (Global Payment Innovation) cải tiến gì so với SWIFT truyền thống?",
    "IBAN và BIC/SWIFT Code được sử dụng như thế nào trong chuyển tiền quốc tế?",
    "Correspondent Bank (ngân hàng đại lý) đóng vai trò gì trong giao dịch chuyển tiền quốc tế?",
    "VietQR là gì và khác với QR Code thông thường của từng ngân hàng ở điểm nào?",
    # --- Factoid / Lookup ---
    "Hạn mức chuyển tiền quốc tế một ngày qua SOL đối với khách hàng cá nhân là bao nhiêu USD?",
    "Phí chuyển tiền trong nước liên ngân hàng qua NAPAS đối với khoản dưới 10 triệu là bao nhiêu?",
    "Thời gian xử lý lệnh chuyển tiền quốc tế SWIFT thông thường là bao nhiêu ngày làm việc?",
    "Tỷ giá mua và bán USD/VND niêm yết tại Shinhan hôm nay là bao nhiêu?",
    "Hạn mức chuyển tiền qua VietQR một giao dịch tối đa là bao nhiêu?",
    "Phí điện phát SWIFT outgoing wire transfer một chiều là bao nhiêu USD?",
    # --- Yes / No ---
    "Người nước ngoài không có tài khoản tại Shinhan có thể nhận kiều hối qua Western Union tại quầy không?",
    "Có thể hủy lệnh chuyển tiền quốc tế sau khi đã gửi SWIFT không?",
    "Chuyển tiền nhanh 24/7 NAPAS có áp dụng cho tài khoản doanh nghiệp không?",
    # --- Extend ---
    "Thời gian chuyển tiền SWIFT mất bao lâu?",
    "Mã BIC là gì?",
    "Có thể tra cứu trạng thái chuyển tiền ở đâu?",
    "Phí chuyển tiền quốc tế gồm những loại nào?",
    "Tỷ giá mua bán ngoại tệ được cập nhật khi nào?",

    ## TRADE_FINANCE
    # --- Definition ---
    "UCP 600 là gì và điều chỉnh loại giao dịch tài trợ thương mại nào?",
    "Standby L/C (thư tín dụng dự phòng) được sử dụng trong trường hợp nào?",
    # --- Factoid / Lookup ---
    "Phí mở L/C nhập khẩu (Import L/C opening fee) tại Shinhan hiện tại là bao nhiêu?",
    "Thời gian tối đa để kiểm tra bộ chứng từ xuất trình theo L/C theo UCP 600 là bao nhiêu ngày làm việc?",
    "Hạn mức bảo lãnh tối đa cho một khách hàng doanh nghiệp là bao nhiêu phần trăm vốn tự có của ngân hàng?",
    "Chứng từ bắt buộc trong bộ chứng từ xuất khẩu theo L/C thông thường gồm những gì?",
    "Phí chiết khấu bộ chứng từ xuất khẩu (export bill discounting) được tính như thế nào?",
    # --- Yes / No ---
    "L/C at sight có thể chuyển đổi thành L/C trả chậm (Usance L/C) sau khi đã phát hành không?",
    "Bảo lãnh ngân hàng có thể được dùng thay thế ký quỹ tiền mặt trong đấu thầu không?",
    "Shinhan có thể xác nhận (confirm) L/C phát hành bởi ngân hàng nước ngoài chưa có quan hệ đại lý không?",
    # --- Extend ---
    "UCP600 là gì?",
    "Incoterms FOB nghĩa là gì?",
    "Bộ chứng từ xuất khẩu gồm những gì?",
    "Standby L/C dùng để làm gì?",
    "Phí mở L/C được tính như thế nào?",

    # --- IT / System ---
    "Core banking system là gì?",
    "API banking dùng để làm gì?",
    "OTP có hiệu lực trong bao lâu?",
    "SLA xử lý ticket IT là bao nhiêu giờ?",
    "Downtime hệ thống được định nghĩa như thế nào?",

    # --- Định nghĩa (Definition) ---
    "KYC là gì?",
    "Bancassurance nghĩa là sao?",
    "Hệ thống phê duyệt tín dụng (LOS) là gì?",
    "Khái niệm tài sản thuế thu nhập hoãn lại (deferred tax assets)?",
    "What is credit card",

    # --- Tra cứu Factoid (1 domain) ---
    "Lãi suất tiết kiệm không kỳ hạn hiện tại bao nhiêu?",
    "Phí thường niên thẻ tín dụng Visa Classic là bao nhiêu?",
    "Hạn mức chuyển khoản NAPAS một ngày tối đa bao nhiêu?",
    "Phí rút tiền ATM Shinhan nội mạng là bao nhiêu?",
    "Điều kiện mở tài khoản s-payroll cần những gì?",
    "Google Pay có hỗ trợ thẻ Shinhan không?",

    # --- Tra cứu Factoid Hành chính Nhân sự (HR & Admin) ---
    "Mức định mức công tác phí lưu trú khách sạn một ngày cho cấp quản lý là bao nhiêu tiền?",
    "SLA tiêu chuẩn để phòng IT cấp quyền truy cập VPN từ xa trên hệ thống là bao nhiêu giờ?",
    "Consent letter form bộ phận Hành chính nhân sự có thể tải ở trên cổng mạng nội bộ nào?",
    "Hồ sơ offboarding (nghỉ việc) bàn giao cho phòng nhân sự bắt buộc phải nộp những công văn gì?",
    "Allowance đi công tác của nhân viên là bao nhiêu?",
    "Consent letter form có thể tải ở đâu?",
    "Phụ cấp đi công tác được tính như thế nào?",
    "Lãi suất gửi ngoại tệ tại Shinhan là bao nhiêu?",

    # --- Tra cứu Factoid Tuân thủ, Tín dụng & Ngoại hối (Risk, Compliance, FX) ---
    "Báo cáo giao dịch đáng ngờ STR (Suspicious Transaction Report) phải nộp cho NHNN trong bao nhiêu ngày?",
    "Định mức cắt lỗ (stop-loss) trong danh mục tự doanh (proprietary trading) năm nay là bao nhiêu phần trăm?",
    "Mức trích lập dự phòng rủi ro tín dụng (provisioning) hiện tại áp dụng cho nợ nhóm 4 là bao nhiêu?",
    "Ai là người có thẩm quyền phê duyệt giải ngân khoản vay Syndicated loan (cho vay hợp vốn) cấp doanh nghiệp?",
    "Thời gian lưu trữ log biên bản giao dịch chuyển tiền quốc tế SWIFT trên hệ thống ngân hàng lõi tối đa bao nhiêu tháng?",
    "Mã lỗi hệ thống Timeout 504 trên máy chủ Core Banking có ý nghĩa chuyên môn là gì bảng tra cứu ở đâu?",

    # --- Yes/No ---
    "Có thể mở tài khoản tiết kiệm online không?",
    "Doanh nghiệp FDI có nợ xấu nhóm 2 thì có đủ điều kiện đệ trình xin vay vốn tín chấp không?",
    "Giấy tờ vay thế chấp sổ đỏ đồng sở hữu có bắt buộc cần chữ ký của tất cả các bên không?",
    "Thẻ ghi nợ có rút tiền mặt tại ATM khác ngân hàng được không?",
    "Vay tiêu dùng tín chấp có cần thế chấp không?",
    "Chuyển tiền qua Zalo OA có miễn phí không?",
    "Shinhan Zone có ưu đãi trả góp 0% không?",

    # --- Nghiệp vụ chuyên sâu (Trade Finance, Capital Market, Compliance...) ---
    "Quy định mở thư tín dụng L/C (Letter of Credit) nhập khẩu cần chứng từ gì?",
    "Phí chuyển tiền quốc tế qua SWIFT gpi được tính ra sao?",
    "Hệ số thanh khoản LCR (Liquidity Coverage Ratio) hiện tại là bao nhiêu?",
    "Chỉ tiêu KPI và OKR của phòng ban doanh nghiệp xem ở hệ thống nào?",
    "Khái niệm tài sản thuế thu nhập hoãn lại (deferred tax assets) là báo cáo nào?",
    "Ai là người có quyền phê duyệt nghiệp vụ bao thanh toán (Factoring) xuất khẩu?",
    "Nếu nhận được ticket từ Contact Center về tra soát thẻ (Chargeback) thì SLA xử lý là bao lâu?",
    "Quy trình KYC / eKYC dành cho khách hàng nước ngoài (Non-resident) gồm những giấy tờ gì?",
    "Hệ thống AITHER là gì và được sử dụng trong nghiệp vụ nào?",
    "Màn hình 8800 trên Core Banking dùng để tra cứu thông tin gì?",
    "Cách cấp quyền truy cập vào hệ thống AITHER cho nhân viên mới?",
    "Tops club là chương trình gì và điều kiện để trở thành hội viên?",

    # ===== SIMPLE - Definitions & Concepts =====
    "What is a Certificate of Deposit (CD)?",
    "What does KYC stand for?",
    "Define Bancassurance.",
    "What is a Letter of Credit (L/C)?",
    "Explain the term 'provisioning' in banking.",

    # ===== SIMPLE - Factoids (numbers, fees, limits) =====
    "What is the current interest rate for a 12-month savings account?",
    "How much is the annual fee for Shinhan Visa Platinum card?",
    "What is the maximum daily ATM withdrawal limit for a debit card?",
    "How many days does it take to disburse an unsecured online loan?",
    "What is the foreign exchange rate USD/VND today?",
    "How much is the Swift transfer fee for outward remittance?",
    "What is the minimum balance required for a payroll account?",
    "What is the penalty rate for early loan repayment?",
    "How long are SWIFT transaction records kept by regulation?",
    "What is the monthly allowance for a junior manager's business trip?",

    # ===== SIMPLE - Yes/No & Checklists =====
    "Can I open a savings account online without going to a branch?",
    "Is it possible to apply for a mortgage loan as a foreigner?",
    "Does the bank charge a fee for interbank ATM withdrawal?",
    "Do I need to buy credit life insurance when taking a personal loan?",
    "What documents are needed for a corporate account opening?",
    "Can I use Google Pay with Shinhan debit card?",
    "Is there a loyalty points program for credit card spending?",

    # ===== SIMPLE - Internal reference / one-step lookup =====
    "Where can I find the latest SLA document for IT incident response?",
    "Who approves the expense claim for department heads?",
    "What is the approval matrix for loan disbursement above 1 billion VND?",
    "Which internal system stores the KYC form template?",
    "What is the current version of the core banking system?",

    # ===== SIMPLE - Definitions & Concepts =====
    "KYC가 무엇인가요?",  # What is KYC?
    "방카슈랑스란 무엇인가요?",  # What is bancassurance?
    "정기예금(CD)이 무엇인가요?",  # What is a certificate of deposit (CD)?
    "대손충당금이 무엇인가요?",  # What is a loan loss provision?
    "신용점수(CIC)란 무엇인가요?",  # What is a credit score (CIC)?

    # ===== SIMPLE - Factoids =====
    "신한은행의 12개월 정기예금 금리는 얼마인가요?",  # What is the 12-month time deposit interest rate at Shinhan Bank?
    "신한 Visa 플래티넘 카드의 연회비는 얼마인가요?",  # What is the annual fee for the Shinhan Visa Platinum card?
    "현금자동입출금기(ATM) 일일 인출 한도는 얼마인가요?",  # What is the daily ATM withdrawal limit?
    "온라인 무담보 대출 của 실행까지 며칠 걸리나요?",  # How many days does it take to disburse an online unsecured loan? # Wait, let's write exact:
    "온라인 무담보 대출의 실행까지 며칠 걸리나요?",  # How many days does it take to disburse an online unsecured loan?
    "오늘의 USD/KRW 환율은 얼마인가요?",  # What is today's USD/KRW exchange rate?
    "해외 송금 수수료(SWIFT)는 얼마인가요?",  # How much is the SWIFT international transfer fee?
    "급여통장 최소 잔액 요구액은 얼마인가요?",  # What is the minimum balance requirement for a payroll account?
    "대출 중도상환 수수료율은 어떻게 되나요?",  # What is the early repayment penalty rate for a loan?
    "SWIFT 거래 기록은 법적으로 얼마나 보관하나요?",  # How long are SWIFT transaction records legally kept?
    "초급 관리자의 출장비 일당은 얼마인가요?",  # What is the daily business trip allowance for a junior manager?

    # ===== SIMPLE - Yes/No & Checklists =====
    "인터넷으로 비대면 예금 계좌를 개설할 수 있나요?",  # Can I open a deposit account online without visiting a branch?
    "외국인도 주택담보대출을 받을 수 있나요?",  # Can foreigners also get a mortgage loan?
    "다른 은행 ATM에서 출금할 때 수수료가 부과되나요?",  # Is a fee charged when withdrawing from another bank's ATM?
    "신용대출을 받을 때 신용보험에 가입해야 하나요?",  # Do I need to purchase credit life insurance when taking a personal loan?
    "법인 계좌 개설에 필요한 서류는 무엇인가요?",  # What documents are needed to open a corporate account?
    "신한 체크카드를 Google Pay에 등록할 수 있나요?",  # Can I register my Shinhan debit card with Google Pay?
    "신용카드 사용에 대한 포인트 적립이 되나요?",  # Are points earned on credit card spending?

    # ===== SIMPLE - Internal reference =====
    "IT 장애 대응 SLA 문서는 어디서 찾을 수 있나요?",  # Where can I find the SLA document for IT incident response?
    "부서장의 경비 청구는 누가 승인하나요?",  # Who approves the expense claim for a department head?
    "10억 원 이상 대출 실행 승인 매트릭스는 무엇인가요?",  # What is the approval matrix for loan disbursements above 1 billion KRW?
    "KYC 양식은 어느 내부 시스템에 저장되나요?",  # Which internal system stores the KYC form?
    "코어뱅킹 시스템의 현재 버전은 무엇인가요?",  # What is the current version of the core banking system?,
    "What documents are required for a travel insurance claim due to medical emergency?",
    "What are the key KPIs for the customer service department this quarter?",
    "의료 응급 상황에 대한 여행자 보험 청구에 필요한 서류는 무엇인가요?",  # What documents are required for a travel insurance claim due to a medical emergency?,
    "고객 서비스 부서의 이번 분기 주요 KPI는 무엇인가요?",  # What are the key KPIs for the customer service department this quarter?,

    # =================================================================
    # [V5] Chat-style / Không dấu câu (30% mix) — Phản ánh cách user thực tế nhập
    # =================================================================
    "phí thường niên thẻ visa platinum",
    "lãi suất vay mua nhà shinhan hiện tại",
    "hạn mức chuyển khoản napas một ngày",
    "thời gian giải ngân vay tín chấp",
    "tra cứu phí chuyển tiền quốc tế",
    "cho tôi xem sao kê thẻ tín dụng",
    "kiểm tra số dư tài khoản tiết kiệm",
    "phí rút tiền atm khác ngân hàng",
    "otp có hiệu lực trong bao lâu",
    "điều kiện mở tài khoản s-payroll",
    "chào ad lãi suất tiết kiệm 12 tháng",
    "hello phí chuyển tiền swift bao nhiêu",
    "em cần check hạn mức thẻ tín dụng",
    "nhờ tư vấn gói vay mua nhà",
    "mở thẻ ghi nợ quốc tế cần gì",
    # --- R&R Simple Lookup ---
    "Ai chịu trách nhiệm phê duyệt quy trình của phòng DBU?",
    "Who is responsible for the R&R of the Corporate Banking Division?",
    "các phòng ban trong Shinhan Bank là gì?",
    "Phòng DBU là gì",
    "Phòng CBD là gì",
    "Phòng strategy division làm những gì",

    # --- [v3] Implicit Lookup — lookup trigger + specifier (không dấu ?) ---
    # Đây là nhóm câu bị clarify sai nhiều nhất trước v3
    # Pattern: [trigger] + [banking_term] + [specifier] (3-6 từ, không WH-word, không ?)
    "điều kiện vay tín chấp cá nhân",
    "lãi suất vay thế chấp hiện tại",
    "phí thường niên thẻ tín dụng",
    "hạn mức chuyển tiền nhanh 24/7",
    "tỷ giá mua usd hôm nay",
    "mức phạt trả nợ trước hạn",
    "form đăng ký vay tín dụng",
    "sla xử lý khiếu nại khách hàng",
    "điều kiện nhóm 2 cic vay lại",
    "biểu phí dịch vụ ngân hàng điện tử",
    "hạn mức thấu chi tài khoản thanh toán",
    "lãi suất tiết kiệm kỳ hạn 12 tháng",
    "điều kiện vay tín chấp nhân viên",
    "phí chuyển tiền swift nội địa",
    "thời gian giải ngân vay mua nhà",
    "lãi suất cho vay doanh nghiệp sme",
    "tỷ lệ ltv tối đa mua căn hộ",
    "phí phạt tất toán khoản vay tín chấp",
    "điều kiện mở tài khoản doanh nghiệp",
    "hồ sơ đăng ký thẻ tín dụng lần đầu",

    # --- [v3] Implicit Lookup bổ sung theo plan Phase 4.1 ---
    # Các dạng câu văn nói ngắn phổ biến trên chatbot ngân hàng
    # Bao gồm các trigger keyword: phí, hạn mức, lãi suất, điều kiện, tỷ giá, sla, form
    "phí thường niên thẻ platinum",
    "lãi suất vay mua nhà hiện hành",
    "hạn mức chuyển tiền swift",
    "điều kiện vay tín chấp",
    "tỷ giá mua usd hôm nay",
    "mức phạt trả nợ trước hạn vay",
    "sla xử lý khiếu nại",
    "điều kiện nhóm 2 cic",
    "biểu phí atm nội địa",
    "hạn mức thẻ tín dụng tối đa",
    "lãi suất tiết kiệm 6 tháng",
    "phí phát hành thẻ quốc tế",
    "điều kiện đăng ký gói vay sme",
    "thời gian xử lý chuyển tiền quốc tế",
    "phí overdraft tài khoản thanh toán",
    "hạn mức rút tiền atm mỗi ngày",
    "điều kiện tất toán khoản vay sớm",
    "lãi suất vay tiêu dùng tín chấp",
    "phí duy trì tài khoản số dư thấp",
    "điều kiện xét duyệt vay mua xe",

    # --- [v3] Câu ngắn có intent rõ (keyword search style, no ?) ---
    "l/c là gì",
    "swift code ngân hàng",
    "kyc nhóm khách hàng",
    "aml screening quy trình",
    "cic nhóm 2 là gì",
    "ltv là gì",
    "dscr là gì",
    "iban là gì",
    "tin nhắn otp atm",
    "mã bin thẻ tín dụng",
]

# =============================================================================
# NHÓM 3: COMPLEX (Câu hỏi phức tạp, So sánh, Đa bước, Nhân quả)
# =============================================================================
# ĐỊNH NGHĨA KỸ:
# - Phải cần kết hợp chéo nhiều mảng tài liệu hoặc áp dụng logic phân tích/tính toán.
# - Bao gồm cấu trúc có từ nối: So sánh (Khác biệt), Hệ quả (Tại sao, Ảnh hưởng),
#   Giả định / Mệnh đề điều kiện (Nếu/Thì/Trường hợp), Quy trình (Các bước, Làm cách nào).
#
# --- UPDATE: 
# Mỗi domain có đủ 4 kiểu lập luận:
# So sánh: "Khác biệt giữa A và B?", "Phân tích ưu nhược điểm?"
# Quy trình: "Các bước thực hiện X?", "Quy trình Y gồm bao nhiêu vòng?"
# Nhân quả/Điều kiện: "Tại sao X xảy ra?", "Nếu Y thì hậu quả là gì?"
# Tổng hợp: Câu hỏi yêu cầu cross-domain reasoning hoặc tư vấn toàn diện
COMPLEX_SENTENCES = [
    ## Loan 
    # --- So sánh ---
    "So sánh điều kiện vay và lãi suất giữa vay tín chấp tiêu dùng và vay thế chấp mua nhà của Shinhan?",
    "Phân tích sự khác nhau giữa cơ cấu lại nợ (loan restructuring) và tái cấp vốn (refinancing) khi khách hàng gặp khó khăn tài chính?",
    "Vay hợp vốn (Syndicated loan) và vay song phương (bilateral loan) khác nhau ở những điểm nào về quy trình phê duyệt và rủi ro?",
    # --- Quy trình ---
    "Quy trình thẩm định và phê duyệt khoản vay mua nhà thế chấp từ lúc tiếp nhận hồ sơ đến giải ngân gồm bao nhiêu bước?",
    "Làm thế nào để yêu cầu cơ cấu lại nợ khi tôi bị mất việc và không thể trả đúng hạn hàng tháng?",
    "Các bước xử lý nội bộ khi một khoản vay chuyển sang nợ xấu nhóm 4 gồm những vòng phê duyệt nào?",
    "Quy trình bổ sung tài sản đảm bảo (margin call) khi giá trị tài sản thế chấp giảm dưới ngưỡng LTV quy định là như thế nào?",
    # --- Nhân quả / Điều kiện ---
    "Nếu khách hàng trả chậm 35 ngày so với ngày đến hạn, khoản vay sẽ bị chuyển sang nhóm nợ nào và ảnh hưởng CIC ra sao?",
    "Trường hợp doanh nghiệp FDI có tỷ lệ nợ vay trên vốn chủ sở hữu vượt 3:1 thì điều kiện cho vay thêm bị ảnh hưởng như thế nào?",
    "Nếu tôi có lịch sử CIC nhóm 2 trong 12 tháng qua thì khả năng được duyệt vay tín chấp và mức lãi suất áp dụng sẽ thay đổi ra sao?",
    "Tại sao khoản vay của doanh nghiệp bị yêu cầu bổ sung tài sản đảm bảo dù đang trả nợ đầy đủ và đúng hạn?",
    # --- Tổng hợp ---
    "Với mức lương 25 triệu/tháng và không có nợ vay hiện tại, đâu là gói vay phù hợp nhất giữa vay mua nhà, vay mua xe và vay tiêu dùng của Shinhan?",
    "Liệt kê toàn bộ chi phí phát sinh (phí + lãi suất) khi vay mua nhà 3 tỷ kỳ hạn 20 năm lãi suất thả nổi tại Shinhan?",
    "với số tiền muốn vay 2,2 tỷ và thu nhập 50 triệu tháng vay mua nhà với thời gian tôi đã là bao lâu và số tiền phải trả ước tính là bao nhiêu?",

    ## Card
    # --- So sánh ---
    "So sánh lợi ích thực tế giữa thẻ Visa Infinite và Mastercard Platinum Shinhan cho khách hàng hay đi công tác nước ngoài?",
    "Điểm khác biệt về cơ chế hoàn tiền (cashback) và tích điểm đổi thưởng giữa thẻ Shinhan Zone và thẻ hạng chuẩn là gì?",
    "So sánh rủi ro pháp lý và trách nhiệm bồi thường giữa ngân hàng phát hành và ngân hàng thanh toán trong một tranh chấp Chargeback?",
    # --- Quy trình ---
    "Quy trình xử lý nội bộ khi khách hàng khiếu nại giao dịch không nhận được hàng (dispute transaction) gồm những bước nào và SLA là bao lâu?",
    "Các bước đăng ký tính năng trả góp 0% qua thẻ tín dụng tại đối tác Shinhan Zone là gì?",
    "Hướng dẫn chi tiết cách liên kết thẻ Shinhan vào Google Pay và kích hoạt thanh toán contactless lần đầu?",
    # --- Nhân quả / Điều kiện ---
    "Tại sao giao dịch thẻ tín dụng của tôi bị từ chối dù hạn mức vẫn còn và thẻ chưa hết hạn?",
    "Nếu tôi trả dư nợ thẻ tín dụng đúng ngày nhưng sau ngày due date thì lãi được tính như thế nào?",
    "Trường hợp thẻ bị mất cắp và có giao dịch gian lận, trách nhiệm bồi thường của ngân hàng và khách hàng được phân bổ ra sao?",
    # --- Tổng hợp ---
    "Tôi chi tiêu chủ yếu cho du lịch, ăn uống và mua sắm online, nên chọn thẻ tín dụng nào của Shinhan để tối ưu lợi ích?",
    "Tổng hợp tất cả ưu đãi trả góp 0% lãi suất đang áp dụng theo danh mục: điện máy, du lịch, thời trang, nhà hàng?",

    ## Deposit
    # --- So sánh ---
    "So sánh hiệu quả sinh lời thực tế giữa gửi tiết kiệm kỳ hạn 6 tháng tái tục liên tiếp và gửi thẳng kỳ hạn 12 tháng cùng số tiền 500 triệu?",
    "Phân tích sự khác biệt giữa gửi tiết kiệm có kỳ hạn thông thường và chứng chỉ tiền gửi (NCD) về tính thanh khoản và lãi suất?",
    # --- Quy trình ---
    "Quy trình phong tỏa một phần số dư tài khoản thanh toán theo lệnh của cơ quan thi hành án gồm những bước nào?",
    "Các bước xử lý khi khách hàng báo mất sổ tiết kiệm vật lý và yêu cầu cấp lại là gì?",
    # --- Nhân quả / Điều kiện ---
    "Nếu ngân hàng áp dụng lãi suất khuyến mãi kỳ hạn 3 tháng nhưng tôi tất toán sau 45 ngày, lãi thực nhận được tính như thế nào?",
    "Tại sao lãi suất huy động tiền gửi lại tăng khi NHNN tăng lãi suất điều hành và điều này ảnh hưởng đến NIM của ngân hàng ra sao?",
    "Trường hợp khách hàng doanh nghiệp cần giải tỏa tài khoản ký quỹ để thanh toán cho đối tác, cần điều kiện và thủ tục gì?",
    # --- Tổng hợp ---
    "Với 2 tỷ đồng nhàn rỗi, nên chia gửi theo kỳ hạn nào và kênh nào (quầy hay online) để tối ưu lãi suất và vẫn giữ tính thanh khoản?",

    ## Capital market
    # --- So sánh ---
    "So sánh rủi ro và lợi suất kỳ vọng giữa đầu tư trái phiếu chính phủ kỳ hạn 10 năm và trái phiếu doanh nghiệp BBB kỳ hạn 5 năm?",
    "Phân tích sự khác biệt giữa chiến lược phòng ngừa rủi ro (hedging) bằng hợp đồng kỳ hạn (forward) và hợp đồng hoán đổi (swap) lãi suất?",
    # --- Quy trình ---
    "Quy trình phát hành trái phiếu doanh nghiệp riêng lẻ (private placement) qua Shinhan gồm những bước nào từ tư vấn đến niêm yết?",
    "Các bước kiểm tra sức chịu đựng vốn (stress testing) theo chuẩn Basel III trên danh mục trái phiếu tự doanh là gì?",
    # --- Nhân quả / Điều kiện ---
    "Tại sao khi FED tăng lãi suất, giá trái phiếu chính phủ Việt Nam lại giảm và điều này ảnh hưởng thế nào đến P&L danh mục tự doanh?",
    "Nếu một trái phiếu doanh nghiệp bị hạ xếp hạng tín nhiệm từ BB xuống B-, phòng quản lý rủi ro cần thực hiện hành động gì theo quy chế nội bộ?",
    # --- Tổng hợp ---
    "Đánh giá toàn diện danh mục trái phiếu tự doanh khi đường cong lợi suất đảo ngược (inverted yield curve): rủi ro gì phát sinh và biện pháp phòng ngừa là gì?",

    ## Finance accounting
    # --- So sánh ---
    "So sánh tác động lên báo cáo lợi nhuận khi áp dụng phương pháp trích lập dự phòng theo IAS 39 (Incurred Loss) và IFRS 9 (Expected Credit Loss)?",
    "Phân tích sự khác nhau giữa ROA và ROE trong đánh giá hiệu quả sinh lời của một ngân hàng thương mại?",
    "ROA và ROE khác nhau như thế nào?",
    # --- Quy trình ---
    "Quy trình đối soát (reconciliation) cuối ngày giữa sổ cái tổng hợp (General Ledger) và hệ thống Core Banking diễn ra như thế nào?",
    "Các bước hạch toán bút toán khi một khoản nợ xấu được xóa sổ (write-off) và sau đó khách hàng hoàn trả một phần là gì?",
    # --- Nhân quả / Điều kiện ---
    "Tại sao việc tăng tỷ lệ CASA lại cải thiện NIM (Net Interest Margin) và ROE của ngân hàng?",
    "Nếu tỷ lệ NPL (nợ xấu) tăng vọt lên 5%, áp lực trích lập dự phòng ảnh hưởng đến chỉ tiêu CIR và lợi nhuận trước thuế như thế nào?",
    "Trường hợp ngân hàng có tỷ lệ LCR dưới 100%, NHNN yêu cầu xử lý theo lộ trình nào?",
    # --- Tổng hợp ---
    "Phân tích toàn diện các chỉ số tài chính (ROA, ROE, NIM, CIR, CAR) để đánh giá sức khỏe tài chính của một ngân hàng thương mại cổ phần?",

    ## HR
    # --- So sánh ---
    "So sánh mô hình đánh giá hiệu suất 90 độ (thượng cấp đánh giá) và 360 độ (đánh giá chéo toàn diện) về ưu nhược điểm trong môi trường ngân hàng?",
    "Phân tích sự khác biệt về chính sách phúc lợi giữa nhân viên hợp đồng xác định thời hạn và hợp đồng không xác định thời hạn?",
    "OKR khác KPI ở điểm nào trong việc đánh giá hiệu suất nhân viên?",
    "KPI và OKR khác nhau thế nào?",
    # --- Quy trình ---
    "Quy trình xử lý kỷ luật nhân viên vi phạm Nội quy lao động lần đầu gồm những bước nào từ phát hiện đến quyết định?",
    "Các bước hoàn thiện hồ sơ offboarding khi nhân viên nghỉ việc tự nguyện và thời hạn bàn giao tối đa là bao nhiêu ngày?",
    "Quy trình thăng tiến (promotion) từ chuyên viên lên chuyên viên cao cấp gồm những tiêu chí và vòng đánh giá nào?",
    # --- Nhân quả / Điều kiện ---
    "Nếu nhân viên vi phạm chính sách bảo mật dữ liệu khách hàng mức độ nghiêm trọng, hình thức kỷ luật và quy trình bồi thường thiệt hại là gì?",
    "Trường hợp headcount phòng ban vượt ngưỡng budget được duyệt, cấp nào có thẩm quyền phê duyệt bổ sung và quy trình xử lý ra sao?",
    # --- Tổng hợp ---
    "Xây dựng kế hoạch succession planning cho vị trí Trưởng phòng Tín dụng cần chuẩn bị những gì về năng lực, thời gian và ngân sách đào tạo?",

    ## Internal control
    # --- So sánh ---
    "So sánh quy trình CDD (Customer Due Diligence) thông thường và EDD (Enhanced Due Diligence) về tài liệu yêu cầu và thời gian xử lý?",
    "Phân tích sự khác biệt giữa rủi ro tín dụng (Credit Risk) và rủi ro tác nghiệp (Operational Risk) theo chuẩn Basel III và cách đo lường vốn yêu cầu?",
    "KRI (Key Risk Indicator) khác KPI ở điểm nào trong quản trị rủi ro?",
    # --- Quy trình ---
    "Quy trình xử lý khi hệ thống AML gắn cờ tự động một giao dịch chuyển tiền quốc tế MT103 là gì?",
    "Các bước điều tra nội bộ khi phát hiện nghi ngờ gian lận nội bộ (internal fraud) của nhân viên gồm những vòng nào?",
    "Hướng dẫn xử lý tình huống khách hàng bị xác định là PEP sau khi đã mở tài khoản và có quan hệ tín dụng?",
    # --- Nhân quả / Điều kiện ---
    "Tại sao một lệnh chuyển tiền có thể bị phong tỏa tự động ngay cả khi đã qua xác thực OTP của khách hàng?",
    "Nếu chi nhánh bị kiểm toán phát hiện vi phạm kiểm soát nội bộ mức độ 2, các hành động khắc phục và hình thức xử lý là gì?",
    "Trường hợp khách hàng cung cấp giấy tờ tùy thân giả mạo trong quá trình eKYC, quy trình ứng phó và báo cáo sự cố như thế nào?",
    # --- Tổng hợp ---
    "Xây dựng framework đánh giá rủi ro AML toàn diện cho phân khúc khách hàng doanh nghiệp xuất nhập khẩu tại Shinhan?",

    ## BANCASSURANCE
    # --- So sánh ---
    "So sánh sản phẩm bảo hiểm tử kỳ (Term Life) và bảo hiểm liên kết đầu tư (Unit-linked) về chi phí, rủi ro và phù hợp với từng nhóm khách hàng?",
    "Phân tích sự khác biệt về quy trình thẩm định (underwriting) giữa bảo hiểm sức khỏe nhóm (Group Health) và bảo hiểm cá nhân (Individual)?",
    # --- Quy trình ---
    "Quy trình xử lý bồi thường tử vong (death claim) qua kênh bancassurance từ tiếp nhận hồ sơ đến thanh toán gồm bao nhiêu ngày và những bước nào?",
    "Các bước tư vấn và bán chéo bảo hiểm nhân thọ đúng chuẩn quy trình Needs-based Selling tại quầy giao dịch là gì?",
    # --- Nhân quả / Điều kiện ---
    "Tại sao tỷ lệ hủy hợp đồng (lapse rate) cao lại ảnh hưởng nghiêm trọng đến doanh thu phí bảo hiểm tái tục và KPI của kênh Banca?",
    "Nếu khách hàng không nộp phí trong thời gian ân hạn (grace period), hợp đồng bị xử lý như thế nào và có thể tái hiệu lực (reinstatement) không?",
    "Trường hợp khách hàng khai báo sức khỏe không trung thực khi ký hợp đồng, yêu cầu bồi thường sẽ bị xử lý như thế nào?",
    # --- Tổng hợp ---
    "Tư vấn gói bảo hiểm tổng thể phù hợp cho khách hàng 35 tuổi, có gia đình, đang vay mua nhà 2 tỷ và thu nhập 40 triệu/tháng?",

    ## ORGANIZATION
    # --- So sánh ---
    "So sánh mô hình phân cấp quản lý tập trung (centralized) và phân quyền (decentralized) trong việc phê duyệt tín dụng tại chi nhánh?",
    "Phân tích ưu nhược điểm của mô hình kinh doanh Retail Banking so với Wholesale Banking trong bối cảnh lãi suất thị trường tăng cao?",
    # --- Quy trình ---
    "Quy trình thành lập phòng giao dịch mới tại một tỉnh/thành phố gồm những thủ tục nội bộ và pháp lý nào?",
    "Các bước triển khai chương trình thay đổi tổ chức (organizational change) theo khung Change Management của Kotter gồm những giai đoạn nào?",
    # --- Nhân quả / Điều kiện ---
    "Tại sao việc phân quyền phán quyết tín dụng xuống cấp chi nhánh lại vừa tăng tốc độ phê duyệt nhưng cũng tăng rủi ro và cách kiểm soát?",
    "Nếu một chi nhánh bị xếp hạng D (hạng thấp nhất) trong đánh giá nội bộ, biện pháp can thiệp và lộ trình khắc phục là gì?",
    # --- Tổng hợp ---
    "Thiết kế cơ cấu tổ chức tối ưu cho một chi nhánh mới tại khu công nghiệp với khách hàng chủ yếu là doanh nghiệp FDI và công nhân?",

    ## REMITTANCE
    # --- So sánh ---
    "So sánh chi phí, tốc độ và rủi ro giữa chuyển tiền quốc tế qua SWIFT MT103, SWIFT GPI và ví điện tử (mobile wallet) cho doanh nghiệp xuất nhập khẩu?",
    "Phân tích điểm khác biệt về quy trình compliance khi xử lý lệnh chuyển tiền đến (inward remittance) từ quốc gia trong danh sách FATF Grey List?",
    # --- Quy trình ---
    "Quy trình tra soát và hoàn tiền khi khách hàng chuyển tiền quốc tế bị giữ tại ngân hàng trung gian gồm những bước nào và mất bao lâu?",
    "Các bước xử lý khi lệnh chuyển tiền MT103 bị kẹt trong hàng đợi SWIFT do trùng khớp tên với danh sách trừng phạt (sanction list) là gì?",
    # --- Nhân quả / Điều kiện ---
    "Tại sao tỷ giá thực hiện (execution rate) khi chuyển tiền lại khác với tỷ giá niêm yết và sự chênh lệch này ảnh hưởng như thế nào đến doanh nghiệp xuất khẩu?",
    "Nếu khách hàng cung cấp sai IBAN nhưng đúng tên ngân hàng hưởng, tiền sẽ đi đâu và quy trình xử lý sự cố là gì?",
    "Trường hợp lệnh chuyển tiền bị phong tỏa bởi OFAC sanctions của Mỹ, ngân hàng phải thực hiện những nghĩa vụ pháp lý nào?",
    # --- Tổng hợp ---
    "Tư vấn giải pháp tối ưu hóa chi phí chuyển tiền quốc tế cho doanh nghiệp SME xuất khẩu thường xuyên sang thị trường Hàn Quốc và Nhật Bản?",

    ## TRADE_FINANCE
    # --- So sánh ---
    "So sánh rủi ro thanh toán đối với nhà xuất khẩu trong 4 phương thức: L/C at sight, D/P, D/A và Open Account, và khi nào nên dùng phương thức nào?",
    "Phân tích sự khác biệt về chi phí tài trợ và kiểm soát rủi ro giữa Tài trợ chuỗi cung ứng (Supply Chain Finance) và Factoring có truy đòi?",
    # --- Quy trình ---
    "Quy trình xử lý bộ chứng từ bất hợp lệ (discrepant documents) theo L/C nhập khẩu: từ phát hiện, thông báo đến quyết định chấp nhận/từ chối gồm bao nhiêu bước?",
    "Các bước phát hành thư tín dụng L/C xuất khẩu giáp lưng (back-to-back L/C) dựa trên L/C gốc của người mua nước ngoài là gì?",
    "Quy trình kiểm tra tính hợp lệ của vận đơn (Bill of Lading) và phát hành giấy bảo lãnh (LOI) khi hàng đến trước bộ chứng từ là gì?",
    # --- Nhân quả / Điều kiện ---
    "Tại sao ngân hàng phát hành L/C phải thanh toán ngay cả khi người nhập khẩu phá sản trước khi hàng đến?",
    "Nếu bộ chứng từ xuất trình có 2 sai sót nhỏ nhưng người mua từ chối waiver (chấp nhận bất hợp lệ), ngân hàng và nhà xuất khẩu có những lựa chọn nào?",
    "Trường hợp hàng hóa bị hải quan nước nhập khẩu giữ lại, nghĩa vụ thanh toán L/C của ngân hàng phát hành thay đổi như thế nào?",
    # --- Tổng hợp ---
    "Tư vấn giải pháp tài trợ thương mại tổng thể cho doanh nghiệp nhập khẩu nguyên liệu từ Trung Quốc và xuất khẩu thành phẩm sang EU, cần tối ưu dòng tiền và giảm rủi ro?",
    "Phân tích toàn bộ rủi ro pháp lý và tài chính khi Shinhan đứng ra xác nhận (confirm) một L/C phát hành bởi ngân hàng tại một quốc gia có xếp hạng tín nhiệm thấp?",

    # --- So sánh đa chiều (Comparison) ---
    "Sự khác biệt căn bản về quản trị rủi ro giữa tài trợ chuỗi cung ứng (Supply chain finance) và bao thanh toán (Factoring) là gì?",
    "Phân biệt giữa rủi ro hoạt động (Operational Risk) và rủi ro tín dụng (Credit risk) theo chuẩn Basel II?",
    "So sánh tác động của đòn bẩy hoạt động (Operating leverage) lên biên lãi ròng (NIM) trong báo cáo lợi nhuận Q3?",
    # --- So sánh (Comparison – 2 domains) ---
    "So sánh lãi suất vay mua nhà và vay mua xe của Shinhan hiện tại?",
    "Thẻ tín dụng Visa và Mastercard của Shinhan khác nhau ở điểm nào?",
    "Gói tiết kiệm 6 tháng và 12 tháng cái nào lãi cao hơn?",
    "Chuyển khoản NAPAS và chuyển tiền quốc tế SWIFT phí nào rẻ hơn?",

    # --- Quy trình / Các bước vận hành đa tầng (Procedure/How-to) ---
    "Hướng dẫn các bước cấu hình lại node dữ liệu (Database replication) khi Datacenter A gặp thảm họa mất điện?",
    "Cách hạch toán bút toán trích lập dự phòng tổn thất tín dụng (provisioning) vào sổ cái và hệ thống Core Banking qua từng bước?",
    "Quy trình đánh giá hiệu suất (Performance Appraisal) cuối năm đánh giá chéo 360 độ cho khối Back-office gồm những vòng nào?",
    # --- Quy trình / Thủ tục nhiều bước (Procedure) ---
    "Các bước mở tài khoản SOL và kích hoạt Google Pay như thế nào?",
    "Quy trình vay tiêu dùng trực tuyến Shinhan gồm những bước gì?",
    "Cách tra soát chuyển tiền thất bại và yêu cầu hoàn tiền ra sao?",
    "Hướng dẫn đăng ký Zalo OA nhận thông báo ưu đãi Shinhan?",

    # --- Nhân quả / Tác động (Causal / Impact Analysis) ---
    "Tại sao việc áp dụng chuẩn mực IFRS 9 lại làm tăng đột biến chi phí trích lập dự phòng của ngân hàng?",
    "Nguyên nhân vì sao lệnh chuyển tiền quốc tế MT103 bị phong tỏa tự động trên hệ thống sàng lọc AML (Chống rửa tiền)?",
    "Tại sao thẻ tín dụng doanh nghiệp của phòng kinh doanh bị khóa dù đã tất toán toàn bộ dư nợ đúng hạn định?",
    "Nếu vay 2 tỷ mua nhà kỳ hạn 15 năm thì mỗi tháng trả góp khoảng bao nhiêu?",
    "Tại sao thẻ tín dụng của tôi bị khóa dù chưa quá hạn thanh toán?",
    "Nếu tôi có nợ xấu CIC nhóm 3 thì có đệ trình xin vay vốn tín chấp được không?",
    "Trường hợp khách hàng là người không cư trú (foreigners) thì chứng từ bắt buộc phải là gì?",
    "Tại sao lệnh chuyển tiền quốc tế MT103 lại bị phong tỏa trên hệ thống AML (Chống rửa tiền)?",

    # --- Điều kiện rẽ nhánh (Conditional Reasoning) ---
    "Trường hợp khách hàng doanh nghiệp bị hệ thống đánh dấu là PEP (Nhân vật chính trị ảnh hưởng) thì luồng thẩm định tín dụng bị rẽ nhánh thay đổi ra sao?",
    "Nếu nhân sự vi phạm nội quy bảo mật dữ liệu khách hàng cấp độ 1 thì quy chế đưa ra hình thức bồi thường và kỷ luật thi hành là gì?",
    "Nếu vay thế chấp mua dự án 2 tỷ đồng với kỳ hạn rải vụ 15 năm lãi suất thả nổi thì sang năm thứ hai mỗi tháng thanh toán dư nợ ước tính khoảng bao nhiêu?",

    # --- Khuyến nghị / Tổng hợp Phức tạp (Synthesis / Aggregation) ---
    "Vui lòng liệt kê hệ thống hóa toàn bộ các điều kiện và ngoại lệ áp dụng cho khoản vay equipment term loan đối với người không cư trú (non-resident)?",
    "Đối với doanh nghiệp Start-up có vòng quay vốn lưu động ngắn, để tối ưu hóa dòng tiền thì chuyên viên quan hệ khách hàng nên tư vấn mở thẻ tín dụng nội địa hay phát hành bảo lãnh thanh toán?",
    "Với lương 20 triệu thì nên chọn gói vay mua nhà hay vay tiêu dùng?",
    "Tôi hay mua sắm online, nên dùng thẻ tín dụng nào của Shinhan?",

    # --- Tổng hợp / Aggregation ---
    "Tổng hợp tất cả ưu đãi trả góp 0% đang áp dụng cho thẻ tín dụng Shinhan?",
    "Liệt kê tất cả điều kiện được miễn phí chuyển khoản nhanh 24/7 qua NAPAS?",

    # --- Vận hành & Hệ thống Nội bộ (Internal Operations / Architecture) ---
    "So sánh sự khác biệt giữa tài trợ chuỗi cung ứng (Supply chain finance) và quy trình chiết khấu bộ chứng từ?",
    "Cách hạch toán bút toán trích lập dự phòng rủi ro tín dụng (provisioning) vào sổ cái và hệ thống Core Banking như thế nào?",
    "Hướng dẫn cách khắc phục khi hệ thống gặp lỗi downtime máy chủ (Disaster Recovery)?",
    "So sánh tác động của đòn bẩy hoạt động (Operating leverage) lên biên lãi ròng (NIM) trong báo cáo Q3?",
    "Các bước tiến hành kiểm tra sức chịu đựng vốn (Stress testing) theo khung chuẩn Basel là làm sao?",

    # --- Multi-domain: Loan + Risk + Compliance ---
    "Nếu khách hàng có CIC nhóm 2 nhưng thu nhập ổn định thì có thể override phê duyệt tín dụng không và cần thêm điều kiện gì?",
    "Trong trường hợp khách hàng là PEP thì quy trình phê duyệt khoản vay sẽ khác gì so với khách hàng thông thường?",
    "Nếu tỷ lệ DTI vượt ngưỡng nhưng có tài sản đảm bảo tốt thì hệ thống scoring xử lý như thế nào?",

    # --- Loan + Accounting ---
    "Việc tái cơ cấu khoản vay ảnh hưởng như thế nào đến provisioning theo IFRS 9?",
    "Khi khoản vay chuyển sang nợ nhóm 3 thì cách hạch toán dự phòng thay đổi ra sao?",

    # --- Card + Fraud + IT ---
    "Nếu hệ thống fraud detection phát hiện giao dịch bất thường thì quy trình khóa thẻ và notify khách hàng diễn ra như thế nào?",
    "Trong trường hợp chargeback xảy ra thì luồng xử lý giữa ngân hàng phát hành và ngân hàng thanh toán diễn ra ra sao?",

    # --- Remittance + Compliance ---
    "Tại sao giao dịch chuyển tiền quốc tế qua SWIFT lại bị giữ lại do sanction screening và cách xử lý tiếp theo là gì?",
    "Nếu giao dịch bị nghi ngờ AML thì quy trình báo cáo STR được thực hiện như thế nào?",

    # --- Trade Finance + Risk ---
    "Sự khác biệt về rủi ro giữa L/C trả ngay và L/C trả chậm trong tài trợ thương mại là gì?",
    "Nếu bộ chứng từ xuất khẩu bị sai lệch thì ngân hàng xử lý như thế nào trong quy trình thanh toán L/C?",

    # --- Deposit + Liquidity + Finance ---
    "Việc tăng lãi suất huy động ảnh hưởng như thế nào đến LCR và chi phí vốn của ngân hàng?",
    "Nếu khách hàng rút tiền hàng loạt thì ngân hàng sẽ kích hoạt cơ chế quản lý thanh khoản như thế nào?",

    # --- IT + Banking Ops ---
    "Khi hệ thống core banking bị downtime thì quy trình chuyển sang disaster recovery site diễn ra như thế nào?",
    "Cách đảm bảo data consistency giữa các node trong database replication khi xảy ra failover?",

    # --- HR + Compliance ---
    "Nếu nhân viên vi phạm quy định bảo mật dữ liệu khách hàng thì quy trình xử lý kỷ luật và báo cáo nội bộ diễn ra ra sao?",
    "Trong đánh giá hiệu suất 360 độ, làm thế nào để đảm bảo tính minh bạch và tránh xung đột lợi ích?",

    # --- Customer + Product Recommendation ---
    "Với khách hàng có thu nhập 30 triệu và chi tiêu nhiều qua online thì nên tư vấn thẻ tín dụng nào để tối ưu cashback và hạn mức?",
    "Nếu khách hàng muốn tối ưu dòng tiền ngắn hạn thì nên chọn vay thấu chi hay thẻ tín dụng?",

    # --- Multi-step Procedure ---
    "Quy trình từ lúc khách hàng đăng ký vay online đến khi giải ngân trên hệ thống LOS gồm những bước nào và các hệ thống nào tham gia?",
    "Các bước từ khi khách hàng thực hiện chuyển tiền quốc tế đến khi người nhận nhận được tiền gồm những khâu trung gian nào?",

    # --- Conditional + Calculation ---
    "Nếu vay 3 tỷ trong 20 năm với lãi suất thả nổi thì khi lãi suất tăng 2% thì số tiền trả hàng tháng thay đổi như thế nào?",
    "Nếu khách hàng trả chậm thẻ tín dụng 10 ngày thì phí phạt và lãi phát sinh được tính ra sao?",

    # --- Strategy / Decision ---
    "Trong bối cảnh lãi suất thị trường tăng, ngân hàng nên điều chỉnh chiến lược huy động và cho vay như thế nào để tối ưu NIM?",
    "Đối với doanh nghiệp xuất nhập khẩu, nên sử dụng L/C hay open account trade để tối ưu chi phí và rủi ro?",

    # ===== COMPLEX - Multi-dimensional comparison =====
    "Compare the benefits of fixed-rate vs. floating-rate home loans at Shinhan.",
    "What are the key differences between factoring and supply chain finance?",
    "How does the risk provisioning for NPL group 2 differ from group 3 under IFRS 9?",
    "Compare the customer acquisition cost between Zalo OA marketing and email campaigns.",
    "Explain the difference in tax treatment between government bonds and corporate bonds.",

    # ===== COMPLEX - Multi-step procedures =====
    "Walk me through the steps to dispute a fraudulent credit card transaction.",
    "Describe the complete offboarding process for a full-time employee from resignation to exit clearance.",
    "How do I trace an international SWIFT payment that has been delayed due to incorrect beneficiary name?",
    "What is the step-by-step process to issue an import L/C for a first-time trading company?",
    "Explain the procedure to reset mobile banking OTP token when changing to a new phone.",
    "How to handle a customer complaint about double deduction from their account?",

    # ===== COMPLEX - Causal / Impact analysis =====
    "Why does a floating interest rate loan still change when the base rate remains unchanged?",
    "What causes a credit card to be suddenly blocked even though the limit is not exceeded?",
    "How does prolonged group 2 bad debt affect future loan eligibility?",
    "Why would the bank request additional collateral when property value drops?",
    "Explain the impact of inflation on both deposit and lending rates.",
    "Why was my outward remittance frozen by the AML system despite clean paperwork?",

    # ===== COMPLEX - Conditional / Branching logic =====
    "If I borrow 5 billion VND for 15 years with a 2-year fixed rate of 7% then floating, what will be the monthly payment in year 3?",
    "In case a customer has a CIC bad debt but fully repaid over 2 years ago, can they still qualify for a new loan? What conditions apply?",
    "If an employee accidentally violates data protection policy, what is the disciplinary and remedial process?",
    "When the core banking system fails during peak hours, how is the disaster recovery plan activated?",

    # ===== COMPLEX - Synthesis / Aggregation =====
    "List all scenarios where the annual credit card fee can be waived.",
    "Summarize all preferential loan packages currently available for SME customers.",
    "List all types of fees applicable when withdrawing cash from a credit card.",
    "Enumerate all authentication methods supported by Shinhan online banking.",
    "Provide a complete list of conditions to get free 24/7 NAPAS transfers.",

    # ===== COMPLEX - Comparison =====
    "신한은행의 고정금리와 변동금리 주택담보대출의 장단점을 비교해 주세요.",  # Compare the pros and cons of fixed-rate vs. floating-rate home loans at Shinhan Bank.
    "팩토링과 공급망 금융의 주요 차이점은 무엇인가요?",  # What are the key differences between factoring and supply chain finance?
    "IFRS 9에 따라 부실채권 2그룹과 3그룹의 대손충당금 설정이 어떻게 다른가요?",  # Under IFRS 9, how does provisioning for bad debt differ between Group 2 and Group 3?
    "Zalo OA 마케팅과 이메일 마케팅의 고객 획득 비용을 비교하세요.",  # Compare customer acquisition cost between Zalo OA marketing and email marketing.
    "국채와 회사채의 세금 처리 차이를 설명해 주세요.",  # Explain the difference in tax treatment between government bonds and corporate bonds.

    # ===== COMPLEX - Multi-step procedures =====
    "신용카드 사기 거래를 이의제기하는 전체 절차를 단계별로 알려주세요.",  # Walk me through the step-by-step process to dispute a fraudulent credit card transaction.
    "정규 직원의 퇴직 절차를 사직서 제출부터 퇴직 정산까지 설명해 주세요.",  # Describe the complete offboarding process for a regular employee from resignation to final settlement.
    "수취인 이름 오류로 지연된 국제 SWIFT 송금을 추적하는 방법은 무엇인가요?",  # How to trace an international SWIFT transfer delayed due to an incorrect beneficiary name?
    "신규 무역 회사가 수입 신용장(L/C)을 발행하는 단계별 절차는 어떻게 되나요?",  # What is the step-by-step procedure for a new trading company to issue an import letter of credit (L/C)?
    "휴대폰 변경 시 모바일뱅킹 OTP 토큰을 재설정하는 방법을 알려주세요.",  # How to reset the mobile banking OTP token when changing to a new phone?
    "계좌에서 이중 인출된 고객 불만을 처리하는 방법은 무엇인가요?",  # How to handle a customer complaint about a double debit from their account?

    # ===== COMPLEX - Causal / Impact analysis =====
    "기준 금리가 변하지 않았는데 변동금리 대출의 금리가 변하는 이유는 무엇인가요?",  # Why does a floating interest rate loan change even when the base rate remains unchanged?
    "한도가 남아 있는데 신용카드가 갑자기 정지되는 원인은 무엇인가요?",  # What causes a credit card to be suddenly blocked even though there is still available credit limit?
    "장기간 연체 2그룹 상태가 미래 대출 자격에 어떤 영향을 미치나요?",  # How does prolonged Group 2 delinquency affect future loan eligibility?
    "담보 자산 가치 하락 시 은행이 추가 담보를 요구하는 이유는 무엇인가요?",  # Why would the bank request additional collateral when the value of the pledged asset declines?
    "인플레이션이 은행의 예금 금리와 대출 금리에 미치는 영향을 설명해 주세요.",  # Explain the impact of inflation on the bank's deposit rates and lending rates.
    "서류가 완벽한데도 AML 시스템에 의해 해외 송금이 동결된 이유는 무엇인가요?",  # Why was my outward remittance frozen by the AML system even though the paperwork is complete?

    # ===== COMPLEX - Conditional / Branching logic =====
    "50억 원을 15년 동안 빌리는데 처음 2년은 고정금리 7% 그 후 변동금리라면, 3년차 월 상환액은 얼마인가요?",  # If I borrow 5 billion KRW for 15 years with a 2-year fixed rate of 7% then floating, what will be the monthly payment in year 3?
    "신용정보원에 부실 기록이 있지만 2년 전에 완제한 고객이 새 대출을 받을 수 있나요? 조건은 무엇인가요?",  # Can a customer with a past bad debt record at the Credit Bureau (fully repaid over 2 years ago) still get a new loan? What are the conditions?
    "직원이 실수로 데이터 보호 정책을 위반한 경우, 징계 및 시정 절차는 어떻게 되나요?",  # If an employee accidentally violates the data protection policy, what is the disciplinary and corrective procedure?
    "코어뱅킹 시스템이 피크 시간에 장애가 발생하면 재해 복구 계획은 어떻게 작동하나요?",  # When the core banking system fails during peak hours, how does the disaster recovery plan get activated?

    # ===== COMPLEX - Synthesis / Aggregation =====
    "신용카드 연회비 면제 조건을 모두 나열해 주세요.",  # List all conditions under which the annual credit card fee can be waived.
    "현재 중소기업(SME) 고객을 위한 우대 대출 상품을 모두 요약해 주세요.",  # Summarize all preferential loan products currently available for SME customers.
    "신용카드 현금 서비스 이용 시 부과되는 모든 수수료 유형을 나열해 주세요.",  # List all types of fees charged when using a credit card cash advance.
    "신한 온라인 뱅킹에서 지원하는 모든 인증 방법을 열거해 주세요.",  # Enumerate all authentication methods supported by Shinhan online banking.
    "NAPAS 24시간 무료 이체 혜택을 받기 위한 모든 조건을 나열해 주세요.",  # List all conditions to qualify for free 24/7 NAPAS transfers.,
    "Auto renewal của tiền gửi hoạt động thế nào?",
    "EBITDA khác với lợi nhuận ròng (Net Income) như thế nào?",
    "ESOP (Employee Stock Ownership Plan) tại ngân hàng hoạt động như thế nào?",
    "Quy trình onboarding gồm những bước gì cơ bản?",
    "AML screening hoạt động ra sao?",
    "Unit-linked insurance (bảo hiểm liên kết đầu tư) khác bảo hiểm tử kỳ thuần túy như thế nào?",
    "Mô hình làm việc ma trận (Matrix Structure) trong tổ chức ngân hàng có ưu và nhược điểm gì?",
    "RTGS (Real-Time Gross Settlement) khác với chuyển tiền NAPAS thông thường ở điểm nào?",
    "Forfaiting khác với Factoring xuất khẩu ở điểm nào về truy đòi (recourse)?",
    "Incoterms CIF và FOB khác nhau về nghĩa vụ người bán/mua như thế nào?",
    "D/P (Documents against Payment) và D/A (Documents against Acceptance) có sự khác biệt rủi ro gì?",
    "Bảo lãnh thực hiện hợp đồng (Performance Guarantee) và bảo lãnh dự thầu (Bid Bond) khác nhau ở điểm nào?",
    "Thủ tục thanh toán expense claim nội bộ gồm những gì?",
    "What is the difference between debit card and credit card?",  # simple comparison,
    "신용카드와 체크카드의 차이는 무엇인가요?",  # What is the difference between a credit card and a debit card?,

    # =================================================================
    # [V5] Chat-style / Không dấu câu (30% mix)
    # =================================================================
    "so sánh vay mua nhà với vay tiêu dùng",
    "vay 2 tỷ lương 40 triệu trả hàng tháng bao nhiêu",
    "nên chọn lãi suất cố định hay thả nổi",
    "cách đăng ký trả góp 0% qua thẻ tín dụng",
    "nếu trả chậm thẻ 10 ngày thì phí phạt bao nhiêu",
    "hướng dẫn mở tài khoản sol và kích hoạt google pay",
    "tại sao thẻ tín dụng bị khóa dù chưa quá hạn",
    "điều kiện khách hàng nợ xấu nhóm 2 vay vốn",
    # --- R&R Complex Exception cases ---
    "vai trò và trách nhiệm của DBU",
    "R&R của CBD",
    "roles and responsibilities of Strategy division",
    "DBU R&R",
    "Vai trò và trách nhiệm của phòng DBU là gì?",
    "Cho mình xem vai trò và trách nhiệm của CBD.",
    "What is the role and responsibility of the Strategy division?",
    "Please show me the R&R of the DBU department.",
    "So sánh vai trò và trách nhiệm giữa phòng DBU và Strategy division.",
    "Quy trình phân chia vai trò và trách nhiệm (R&R) của phòng CBD được thực hiện thế nào?",
    "Compare the roles and responsibilities of DBU and CBD departments.",
    "How to define and update the roles and responsibilities for each department in Shinhan Bank?",

    # --- [v3] Additional Procedural, Guideline, Regulation, Comparison anchors ---
    "So sánh lãi suất VCB và Techcombank",
    "Thẻ Visa và Mastercard khác nhau thế nào?",
    "Tại sao lãi suất thay đổi từ 2020-2024?",
    "Các bước mở tài khoản doanh nghiệp như thế nào?",
    "Hướng dẫn cách đăng ký Internet Banking",
    "Hướng dẫn màn hình 0001",
    "Quy định về việc sử dụng màn hình 7925",
    "quy trình thanh toán",
    "hướng dẫn về việc ban hành sản phẩm mới",
    "Tổng hợp tất cả chính sách ưu đãi cho sinh viên",
    "QUY ĐỊNH VỀ THẾ CHẤP TÀI SẢN ĐẢM BẢO LÀ MÁY MÓC THIẾT BỊ",
    "các thông tin liên quan đến high risk loại A & B",
    "high risk loại A và B",
    "tài sản chồng đứng tên nhưng mua trong hôn nhân thì có được thế chấp không",
    "shinhanbank có cho vay đối tượng doanh nghiệp là nhà thầu không",
    "Doanh nghiệp 100% vốn nước ngoài có được mở tài khoản online không",
    "Người nước ngoài có được phát hành thẻ tín dụng tại Shinhan không",
    "với số tiền muốn vay 2,2 tỷ và thu nhập 50 triệu tháng vay mua nhà bao lâu",
]


# =============================================================================
# [V5] NHÓM 4: CLARIFY (Câu mơ hồ, bare noun, thiếu intent rõ ràng)
# =============================================================================
# ĐỊNH NGHĨA KỸ:
# - Là các query chỉ chứa danh từ/cụm danh từ chuyên ngành, KHÔNG có động từ,
#   từ hỏi, hay dấu hiệu hành động nào.
# - User chưa nói rõ muốn hỏi gì: bao nhiêu? điều kiện gì? so sánh cái gì?
# - Bao gồm: Bare keyword, bare acronym, cụm danh từ banking thuần túy.
# - Hệ thống cần hỏi lại để làm rõ intent trước khi tra cứu.
#
# QUAN TRỌNG: Centroid này giúp Semantic Router phân biệt bare noun với
# câu hỏi thực sự (simple). Không có centroid này, bare noun sẽ bị kéo
# về simple centroid gần nhất → sai.
CLARIFY_SENTENCES = [
    # --- Bare entity / keyword đơn lẻ ---
    "swift",
    "dragon pass",
    "ltv",
    "cic",
    "overdraft",
    "sanction screening",
    "otp",
    "napas",
    "ekyc",
    # --- Cụm danh từ banking không có intent ---
    "bảo hiểm khoản vay",
    "thẻ tín dụng",
    "lãi suất vay mua nhà",
    "phí atm",
    "phí swift",
    "hạn mức",
    "khoản vay",
    "tài khoản thanh toán",
    "chuyển tiền quốc tế",
    "vay mua nhà",
    "tiết kiệm",
    # --- Khái niệm chung / mơ hồ ---
    "bảo hiểm",
    "lãi suất",
    "phí dịch vụ",
    "thủ tục vay",
    "cic nhóm 2",
    # --- 150+ Bare Nouns / Từ khóa lấp lửng bổ sung ---
    "thẻ ghi nợ", "vay thế chấp", "vay tín chấp", "tài khoản lương", 
    "tài khoản tiết kiệm", "sổ tiết kiệm", "chứng chỉ tiền gửi", 
    "lãi suất tiền gửi", "lãi suất cho vay", "phí chuyển tiền", 
    "phí rút tiền", "phí duy trì tài khoản", "phí thường niên", 
    "phát hành thẻ", "mở thẻ", "mở tài khoản", "chuyển khoản", 
    "thanh toán quốc tế", "tỷ giá hối đoái", "tỷ giá usd", 
    "tỷ giá ngoại tệ", "hạn mức tín dụng", "hạn mức chuyển tiền", 
    "hạn mức rút tiền", "hạn mức thấu chi", "hạn mức thẻ", 
    "bảo mật 2 lớp", "mật khẩu", "mã otp", "chặn thẻ", "khóa thẻ", 
    "hủy thẻ", "gia hạn thẻ", "thay đổi pin", "quên mật khẩu", 
    "vấn tin số dư", "sao kê tài khoản", "sao kê thẻ", "trả góp", 
    "vay mua nhà", "vay mua xe", "vay du học", "bảo hiểm nhân thọ", 
    "bancassurance", "trái phiếu", "cổ phiếu", "quỹ đầu tư", 
    "chứng khoán", "ngân hàng điện tử", "internet banking", 
    "mobile banking", "app ngân hàng", "giao dịch trực tuyến", 
    "thanh toán hóa đơn", "nạp tiền điện thoại", "quét mã qr", 
    "rút tiền không cần thẻ", "thẻ visa", "thẻ mastercard", 
    "thẻ napas", "máy atm", "phòng giao dịch", "chi nhánh", 
    "tổng đài cskh", "giờ làm việc", "đáo hạn", "tất toán", 
    "nợ xấu", "điểm tín dụng", "thẩm định tài sản", "giải ngân", 
    "khế ước nhận nợ", "chứng minh thu nhập", "hợp đồng lao động", 
    "bảng lương", "sao kê lương", "bảo lãnh", "bảo lãnh dự thầu", 
    "l/c", "thư tín dụng", "thanh toán l/c", "nhờ thu", "bao thanh toán", 
    "tiền mặt", "séc", "ủy nhiệm chi", "báo có", "báo nợ", "số tài khoản", 
    "tên chủ tài khoản", "chủ sở hữu", "đồng sở hữu", "tài sản đảm bảo", 
    "đăng ký thế chấp", "công chứng", "giấy tờ nhà đất", "sổ đỏ", 
    "sổ hồng", "đăng ký kinh doanh", "thuế thu nhập cá nhân", 
    "thuế tndn", "hóa đơn gtgt", "mã số thuế", "đại diện pháp luật", 
    "con dấu", "chữ ký số", "hợp đồng kinh tế", "phương án kinh doanh", 
    "tờ khai hải quan", "vận đơn", "c/o", "c/q", "swift code", "iban", 
    "mã ngân hàng", "chi nhánh mở tài khoản", "phí phát hành", 
    "phí cấp lại", "phí quản lý", "lãi quá hạn", "lãi phạt", "vốn tự có", 
    "nguồn vốn", "room tín dụng", "tín dụng đen", "cảnh báo lừa đảo", 
    "phishing", "malware", "virus", "hacker", "lỗ hổng bảo mật", 
    "an toàn thông tin", "dữ liệu cá nhân", "bảo mật thông tin", 
    "khách hàng vip", "ưu đãi", "khuyến mãi", "hoàn tiền", "cashback", 
    "tích điểm", "đổi quà", "dặm bay", "quyền lợi", "tiện ích", 
    "dịch vụ", "sản phẩm", "gói cước", "combo", "biểu phí", 
    "biểu lãi suất", "hướng dẫn", "thủ tục", "hồ sơ", "giấy tờ", 
    "điều kiện", "đối tượng", "hủy bỏ", "thay đổi", "cập nhật", 
    "nâng cấp", "bảo trì", "lỗi hệ thống", "không đăng nhập được", 
    "quên user", "sai mật khẩu", "khóa tài khoản", "bị nuốt thẻ", 
    "mất thẻ", "thẻ hết hạn", "đổi thẻ", "thẻ từ", "thẻ chip", 
    "contactless", "chạm", "pos", "mpos", "edc", "nfc", "samsung pay", 
    "apple pay", "google pay", "qr code", "vnpay", "momo", "zalopay", 
    "viettel money", "shopeepay", "ví điện tử", "liên kết ví", 
    "nạp rút ví", "tops club", "aither", "màn hình 8800", 
    "resignation process", "overdue loan", "nguyên tắc nhận tài sản bảo đảm",
]
