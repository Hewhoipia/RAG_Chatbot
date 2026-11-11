corpus = {
    "doc_001": {
        "title": "Individual CIF Classification", 
        "text": "Individual customers are classified into CIF Type 2-Person, which includes five main CIF Classes: 100 Individual_Vietnamese, 110 Individual_Foreigner(Korean), 120 Individual_Foreigner(Except Korean), 130 Staff_Vietnamese, and 140 Staff_Korean [1]."
    },
    "doc_002": {
        "title": "Foreigner Residency Status", 
        "text": "Foreigners are permitted to reside in Vietnam with a term of 12 months (360 days) or more [2]. The residency status is determined based on the temporary residence period indicated on the customer's identity documents [2]."
    },
    "doc_003": {
        "title": "Account Opening Forms", 
        "text": "To register a new CIF and open an account/service for individual customers, you must submit the Deposit 01a form [3]. To request the opening of a demand payment account, you must submit the Deposit 04a form [3]."
    },
    "doc_004": {
        "title": "Inactive Account Policy", 
        "text": "A payment account will be automatically classified into an 'Inactive account' status if it has no transactions for 12 months and holds a balance of less than 100,000 VND or equivalent [4]."
    },
    "doc_005": {
        "title": "S-Payroll Account Conditions", 
        "text": "To maintain promotional benefits on an S-Payroll account, a customer must have at least one salary transaction or receive an income transfer of at least 5 million VND per month [5]."
    },
    "doc_006": {
        "title": "K-Study Account Overview", 
        "text": "The K-Study account is a payment account specifically designed for customers who need to prove their financial capacity to study abroad in South Korea [6]. It requires a guaranteed deposit equivalent to 8,000,000 KRW or 10,000,000 KRW [6]."
    },
    "doc_007": {
        "title": "Tops Club Membership Tiers", 
        "text": "The Tops Club program is a loyalty program designed for Shinhan Bank's loyal customers [7]. The membership is divided into four evaluation tiers: PREMIER, ACE, BEST, and CLASSIC [7]."
    },
    "doc_008": {
        "title": "ATM Withdrawal Limits", 
        "text": "For ATM transactions, the cash withdrawal limit is 50,000,000 VND per transaction, with a maximum daily withdrawal limit of 100,000,000 VND [8]."
    },
    "doc_009": {"title": "Visa Signature Lounge Access", "text": "Shinhan Visa Signature cardholders are granted 2 free airport lounge visits per quarter for the primary cardholder and 1 visit per quarter for the supplementary cardholder via the DragonPass application."},
    "doc_010": {"title": "PWM Platinum Annual Fee", "text": "The Shinhan Visa PWM Platinum credit card offers a complete annual fee waiver for both the primary and supplementary cards as an exclusive benefit for PWM customers."},
    "doc_011": {"title": "Travel Platinum Miles", "text": "The Travel Platinum card accumulates a basic 1 Shinhan mile for every 25,000 VND spent and offers triple (3x) reward miles for international transactions made on weekends."},
    "doc_012": {"title": "Korean Air Card Cashback", "text": "Customers using the Korean Air - Shinhan consumer credit card will receive a cashback of 300,000 VND when their total spending in the statement cycle reaches a minimum of 30,000,000 VND."},
    "doc_013": {"title": "Foreign Transaction Fee", "text": "The foreign currency transaction fee applied to the Shinhan Visa Signature credit card is 1.99% on the transaction amount, while the Travel Platinum card applies a 2.9% fee."},
    "doc_014": {"title": "Korea Visa Privilege", "text": "Shinhan Platinum and Signature credit cardholders are exempt from financial proof procedures and enjoy simplified processes when applying for a South Korean tourist visa."},
    "doc_015": {"title": "Shinhan Point Redemption", "text": "Customers can redeem accumulated Shinhan Points into cash at a rate of 1 point = 1 VND, requiring a minimum redemption amount of 15,000 points. Alternatively, 280,000 points can be converted to 1,000 Vietnam Airlines Lotusmiles."},
    "doc_016": {"title": "Credit Card Cash Advance Limit", "text": "For personal credit cards, the cash advance limit granted by the bank is up to 25% of the total credit limit of the card. The cash advance fee is 4% of the withdrawn amount, with a minimum of 50,000 VND."},
    "doc_017": {"title": "Card Replacement Fee", "text": "If a customer requests a credit card replacement due to loss or damage, the bank will charge a replacement fee of 220,000 VND for each issuance."},
    "doc_018": {"title": "SOL Travel Debit Card Benefits", "text": "The Shinhan SOL Travel debit card applies a low foreign currency conversion fee of 0.99%. Cardholders also get special point rewards of 1% or 2% if their previous month's spending reached 3 million VND."}
}

queries = {
    "q_101": "How are individual customers classified in the CIF system based on their nationality and staff status?",
    "q_102": "What is the requirement for a foreigner to be considered a resident in Vietnam?",
    "q_103": "Which specific forms do I need to submit to register a new CIF and open a demand payment account?",
    "q_104": "Under what conditions will my payment account be automatically locked or classified as inactive?",
    "q_105": "What are the minimum monthly deposit or salary transfer requirements to receive S-Payroll account benefits?",
    "q_106": "Does the bank offer any specific account for students planning to study abroad in South Korea?",
    "q_107": "What are the different membership tiers available in the Tops Club loyalty program?",
    "q_108": "What is the maximum amount of cash I can withdraw from an ATM in a single day?",
    "q_109": "How many times can I use the airport lounge for free with a Visa Signature card?",
    "q_110": "Is there an annual fee for the PWM Platinum credit card?",
    "q_111": "How are miles calculated for weekend overseas purchases on the Travel Platinum card?",
    "q_112": "What is the spending requirement to get the 300,000 VND cashback on the Korean Air card?",
    "q_113": "What is the foreign exchange markup fee for the Visa Signature and Travel Platinum cards?",
    "q_114": "Do I need to prove my finances for a Korean visa if I have a Platinum card?",
    "q_115": "What is the conversion rate for redeeming Shinhan Points to Lotusmiles?",
    "q_116": "How much of my credit limit can I withdraw as cash and what is the fee?",
    "q_117": "How much does it cost to get a replacement card if I lose mine?",
    "q_118": "What is the foreign transaction fee for the SOL Travel debit card?"
}

# Mapping query IDs to corpus IDs (1 means highly relevant)
qrels = {
    "q_101": {"doc_001": 1},
    "q_102": {"doc_002": 1},
    "q_103": {"doc_003": 1},
    "q_104": {"doc_004": 1},
    "q_105": {"doc_005": 1},
    "q_106": {"doc_006": 1},
    "q_107": {"doc_007": 1},
    "q_108": {"doc_008": 1},
    "q_109": {"doc_001": 1},
    "q_110": {"doc_002": 1},
    "q_111": {"doc_003": 1},
    "q_112": {"doc_004": 1},
    "q_113": {"doc_005": 1},
    "q_114": {"doc_006": 1},
    "q_115": {"doc_007": 1},
    "q_116": {"doc_008": 1},
    "q_117": {"doc_009": 1},
    "q_118": {"doc_010": 1}
}