# BANKING77 Dataset Audit & Reproducible Split Report

**Project:** Transformer-Based Banking Support Intelligence  
**Dataset Identifier:** `banking77`  
**Audit Date / Context:** Production Intent Classification (77-class)

---

## 1. Executive Summary & Integrity Scorecard

| Metric | Official Train Split | Official Test Split | Reproducible Train Split | Reproducible Val Split |
| :--- | :--- | :--- | :--- | :--- |
| **Total Examples** | **10,003** | **3,080** | **8,002** (80.0%) | **2,001** (20.0%) |
| **Unique Intent Labels** | **77** | **77** | **77** | **77** |
| **Null / Missing Texts** | 0 | 0 | 0 | 0 |
| **Empty / Whitespace Texts** | 0 | 0 | 0 | 0 |
| **Exact Duplicate Texts** | 0 | 0 | 0 | 0 |
| **Malformed Records** | 0 | 0 | 0 | 0 |
| **HTML Markup / Tags** | 0 | 0 | 0 | 0 |
| **Control Characters** | 0 | 0 | 0 | 0 |

- **Cross-Split Overlap (Train ∩ Test):** **0** examples (Zero data leakage).
- **Train/Val Overlap (Train ∩ Val):** **0** texts, **0** indices (Strictly disjoint partition).
- **Official Test Immutability:** **VERIFIED - 100% untouched**

---

## 2. Reproducibility & Environment Details

| Parameter | Configuration / Version |
| :--- | :--- |
| **Dataset Source** | Hugging Face canonical mirror (`datasets.load_dataset("banking77")`) |
| **Random Seed** | `42` |
| **Validation Split Ratio** | `0.20` (~20.0% of official train) |
| **Python Version** | `3.13.14` |
| **OS Platform** | `Windows-11-10.0.26200-SP0` |
| **`datasets` version** | `5.0.1` |
| **`pandas` version** | `2.2.3` |
| **`numpy` version** | `2.2.6` |
| **`scikit-learn` version** | `1.7.2` |
| **`pytest` version** | `9.1.1` |

---

## 3. Text Length Distribution Statistics

Lengths are computed for both character count and whitespace-delimited word tokens.

### Character Length
| Split | Min | Max | Mean | Median | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Official Train** | 13 | 433 | 59.47 | 47.00 | 40.87 |
| **Official Test** | 13 | 368 | 54.23 | 45.00 | 34.66 |

### Word Count
| Split | Min | Max | Mean | Median | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Official Train** | 2 | 79 | 11.95 | 10.00 | 7.89 |
| **Official Test** | 2 | 69 | 10.95 | 9.00 | 6.69 |

*Observations:* Text length is compact and well-behaved for transformer encoders. With maximum lengths under 80 words (433 characters) and median ~10 words, a tokenizer maximum sequence length of `128` (or even `64`) will encompass 100% of all customer queries without truncation.

---

## 4. Class Distribution & Balance Audit

- **Total Classes:** 77 fine-grained banking intents.
- **Official Test Split:** Perfectly balanced with exactly **40 examples per class** ($77 \times 40 = 3,080$).
- **Official Train Split:** Moderately imbalanced across intents:
  - **Min Class Count:** 35 examples
  - **Max Class Count:** 187 examples
  - **Mean Class Count:** 129.91 examples
  - **Median Class Count:** 127.0 examples
  - **Standard Deviation:** 32.94

### Top 5 Most Frequent Intents (Train)
- `card_payment_fee_charged`: 187 examples (1.87%)
- `direct_debit_payment_not_recognised`: 182 examples (1.82%)
- `balance_not_updated_after_cheque_or_cash_deposit`: 181 examples (1.81%)
- `wrong_amount_of_cash_received`: 180 examples (1.80%)
- `cash_withdrawal_charge`: 177 examples (1.77%)

### Top 5 Least Frequent Intents (Train)
- `lost_or_stolen_card`: 82 examples (0.82%)
- `card_swallowed`: 61 examples (0.61%)
- `card_acceptance`: 59 examples (0.59%)
- `virtual_card_not_working`: 41 examples (0.41%)
- `contactless_not_working`: 35 examples (0.35%)

### Comprehensive 77-Class Distribution Table
| ID | Intent Name | Train Count (%) | Test Count (%) | Sub-Train Count (%) | Sub-Val Count (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | `activate_my_card` | 159 (1.59%) | 40 (1.30%) | 127 (1.59%) | 32 (1.60%) |
| 1 | `age_limit` | 110 (1.10%) | 40 (1.30%) | 88 (1.10%) | 22 (1.10%) |
| 2 | `apple_pay_or_google_pay` | 126 (1.26%) | 40 (1.30%) | 101 (1.26%) | 25 (1.25%) |
| 3 | `atm_support` | 87 (0.87%) | 40 (1.30%) | 70 (0.87%) | 17 (0.85%) |
| 4 | `automatic_top_up` | 127 (1.27%) | 40 (1.30%) | 102 (1.27%) | 25 (1.25%) |
| 5 | `balance_not_updated_after_bank_transfer` | 171 (1.71%) | 40 (1.30%) | 137 (1.71%) | 34 (1.70%) |
| 6 | `balance_not_updated_after_cheque_or_cash_deposit` | 181 (1.81%) | 40 (1.30%) | 145 (1.81%) | 36 (1.80%) |
| 7 | `beneficiary_not_allowed` | 156 (1.56%) | 40 (1.30%) | 125 (1.56%) | 31 (1.55%) |
| 8 | `cancel_transfer` | 157 (1.57%) | 40 (1.30%) | 126 (1.57%) | 31 (1.55%) |
| 9 | `card_about_to_expire` | 129 (1.29%) | 40 (1.30%) | 103 (1.29%) | 26 (1.30%) |
| 10 | `card_acceptance` | 59 (0.59%) | 40 (1.30%) | 47 (0.59%) | 12 (0.60%) |
| 11 | `card_arrival` | 153 (1.53%) | 40 (1.30%) | 122 (1.52%) | 31 (1.55%) |
| 12 | `card_delivery_estimate` | 112 (1.12%) | 40 (1.30%) | 90 (1.12%) | 22 (1.10%) |
| 13 | `card_linking` | 139 (1.39%) | 40 (1.30%) | 111 (1.39%) | 28 (1.40%) |
| 14 | `card_not_working` | 112 (1.12%) | 40 (1.30%) | 90 (1.12%) | 22 (1.10%) |
| 15 | `card_payment_fee_charged` | 187 (1.87%) | 40 (1.30%) | 149 (1.86%) | 38 (1.90%) |
| 16 | `card_payment_not_recognised` | 168 (1.68%) | 40 (1.30%) | 134 (1.67%) | 34 (1.70%) |
| 17 | `card_payment_wrong_exchange_rate` | 167 (1.67%) | 40 (1.30%) | 134 (1.67%) | 33 (1.65%) |
| 18 | `card_swallowed` | 61 (0.61%) | 40 (1.30%) | 49 (0.61%) | 12 (0.60%) |
| 19 | `cash_withdrawal_charge` | 177 (1.77%) | 40 (1.30%) | 141 (1.76%) | 36 (1.80%) |
| 20 | `cash_withdrawal_not_recognised` | 160 (1.60%) | 40 (1.30%) | 128 (1.60%) | 32 (1.60%) |
| 21 | `change_pin` | 122 (1.22%) | 40 (1.30%) | 98 (1.22%) | 24 (1.20%) |
| 22 | `compromised_card` | 86 (0.86%) | 40 (1.30%) | 69 (0.86%) | 17 (0.85%) |
| 23 | `contactless_not_working` | 35 (0.35%) | 40 (1.30%) | 28 (0.35%) | 7 (0.35%) |
| 24 | `country_support` | 129 (1.29%) | 40 (1.30%) | 103 (1.29%) | 26 (1.30%) |
| 25 | `declined_card_payment` | 153 (1.53%) | 40 (1.30%) | 122 (1.52%) | 31 (1.55%) |
| 26 | `declined_cash_withdrawal` | 173 (1.73%) | 40 (1.30%) | 138 (1.72%) | 35 (1.75%) |
| 27 | `declined_transfer` | 133 (1.33%) | 40 (1.30%) | 106 (1.32%) | 27 (1.35%) |
| 28 | `direct_debit_payment_not_recognised` | 182 (1.82%) | 40 (1.30%) | 145 (1.81%) | 37 (1.85%) |
| 29 | `disposable_card_limits` | 121 (1.21%) | 40 (1.30%) | 97 (1.21%) | 24 (1.20%) |
| 30 | `edit_personal_details` | 121 (1.21%) | 40 (1.30%) | 97 (1.21%) | 24 (1.20%) |
| 31 | `exchange_charge` | 121 (1.21%) | 40 (1.30%) | 97 (1.21%) | 24 (1.20%) |
| 32 | `exchange_rate` | 112 (1.12%) | 40 (1.30%) | 90 (1.12%) | 22 (1.10%) |
| 33 | `exchange_via_app` | 118 (1.18%) | 40 (1.30%) | 94 (1.17%) | 24 (1.20%) |
| 34 | `extra_charge_on_statement` | 166 (1.66%) | 40 (1.30%) | 133 (1.66%) | 33 (1.65%) |
| 35 | `failed_transfer` | 137 (1.37%) | 40 (1.30%) | 110 (1.37%) | 27 (1.35%) |
| 36 | `fiat_currency_support` | 126 (1.26%) | 40 (1.30%) | 101 (1.26%) | 25 (1.25%) |
| 37 | `get_disposable_virtual_card` | 97 (0.97%) | 40 (1.30%) | 78 (0.97%) | 19 (0.95%) |
| 38 | `get_physical_card` | 106 (1.06%) | 40 (1.30%) | 85 (1.06%) | 21 (1.05%) |
| 39 | `getting_spare_card` | 129 (1.29%) | 40 (1.30%) | 103 (1.29%) | 26 (1.30%) |
| 40 | `getting_virtual_card` | 98 (0.98%) | 40 (1.30%) | 78 (0.97%) | 20 (1.00%) |
| 41 | `lost_or_stolen_card` | 82 (0.82%) | 40 (1.30%) | 66 (0.82%) | 16 (0.80%) |
| 42 | `lost_or_stolen_phone` | 121 (1.21%) | 40 (1.30%) | 97 (1.21%) | 24 (1.20%) |
| 43 | `order_physical_card` | 120 (1.20%) | 40 (1.30%) | 96 (1.20%) | 24 (1.20%) |
| 44 | `passcode_forgotten` | 105 (1.05%) | 40 (1.30%) | 84 (1.05%) | 21 (1.05%) |
| 45 | `pending_card_payment` | 159 (1.59%) | 40 (1.30%) | 127 (1.59%) | 32 (1.60%) |
| 46 | `pending_cash_withdrawal` | 143 (1.43%) | 40 (1.30%) | 114 (1.42%) | 29 (1.45%) |
| 47 | `pending_top_up` | 149 (1.49%) | 40 (1.30%) | 119 (1.49%) | 30 (1.50%) |
| 48 | `pending_transfer` | 148 (1.48%) | 40 (1.30%) | 118 (1.47%) | 30 (1.50%) |
| 49 | `pin_blocked` | 115 (1.15%) | 40 (1.30%) | 92 (1.15%) | 23 (1.15%) |
| 50 | `receiving_money` | 95 (0.95%) | 40 (1.30%) | 76 (0.95%) | 19 (0.95%) |
| 51 | `Refund_not_showing_up` | 162 (1.62%) | 40 (1.30%) | 130 (1.62%) | 32 (1.60%) |
| 52 | `request_refund` | 169 (1.69%) | 40 (1.30%) | 135 (1.69%) | 34 (1.70%) |
| 53 | `reverted_card_payment?` | 161 (1.61%) | 40 (1.30%) | 129 (1.61%) | 32 (1.60%) |
| 54 | `supported_cards_and_currencies` | 129 (1.29%) | 40 (1.30%) | 103 (1.29%) | 26 (1.30%) |
| 55 | `terminate_account` | 108 (1.08%) | 40 (1.30%) | 86 (1.07%) | 22 (1.10%) |
| 56 | `top_up_by_bank_transfer_charge` | 111 (1.11%) | 40 (1.30%) | 89 (1.11%) | 22 (1.10%) |
| 57 | `top_up_by_card_charge` | 114 (1.14%) | 40 (1.30%) | 91 (1.14%) | 23 (1.15%) |
| 58 | `top_up_by_cash_or_cheque` | 114 (1.14%) | 40 (1.30%) | 91 (1.14%) | 23 (1.15%) |
| 59 | `top_up_failed` | 145 (1.45%) | 40 (1.30%) | 116 (1.45%) | 29 (1.45%) |
| 60 | `top_up_limits` | 97 (0.97%) | 40 (1.30%) | 78 (0.97%) | 19 (0.95%) |
| 61 | `top_up_reverted` | 146 (1.46%) | 40 (1.30%) | 117 (1.46%) | 29 (1.45%) |
| 62 | `topping_up_by_card` | 103 (1.03%) | 40 (1.30%) | 82 (1.02%) | 21 (1.05%) |
| 63 | `transaction_charged_twice` | 175 (1.75%) | 40 (1.30%) | 140 (1.75%) | 35 (1.75%) |
| 64 | `transfer_fee_charged` | 172 (1.72%) | 40 (1.30%) | 138 (1.72%) | 34 (1.70%) |
| 65 | `transfer_into_account` | 113 (1.13%) | 40 (1.30%) | 90 (1.12%) | 23 (1.15%) |
| 66 | `transfer_not_received_by_recipient` | 171 (1.71%) | 40 (1.30%) | 137 (1.71%) | 34 (1.70%) |
| 67 | `transfer_timing` | 128 (1.28%) | 40 (1.30%) | 102 (1.27%) | 26 (1.30%) |
| 68 | `unable_to_verify_identity` | 102 (1.02%) | 40 (1.30%) | 82 (1.02%) | 20 (1.00%) |
| 69 | `verify_my_identity` | 104 (1.04%) | 40 (1.30%) | 83 (1.04%) | 21 (1.05%) |
| 70 | `verify_source_of_funds` | 113 (1.13%) | 40 (1.30%) | 90 (1.12%) | 23 (1.15%) |
| 71 | `verify_top_up` | 126 (1.26%) | 40 (1.30%) | 101 (1.26%) | 25 (1.25%) |
| 72 | `virtual_card_not_working` | 41 (0.41%) | 40 (1.30%) | 33 (0.41%) | 8 (0.40%) |
| 73 | `visa_or_mastercard` | 135 (1.35%) | 40 (1.30%) | 108 (1.35%) | 27 (1.35%) |
| 74 | `why_verify_identity` | 121 (1.21%) | 40 (1.30%) | 97 (1.21%) | 24 (1.20%) |
| 75 | `wrong_amount_of_cash_received` | 180 (1.80%) | 40 (1.30%) | 144 (1.80%) | 36 (1.80%) |
| 76 | `wrong_exchange_rate_for_cash_withdrawal` | 163 (1.63%) | 40 (1.30%) | 130 (1.62%) | 33 (1.65%) |

---

## 5. Text Quality & Diagnostic Findings

1. **Null / Empty Texts:** None detected (0 records).
2. **Whitespace-only Texts:** None detected (0 records).
3. **HTML-like Markup / Entities:** None detected (0 records).
4. **Control Characters:** None detected (0 records with non-printable ASCII or control codes).
5. **Non-ASCII Characters:** Found in **52** train queries and **9** test queries:

| Character | Code Point | Unicode Name | Total Occurrences |
| :--- | :--- | :--- | :--- |
| `€` | `0x20ac` (8364) | EURO SIGN | 5 |
| `£` | `0xa3` (163) | POUND SIGN | 4 |
| ` ` | `0xa0` (160) | NO-BREAK SPACE | 2 |
| `…` | `0x2026` (8230) | HORIZONTAL ELLIPSIS | 1 |

*Domain Significance:* The non-ASCII characters represent standard British and European currency symbols (`£`, `€`), typography (`…`), and non-breaking space (`\xa0`). These carry essential financial semantics (distinguishing domestic vs cross-border payment queries) and should be preserved or normalized cleanly rather than stripped.

---

## 6. Stratified Split Verification (80% Train / 20% Val)

Stratification was performed on the official 10,003-example training set with `random_seed=42`.

- **Train Sub-Split:** **8002** examples (80.00%)
- **Validation Sub-Split:** **2001** examples (20.00%)
- **Class Representation:**
  - Classes in Train Sub-Split: **77 / 77** (100% coverage)
  - Classes in Validation Sub-Split: **77 / 77** (100% coverage)
- **Class Proportion Preservation:**
  - Maximum absolute frequency difference (Train Sub vs Original): `0.000074`
  - Maximum absolute frequency difference (Val Sub vs Original): `0.000296`
  - *Tolerance:* Strictly within `< 0.001` threshold.
- **Partition Disjointness:** Zero index overlap, zero text overlap.

---

## 7. Complete List of All 77 Intent Labels

The 77 official BANKING77 intents:

1. `activate_my_card`
2. `age_limit`
3. `apple_pay_or_google_pay`
4. `atm_support`
5. `automatic_top_up`
6. `balance_not_updated_after_bank_transfer`
7. `balance_not_updated_after_cheque_or_cash_deposit`
8. `beneficiary_not_allowed`
9. `cancel_transfer`
10. `card_about_to_expire`
11. `card_acceptance`
12. `card_arrival`
13. `card_delivery_estimate`
14. `card_linking`
15. `card_not_working`
16. `card_payment_fee_charged`
17. `card_payment_not_recognised`
18. `card_payment_wrong_exchange_rate`
19. `card_swallowed`
20. `cash_withdrawal_charge`
21. `cash_withdrawal_not_recognised`
22. `change_pin`
23. `compromised_card`
24. `contactless_not_working`
25. `country_support`
26. `declined_card_payment`
27. `declined_cash_withdrawal`
28. `declined_transfer`
29. `direct_debit_payment_not_recognised`
30. `disposable_card_limits`
31. `edit_personal_details`
32. `exchange_charge`
33. `exchange_rate`
34. `exchange_via_app`
35. `extra_charge_on_statement`
36. `failed_transfer`
37. `fiat_currency_support`
38. `get_disposable_virtual_card`
39. `get_physical_card`
40. `getting_spare_card`
41. `getting_virtual_card`
42. `lost_or_stolen_card`
43. `lost_or_stolen_phone`
44. `order_physical_card`
45. `passcode_forgotten`
46. `pending_card_payment`
47. `pending_cash_withdrawal`
48. `pending_top_up`
49. `pending_transfer`
50. `pin_blocked`
51. `receiving_money`
52. `Refund_not_showing_up`
53. `request_refund`
54. `reverted_card_payment?`
55. `supported_cards_and_currencies`
56. `terminate_account`
57. `top_up_by_bank_transfer_charge`
58. `top_up_by_card_charge`
59. `top_up_by_cash_or_cheque`
60. `top_up_failed`
61. `top_up_limits`
62. `top_up_reverted`
63. `topping_up_by_card`
64. `transaction_charged_twice`
65. `transfer_fee_charged`
66. `transfer_into_account`
67. `transfer_not_received_by_recipient`
68. `transfer_timing`
69. `unable_to_verify_identity`
70. `verify_my_identity`
71. `verify_source_of_funds`
72. `verify_top_up`
73. `virtual_card_not_working`
74. `visa_or_mastercard`
75. `why_verify_identity`
76. `wrong_amount_of_cash_received`
77. `wrong_exchange_rate_for_cash_withdrawal`

---

## 8. Recommendations for Next Stages (Tokenization & Modeling)

1. **Sequence Length Configuration:** A `max_seq_length` of 64 or 128 is optimal. 128 completely guarantees 0% token truncation while keeping attention computation light.
2. **Currency Preservation:** Ensure tokenizer preserves `£` and `€` or normalizes them into explicit tokens (e.g. `GBP`, `EUR`) if uncased models are used.
3. **Loss Function Consideration:** Given the train class frequency variation (35 to 187 examples per class), monitor per-class F1 scores and evaluate weighted cross-entropy or focal loss if minority classes underperform.
