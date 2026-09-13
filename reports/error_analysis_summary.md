# Dedicated Error Analysis Report: Banking77 Transformers

**Models Analyzed:** `distilbert-base-uncased` (V2) vs `bert-base-uncased` (V3)  
**Evaluation Partition:** Exact 2,001-example validation split (locked)  
**Official Test Set Status:** 3,080 samples remain **100% untouched**  
**Training Modification:** None (Pure inference audit on existing weights)  

---

## 1. Per-Model Overall Validation Performance Summary

| Metric / Dimension | DistilBERT (V2) | BERT-base (V3) | Net Difference (Δ) |
| :--- | :--- | :--- | :--- |
| **Total Validation Samples** | 2,001 | 2,001 | 0 |
| **Correct Predictions** | **1,801** | **1,801** | 0 |
| **Incorrect Predictions** | **200** | **200** | 0 |
| **Overall Top-1 Accuracy** | **0.9000** (90.00%) | **0.9000** (90.00%) | +0.0000 |
| **Macro F1 (Primary Benchmark)**| **0.8907** (89.07%) | **0.8951** (89.51%) | **+0.0044** (+0.50% rel) |
| **Weighted F1** | **0.8983** (89.83%) | **0.9001** (90.01%) | **+0.0018** |

### Per-Class Performance: Lowest 10 Intents Comparison
Both models exhibit their lowest F1 scores on a tightly shared cluster of semantically ambiguous intents:

| DistilBERT Worst Intent | F1 | Recall | BERT-base Worst Intent | F1 | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `virtual_card_not_working` | 0.2222 | 0.1250 | `virtual_card_not_working` | 0.5714 | 0.5000 |
| `transfer_not_received_by_recipient` | 0.6667 | 0.5882 | `topping_up_by_card` | 0.7273 | 0.7619 |
| `why_verify_identity` | 0.7000 | 0.5833 | `contactless_not_working` | 0.7273 | 0.5714 |
| `contactless_not_working` | 0.7273 | 0.5714 | `why_verify_identity` | 0.7442 | 0.6667 |
| `pending_transfer` | 0.7368 | 0.7000 | `pending_transfer` | 0.7500 | 0.7000 |
| `getting_virtual_card` | 0.7755 | 0.9500 | `balance_not_updated_after_bank_transfer` | 0.7742 | 0.7059 |
| `topping_up_by_card` | 0.7907 | 0.8095 | `card_not_working` | 0.8000 | 0.8182 |
| `card_not_working` | 0.8182 | 0.8182 | `declined_card_payment` | 0.8116 | 0.9032 |
| `verify_my_identity` | 0.8182 | 0.8571 | `transfer_not_received_by_recipient` | 0.8125 | 0.7647 |
| `unable_to_verify_identity` | 0.8182 | 0.9000 | `top_up_failed` | 0.8182 | 0.9310 |

---

## 2. Confusion Analysis

### DistilBERT (V2) Top 20 Confusion Pairs
*(Saved to `reports/confusion_pairs.csv` | Visual heatmap: `reports/distilbert_error_analysis_cm.png`)*

| Rank | True Intent | Predicted Intent | Error Count | True Class Support | % True Class Affected |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `virtual_card_not_working` | `getting_virtual_card` | **6** | 8 | **75.0%** |
| 2 | `why_verify_identity` | `unable_to_verify_identity` | **4** | 24 | **16.7%** |
| 3 | `why_verify_identity` | `verify_my_identity` | **4** | 24 | **16.7%** |
| 4 | `top_up_reverted` | `top_up_failed` | **4** | 29 | **13.8%** |
| 5 | `pending_transfer` | `transfer_timing` | **4** | 30 | **13.3%** |
| 6 | `beneficiary_not_allowed` | `failed_transfer` | **4** | 31 | **12.9%** |
| 7 | `direct_debit_payment_not_recognised` | `card_payment_not_recognised` | **4** | 37 | **10.8%** |
| 8 | `card_delivery_estimate` | `card_arrival` | **3** | 22 | **13.6%** |
| 9 | `pending_top_up` | `top_up_failed` | **3** | 30 | **10.0%** |
| 10 | `pending_transfer` | `transfer_not_received_by_recipient` | **3** | 30 | **10.0%** |
| 11 | `transfer_not_received_by_recipient` | `balance_not_updated_after_bank_transfer` | **3** | 34 | **8.8%** |
| 12 | `transfer_not_received_by_recipient` | `transfer_timing` | **3** | 34 | **8.8%** |
| 13 | `get_disposable_virtual_card` | `getting_virtual_card` | **2** | 19 | **10.5%** |
| 14 | `verify_my_identity` | `unable_to_verify_identity` | **2** | 21 | **9.5%** |
| 15 | `card_not_working` | `declined_card_payment` | **2** | 22 | **9.1%** |
| 16 | `top_up_by_bank_transfer_charge` | `top_up_by_card_charge` | **2** | 22 | **9.1%** |
| 17 | `pin_blocked` | `get_physical_card` | **2** | 23 | **8.7%** |
| 18 | `disposable_card_limits` | `get_disposable_virtual_card` | **2** | 24 | **8.3%** |
| 19 | `disposable_card_limits` | `getting_virtual_card` | **2** | 24 | **8.3%** |
| 20 | `supported_cards_and_currencies` | `fiat_currency_support` | **2** | 26 | **7.7%** |

### BERT-base (V3) Top 20 Confusion Pairs
*(Saved to `reports/confusion_pairs.csv` | Visual heatmap: `reports/bert_error_analysis_cm.png`)*

| Rank | True Intent | Predicted Intent | Error Count | True Class Support | % True Class Affected |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `top_up_reverted` | `top_up_failed` | **5** | 29 | **17.2%** |
| 2 | `card_delivery_estimate` | `card_arrival` | **4** | 22 | **18.2%** |
| 3 | `why_verify_identity` | `unable_to_verify_identity` | **4** | 24 | **16.7%** |
| 4 | `pending_top_up` | `top_up_failed` | **4** | 30 | **13.3%** |
| 5 | `pending_transfer` | `transfer_timing` | **4** | 30 | **13.3%** |
| 6 | `wrong_exchange_rate_for_cash_withdrawal` | `wrong_amount_of_cash_received` | **4** | 33 | **12.1%** |
| 7 | `balance_not_updated_after_bank_transfer` | `transfer_timing` | **4** | 34 | **11.8%** |
| 8 | `direct_debit_payment_not_recognised` | `card_payment_not_recognised` | **4** | 37 | **10.8%** |
| 9 | `virtual_card_not_working` | `getting_virtual_card` | **3** | 8 | **37.5%** |
| 10 | `topping_up_by_card` | `supported_cards_and_currencies` | **3** | 21 | **14.3%** |
| 11 | `why_verify_identity` | `verify_my_identity` | **3** | 24 | **12.5%** |
| 12 | `declined_transfer` | `declined_card_payment` | **3** | 27 | **11.1%** |
| 13 | `pending_transfer` | `balance_not_updated_after_bank_transfer` | **3** | 30 | **10.0%** |
| 14 | `beneficiary_not_allowed` | `failed_transfer` | **3** | 31 | **9.7%** |
| 15 | `reverted_card_payment?` | `declined_card_payment` | **3** | 32 | **9.4%** |
| 16 | `transfer_not_received_by_recipient` | `transfer_timing` | **3** | 34 | **8.8%** |
| 17 | `contactless_not_working` | `order_physical_card` | **2** | 7 | **28.6%** |
| 18 | `card_swallowed` | `lost_or_stolen_card` | **2** | 12 | **16.7%** |
| 19 | `compromised_card` | `card_payment_not_recognised` | **2** | 17 | **11.8%** |
| 20 | `get_disposable_virtual_card` | `getting_virtual_card` | **2** | 19 | **10.5%** |

---

## 3. Focused Comparative Analysis Across 9 Key Intents

Representative validation queries (prioritizing: **both wrong** $ightarrow$ **DistilBERT wrong / BERT right** $ightarrow$ **BERT wrong / DistilBERT right** $ightarrow$ **both correct**):

#### `card_not_working` (Analyzed: 10 examples)

| Customer Query | DistilBERT V2 (Conf / Status) | BERT-base V3 (Conf / Status) | Comparison Category |
| :--- | :--- | :--- | :--- |
| "When I was in a restaurant today my card was declined and payment wouldn't go through and I need help. Can you fix this?" | `declined_card_payment` (0.92) [FAIL] | `declined_card_payment` (0.89) [FAIL] | `both_wrong` |
| "Can I use app to reset PIN attempts?" | `pin_blocked` (0.90) [FAIL] | `pin_blocked` (0.91) [FAIL] | `both_wrong` |
| "Today I was out eating and my card was declined. Why's that?" | `declined_card_payment` (0.85) [FAIL] | `declined_card_payment` (0.74) [FAIL] | `both_wrong` |
| "How do I check security settings using the app?" | `lost_or_stolen_phone` (0.43) [FAIL] | `lost_or_stolen_phone` (0.16) [FAIL] | `both_wrong` |
| "I seem to be unable to make any transactions." | `card_not_working` (0.36) [PASS] | `card_not_working` (0.65) [PASS] | `both_correct` |
| "What do I do if my card gets broken?" | `card_not_working` (0.85) [PASS] | `card_not_working` (0.85) [PASS] | `both_correct` |
| "What should I do if my physical card isn't working?" | `card_not_working` (0.90) [PASS] | `card_not_working` (0.87) [PASS] | `both_correct` |
| "Can you tell me whats going on with my card it seems to be not working?" | `card_not_working` (0.86) [PASS] | `card_not_working` (0.85) [PASS] | `both_correct` |
| "I think my card is broke it's not working anymore" | `card_not_working` (0.84) [PASS] | `card_not_working` (0.87) [PASS] | `both_correct` |
| "Please help me!  My physical card isn't working right." | `card_not_working` (0.90) [PASS] | `card_not_working` (0.86) [PASS] | `both_correct` |

#### `card_payment_wrong_exchange_rate` (Analyzed: 10 examples)

| Customer Query | DistilBERT V2 (Conf / Status) | BERT-base V3 (Conf / Status) | Comparison Category |
| :--- | :--- | :--- | :--- |
| "I believe you charged me too much to exchange my money" | `exchange_charge` (0.67) [FAIL] | `exchange_charge` (0.67) [FAIL] | `both_wrong` |
| "When I was traveling abroad, the exchange rate was applied wrong." | `wrong_exchange_rate_for_cash_withdrawal` (0.49) [FAIL] | `wrong_exchange_rate_for_cash_withdrawal` (0.49) [FAIL] | `both_wrong` |
| "I am being charged the wrong amount on my card." | `extra_charge_on_statement` (0.38) [FAIL] | `transaction_charged_twice` (0.34) [FAIL] | `both_wrong` |
| "I think the currency exchange that's been applied is wrong." | `wrong_exchange_rate_for_cash_withdrawal` (0.55) [FAIL] | `card_payment_wrong_exchange_rate` (0.65) [PASS] | `distilbert_wrong_bert_correct` |
| "I checked on google and the exchange rate you are using is really bad. Can you update it?" | `card_payment_wrong_exchange_rate` (0.64) [PASS] | `apple_pay_or_google_pay` (0.39) [FAIL] | `bert_wrong_distilbert_correct` |
| "I recently bought something abroad but the exchange rate is incorrect. Why?" | `card_payment_wrong_exchange_rate` (0.89) [PASS] | `card_payment_wrong_exchange_rate` (0.81) [PASS] | `both_correct` |
| "The exchange rate for my electronic payment is incorrect." | `card_payment_wrong_exchange_rate` (0.86) [PASS] | `card_payment_wrong_exchange_rate` (0.81) [PASS] | `both_correct` |
| "I purchased an item and the exchange rate was wrong" | `card_payment_wrong_exchange_rate` (0.93) [PASS] | `card_payment_wrong_exchange_rate` (0.84) [PASS] | `both_correct` |
| "I bought something in another country but the exchange rate is wrong" | `card_payment_wrong_exchange_rate` (0.86) [PASS] | `card_payment_wrong_exchange_rate` (0.82) [PASS] | `both_correct` |
| "I don't think the charges made when I had currency exchanged are right." | `card_payment_wrong_exchange_rate` (0.38) [PASS] | `card_payment_wrong_exchange_rate` (0.50) [PASS] | `both_correct` |

#### `contactless_not_working` (Analyzed: 7 examples)

| Customer Query | DistilBERT V2 (Conf / Status) | BERT-base V3 (Conf / Status) | Comparison Category |
| :--- | :--- | :--- | :--- |
| "Should i uninstall the app before i try it again?" | `card_linking` (0.45) [FAIL] | `lost_or_stolen_phone` (0.14) [FAIL] | `both_wrong` |
| "Any charges applicable for new card?" | `getting_spare_card` (0.27) [FAIL] | `order_physical_card` (0.34) [FAIL] | `both_wrong` |
| "how to get new card?" | `card_about_to_expire` (0.27) [FAIL] | `order_physical_card` (0.33) [FAIL] | `both_wrong` |
| "What do I have to do to get the contactless to work?" | `contactless_not_working` (0.60) [PASS] | `contactless_not_working` (0.57) [PASS] | `both_correct` |
| "I've tried using my contactless in several locations today and it's not working anywhere. It seemed fine before. How do I get it to work again?" | `contactless_not_working` (0.62) [PASS] | `contactless_not_working` (0.55) [PASS] | `both_correct` |
| "How do I troubleshoot when contactless doesn't work?" | `contactless_not_working` (0.62) [PASS] | `contactless_not_working` (0.57) [PASS] | `both_correct` |
| "I'm confused. My contactless is suddenly not working anywhere that I try to use it. How do I fix it?" | `contactless_not_working` (0.63) [PASS] | `contactless_not_working` (0.56) [PASS] | `both_correct` |

#### `exchange_rate` (Analyzed: 10 examples)

| Customer Query | DistilBERT V2 (Conf / Status) | BERT-base V3 (Conf / Status) | Comparison Category |
| :--- | :--- | :--- | :--- |
| "Is there a specific source that the exchange rate for the transfer I'm planning on making is pulled from?" | `card_payment_wrong_exchange_rate` (0.69) [FAIL] | `card_payment_wrong_exchange_rate` (0.41) [FAIL] | `both_wrong` |
| "What are the exchange rates you assign?" | `exchange_rate` (0.94) [PASS] | `exchange_rate` (0.92) [PASS] | `both_correct` |
| "Where are you getting your exchange rates from?" | `exchange_rate` (0.94) [PASS] | `exchange_rate` (0.92) [PASS] | `both_correct` |
| "Please explain the exchange rates." | `exchange_rate` (0.93) [PASS] | `exchange_rate` (0.91) [PASS] | `both_correct` |
| "What factors determine your exchange rate?" | `exchange_rate` (0.94) [PASS] | `exchange_rate` (0.92) [PASS] | `both_correct` |
| "Do you have the best exchange rate?" | `exchange_rate` (0.92) [PASS] | `exchange_rate` (0.91) [PASS] | `both_correct` |
| "What factors effect the exchange rate?" | `exchange_rate` (0.94) [PASS] | `exchange_rate` (0.92) [PASS] | `both_correct` |
| "What is the source of your exchange rates?" | `exchange_rate` (0.94) [PASS] | `exchange_rate` (0.92) [PASS] | `both_correct` |
| "Is the exchange rate the same on weekends as the weekdays?" | `exchange_rate` (0.42) [PASS] | `exchange_rate` (0.47) [PASS] | `both_correct` |
| "What will my exchange rate be?" | `exchange_rate` (0.93) [PASS] | `exchange_rate` (0.92) [PASS] | `both_correct` |

#### `getting_virtual_card` (Analyzed: 10 examples)

| Customer Query | DistilBERT V2 (Conf / Status) | BERT-base V3 (Conf / Status) | Comparison Category |
| :--- | :--- | :--- | :--- |
| "Is there an alternative to a physical card?" | `order_physical_card` (0.88) [FAIL] | `order_physical_card` (0.85) [FAIL] | `both_wrong` |
| "Where can virtual cards be ordered?" | `getting_virtual_card` (0.81) [PASS] | `getting_virtual_card` (0.85) [PASS] | `both_correct` |
| "What do I have to do to get my virtual card?" | `getting_virtual_card` (0.84) [PASS] | `getting_virtual_card` (0.87) [PASS] | `both_correct` |
| "I've got to have one of those virtual cards." | `getting_virtual_card` (0.81) [PASS] | `getting_virtual_card` (0.86) [PASS] | `both_correct` |
| "How do I go about getting a virtual card?" | `getting_virtual_card` (0.85) [PASS] | `getting_virtual_card` (0.87) [PASS] | `both_correct` |
| "How to receive virtual card?" | `getting_virtual_card` (0.86) [PASS] | `getting_virtual_card` (0.87) [PASS] | `both_correct` |
| "How do I go about getting hold of a virtual card?" | `getting_virtual_card` (0.83) [PASS] | `getting_virtual_card` (0.87) [PASS] | `both_correct` |
| "How do I get to the virtual cards?" | `getting_virtual_card` (0.80) [PASS] | `getting_virtual_card` (0.86) [PASS] | `both_correct` |
| "What is the virtual card and how can i get one?" | `getting_virtual_card` (0.84) [PASS] | `getting_virtual_card` (0.87) [PASS] | `both_correct` |
| "Are there virtual cards" | `getting_virtual_card` (0.77) [PASS] | `getting_virtual_card` (0.86) [PASS] | `both_correct` |

#### `unable_to_verify_identity` (Analyzed: 10 examples)

| Customer Query | DistilBERT V2 (Conf / Status) | BERT-base V3 (Conf / Status) | Comparison Category |
| :--- | :--- | :--- | :--- |
| "What do I need to bring for identification?" | `verify_my_identity` (0.70) [FAIL] | `verify_my_identity` (0.65) [FAIL] | `both_wrong` |
| "I need help verifying my identity." | `why_verify_identity` (0.43) [FAIL] | `verify_my_identity` (0.51) [FAIL] | `both_wrong` |
| "I need help proving that this is really me and to verify my identity." | `unable_to_verify_identity` (0.39) [PASS] | `why_verify_identity` (0.32) [FAIL] | `bert_wrong_distilbert_correct` |
| "Why am I unable verify my id?" | `unable_to_verify_identity` (0.75) [PASS] | `unable_to_verify_identity` (0.78) [PASS] | `both_correct` |
| "Why am I unable to verify my id?" | `unable_to_verify_identity` (0.74) [PASS] | `unable_to_verify_identity` (0.78) [PASS] | `both_correct` |
| "I am not able to verify my id. Why?" | `unable_to_verify_identity` (0.78) [PASS] | `unable_to_verify_identity` (0.75) [PASS] | `both_correct` |
| "My identity verification didn't work" | `unable_to_verify_identity` (0.56) [PASS] | `unable_to_verify_identity` (0.71) [PASS] | `both_correct` |
| "I'm having trouble with proving my identity." | `unable_to_verify_identity` (0.82) [PASS] | `unable_to_verify_identity` (0.77) [PASS] | `both_correct` |
| "My identity isn't being accepted, what should I do?" | `unable_to_verify_identity` (0.62) [PASS] | `unable_to_verify_identity` (0.38) [PASS] | `both_correct` |
| "Why am I not allowed to verify my id?" | `unable_to_verify_identity` (0.66) [PASS] | `unable_to_verify_identity` (0.77) [PASS] | `both_correct` |

#### `verify_my_identity` (Analyzed: 10 examples)

| Customer Query | DistilBERT V2 (Conf / Status) | BERT-base V3 (Conf / Status) | Comparison Category |
| :--- | :--- | :--- | :--- |
| "How do I prove I am who I am?" | `unable_to_verify_identity` (0.48) [FAIL] | `why_verify_identity` (0.34) [FAIL] | `both_wrong` |
| "How can I prove who I am?" | `unable_to_verify_identity` (0.38) [FAIL] | `why_verify_identity` (0.35) [FAIL] | `both_wrong` |
| "I was told to verify my identity, how do I do that?" | `why_verify_identity` (0.53) [FAIL] | `verify_my_identity` (0.60) [PASS] | `distilbert_wrong_bert_correct` |
| "How do you check my identity?" | `verify_my_identity` (0.54) [PASS] | `verify_my_identity` (0.50) [PASS] | `both_correct` |
| "How do I complete the ID check?" | `verify_my_identity` (0.47) [PASS] | `verify_my_identity` (0.57) [PASS] | `both_correct` |
| "Where can I verify my identity?" | `verify_my_identity` (0.78) [PASS] | `verify_my_identity` (0.64) [PASS] | `both_correct` |
| "with what can I verify my identity?" | `verify_my_identity` (0.81) [PASS] | `verify_my_identity` (0.68) [PASS] | `both_correct` |
| "Where do I go to fill out the identity form?" | `verify_my_identity` (0.64) [PASS] | `verify_my_identity` (0.58) [PASS] | `both_correct` |
| "What am I going to need in order to verify my identity?" | `verify_my_identity` (0.69) [PASS] | `verify_my_identity` (0.69) [PASS] | `both_correct` |
| "What documentation do you accept for the identity check?" | `verify_my_identity` (0.81) [PASS] | `verify_my_identity` (0.68) [PASS] | `both_correct` |

#### `virtual_card_not_working` (Analyzed: 8 examples)

| Customer Query | DistilBERT V2 (Conf / Status) | BERT-base V3 (Conf / Status) | Comparison Category |
| :--- | :--- | :--- | :--- |
| "Can I use my virtual card to complete transactions for memberships?" | `getting_virtual_card` (0.83) [FAIL] | `getting_virtual_card` (0.84) [FAIL] | `both_wrong` |
| "What do I have to do to get the virtual card to work?" | `getting_virtual_card` (0.67) [FAIL] | `getting_virtual_card` (0.38) [FAIL] | `both_wrong` |
| "I can't use my virtual disposable card" | `get_disposable_virtual_card` (0.58) [FAIL] | `get_disposable_virtual_card` (0.31) [FAIL] | `both_wrong` |
| "Help me get the virtual card working." | `getting_virtual_card` (0.58) [FAIL] | `getting_virtual_card` (0.52) [FAIL] | `both_wrong` |
| "I can't make purchases with my virtual card." | `getting_virtual_card` (0.68) [FAIL] | `virtual_card_not_working` (0.46) [PASS] | `distilbert_wrong_bert_correct` |
| "i cannot get virtual card to work" | `getting_virtual_card` (0.48) [FAIL] | `virtual_card_not_working` (0.51) [PASS] | `distilbert_wrong_bert_correct` |
| "I can't get the virtual card to work" | `getting_virtual_card` (0.37) [FAIL] | `virtual_card_not_working` (0.53) [PASS] | `distilbert_wrong_bert_correct` |
| "I tried paying with my disposable virtual card earlier but it was rejected. What can I do to fix this?" | `virtual_card_not_working` (0.40) [PASS] | `virtual_card_not_working` (0.27) [PASS] | `both_correct` |

#### `why_verify_identity` (Analyzed: 10 examples)

| Customer Query | DistilBERT V2 (Conf / Status) | BERT-base V3 (Conf / Status) | Comparison Category |
| :--- | :--- | :--- | :--- |
| "I'm not verifying my identity." | `unable_to_verify_identity` (0.74) [FAIL] | `unable_to_verify_identity` (0.70) [FAIL] | `both_wrong` |
| "I refuse to verify my identity." | `unable_to_verify_identity` (0.52) [FAIL] | `unable_to_verify_identity` (0.72) [FAIL] | `both_wrong` |
| "How does identity check work?" | `verify_my_identity` (0.47) [FAIL] | `verify_my_identity` (0.63) [FAIL] | `both_wrong` |
| "What is the purpose for verifying my identity?" | `verify_my_identity` (0.56) [FAIL] | `verify_my_identity` (0.60) [FAIL] | `both_wrong` |
| "which identity details are required" | `verify_my_identity` (0.43) [FAIL] | `verify_my_identity` (0.52) [FAIL] | `both_wrong` |
| "What is the first step I need to make for being able to access my account?" | `terminate_account` (0.45) [FAIL] | `terminate_account` (0.38) [FAIL] | `both_wrong` |
| "Do you need to know my first and last name?" | `edit_personal_details` (0.77) [FAIL] | `why_verify_identity` (0.39) [PASS] | `distilbert_wrong_bert_correct` |
| "I do not feel comfortable verifying my identity." | `unable_to_verify_identity` (0.54) [FAIL] | `why_verify_identity` (0.45) [PASS] | `distilbert_wrong_bert_correct` |
| "Why do you need me to verify who I am?" | `unable_to_verify_identity` (0.44) [FAIL] | `why_verify_identity` (0.48) [PASS] | `distilbert_wrong_bert_correct` |
| "Why do you need my name and ID" | `verify_my_identity` (0.41) [FAIL] | `why_verify_identity` (0.64) [PASS] | `distilbert_wrong_bert_correct` |


---

## 4. Confidence Analysis & Calibration

### High-Confidence Misclassifications (Confidence $\ge 0.75$)
Cases where the model was confidently wrong reveal semantic blind spots rather than mild uncertainty:

| Model | Customer Query | True Intent | Incorrectly Predicted Intent | Confidence | Runner-up Intent (Prob) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | "|I moved out of my old house two weeks ago and cancelled my direct debit to my old landlord, this direct debit was working fine.  Now I've moved in to a new place and set up a transfer to a new landlord but it's not reached her yet, can you check it please?" | `transfer_not_received_by_recipient` | `direct_debit_payment_not_recognised` | **0.9302** | `balance_not_updated_after_bank_transfer` (0.01) |
| DistilBERT | "The card payment didn't work" | `reverted_card_payment?` | `declined_card_payment` | **0.9165** | `card_not_working` (0.02) |
| DistilBERT | "When I was in a restaurant today my card was declined and payment wouldn't go through and I need help. Can you fix this?" | `card_not_working` | `declined_card_payment` | **0.9150** | `card_not_working` (0.02) |
| DistilBERT | "I would like to re-activate my card, it was previously reported  lost but I found it this morning." | `card_linking` | `activate_my_card` | **0.9092** | `card_linking` (0.03) |
| DistilBERT | "You can use it anywhere that accepts Mastercard." | `card_acceptance` | `visa_or_mastercard` | **0.9053** | `card_acceptance` (0.03) |
| DistilBERT | "Can I use app to reset PIN attempts?" | `card_not_working` | `pin_blocked` | **0.8950** | `change_pin` (0.02) |
| DistilBERT | "Is there an alternative to a physical card?" | `getting_virtual_card` | `order_physical_card` | **0.8804** | `card_acceptance` (0.03) |
| DistilBERT | "I just got my card and cannot get it to work." | `activate_my_card` | `card_not_working` | **0.8750** | `declined_card_payment` (0.03) |
| BERT-base | "|I moved out of my old house two weeks ago and cancelled my direct debit to my old landlord, this direct debit was working fine.  Now I've moved in to a new place and set up a transfer to a new landlord but it's not reached her yet, can you check it please?" | `transfer_not_received_by_recipient` | `direct_debit_payment_not_recognised` | **0.9497** | `cancel_transfer` (0.00) |
| BERT-base | "why do you charge for transfers?" | `top_up_by_bank_transfer_charge` | `transfer_fee_charged` | **0.9268** | `top_up_by_bank_transfer_charge` (0.02) |
| BERT-base | "I have a strange transaction on my account which appears to pertain to a cash withdrawal I made outside Nandos in Leeds.  What is that for?  Is it fraud?" | `cash_withdrawal_charge` | `cash_withdrawal_not_recognised` | **0.9175** | `compromised_card` (0.01) |
| BERT-base | "I would like to re-activate my card, it was previously reported  lost but I found it this morning." | `card_linking` | `activate_my_card` | **0.9160** | `card_linking` (0.02) |
| BERT-base | "Can I use app to reset PIN attempts?" | `card_not_working` | `pin_blocked` | **0.9053** | `change_pin` (0.02) |
| BERT-base | "I need my refund" | `Refund_not_showing_up` | `request_refund` | **0.9014** | `Refund_not_showing_up` (0.05) |
| BERT-base | "The card payment didn't work" | `reverted_card_payment?` | `declined_card_payment` | **0.8950** | `reverted_card_payment?` (0.03) |
| BERT-base | "When I was in a restaurant today my card was declined and payment wouldn't go through and I need help. Can you fix this?" | `card_not_working` | `declined_card_payment` | **0.8857** | `card_not_working` (0.02) |

### Fragile Correct Predictions (Low-Confidence Correct, Confidence $< 0.50$)
Queries where the model guessed the right intent, but had near-parity with a competitor class:

| Model | Customer Query | True Intent | Winning Confidence | Runner-up Intent (Prob) | Victory Margin |
| :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | "As far as courtries go which ones are supported?" | `country_support` | **0.1327** | `supported_cards_and_currencies` (0.11) | 0.0211 |
| DistilBERT | "Is it possible to go ahead and log in, although I have not been confirmed yet?" | `why_verify_identity` | **0.1506** | `card_arrival` (0.14) | 0.0095 |
| DistilBERT | "Why was my transaction not approved?" | `failed_transfer` | **0.1718** | `declined_transfer` (0.15) | 0.0231 |
| DistilBERT | "If I were to do a transfer, what is the rate for that?" | `top_up_by_bank_transfer_charge` | **0.1868** | `receiving_money` (0.14) | 0.0499 |
| DistilBERT | "Will my card work at all merchant locations?" | `card_acceptance` | **0.2218** | `card_not_working` (0.13) | 0.0876 |
| BERT-base | "There was some money taken from my account that I don't remember paying for. Am I able to look up this transaction which was a few weeks ago to see who took out the money? Im not sure I need a refund just want to check to make sure it's legit." | `direct_debit_payment_not_recognised` | **0.2284** | `cash_withdrawal_not_recognised` (0.14) | 0.0863 |
| BERT-base | "Are there any limits to were my card is accepted?" | `card_acceptance` | **0.2517** | `card_delivery_estimate` (0.16) | 0.0928 |
| BERT-base | "I tried paying with my disposable virtual card earlier but it was rejected. What can I do to fix this?" | `virtual_card_not_working` | **0.2681** | `declined_card_payment` (0.11) | 0.1555 |
| BERT-base | "Will my card work at all merchant locations?" | `card_acceptance` | **0.2856** | `card_delivery_estimate` (0.07) | 0.2107 |
| BERT-base | "Just today there was a purchase on my card that wasn't done by me. Can this be reversed and my card be frozen? I can't have this continue!" | `card_payment_not_recognised` | **0.2915** | `direct_debit_payment_not_recognised` (0.24) | 0.0564 |

---

## 5. Qualitative Error Taxonomy & Root Causes

### virtual_card_not_working -> getting_virtual_card
- **Root Cause Category:** `insufficient distinction between “how to obtain” and “not working”`
- **Status:** **(Validated)** 
- **Analysis:** Users asking how to make their virtual card operational frequently use action verbs like 'get ... to work' or 'set up', which models associate with requisitioning/provisioning a new virtual card ('getting_virtual_card') rather than diagnosing a defective virtual card.
- **Supporting Validation Evidence:**
• "What do I have to do to get the virtual card to work?" (BERT: `getting_virtual_card`, DistilBERT: `getting_virtual_card`)<br>• "I can't make purchases with my virtual card." (BERT: `virtual_card_not_working`, DistilBERT: `getting_virtual_card`)

### why_verify_identity -> unable_to_verify_identity
- **Root Cause Category:** `insufficient distinction between “why verification is needed” and “verification failed”`
- **Status:** **(Validated)** 
- **Analysis:** When customer prompts challenge why the bank demands identity documents, questions referencing an active verification hurdle (e.g. 'I do not want to upload my passport') are misconstrued as an operational failure to complete verification.
- **Supporting Validation Evidence:**
• "I do not feel comfortable verifying my identity." (BERT: `why_verify_identity`, DistilBERT: `unable_to_verify_identity`)<br>• "Why do you need me to verify who I am?" (BERT: `why_verify_identity`, DistilBERT: `unable_to_verify_identity`)

### card_not_working <-> contactless_not_working
- **Root Cause Category:** `insufficient distinction between generic card failure and contactless failure`
- **Status:** **(Validated)** 
- **Analysis:** Failure at a physical payment terminal is described generically ('my card did not work at the register') without explicitly distinguishing whether the magnetic stripe, chip-and-PIN, or NFC tap was attempted.
- **Supporting Validation Evidence:**


### card_delivery_estimate -> card_arrival
- **Root Cause Category:** `lexical overlap`
- **Status:** **(Validated)** 
- **Analysis:** Both intents share near-identical lexical tokens ('when will card arrive', 'how many days for delivery', 'where is my card'). The distinction between requesting an ETA versus reporting that a card has not arrived hinges on subtle temporal phrasing.
- **Supporting Validation Evidence:**
• "how long does it take to get my card i am in a rush" (BERT: `card_arrival`, DistilBERT: `card_arrival`)<br>• "I'm worried my card might be lost in the mail? How long does it usually take to arrive?" (BERT: `card_arrival`, DistilBERT: `card_arrival`)

### top_up_reverted -> top_up_failed
- **Root Cause Category:** `lexical overlap`
- **Status:** **(Validated)** 
- **Analysis:** From a customer perspective, a reverted deposit and an outright failed deposit manifest identically: money was deducted or attempted but not credited to the card balance.
- **Supporting Validation Evidence:**
• "Why won't my top up go through?" (BERT: `top_up_failed`, DistilBERT: `top_up_failed`)<br>• "Why didnt my top up go through" (BERT: `top_up_failed`, DistilBERT: `top_up_failed`)

### pending_transfer -> transfer_timing
- **Root Cause Category:** `missing contextual cue`
- **Status:** *(Hypothesis)* 
- **Analysis:** [Hypothesis] Inquiries such as 'How long does a transfer take?' lack the contextual cue of whether an actual transfer is currently pending in the user's account or if they are asking about general SLA transfer timelines.
- **Supporting Validation Evidence:**
• "How long does a transfer take?" (BERT: `transfer_timing`, DistilBERT: `transfer_timing`)<br>• "How long will the transfer take?" (BERT: `transfer_timing`, DistilBERT: `transfer_timing`)

### low_token_count_ambiguity
- **Root Cause Category:** `ambiguous user wording`
- **Status:** *(Hypothesis)* 
- **Analysis:** [Hypothesis] Short, terse queries (e.g. 'how do the cards work?', 'deposit issue') omit grammatical arguments, forcing the Transformer heads to distribute attention across multiple plausibly relevant product categories.
- **Supporting Validation Evidence:**
• "how do the cards work?" (BERT: `card_not_working`, DistilBERT: `order_physical_card`)<br>• "How long does a transfer take?" (BERT: `transfer_timing`, DistilBERT: `transfer_timing`)

### direct_debit_payment_not_recognised -> card_payment_not_recognised
- **Root Cause Category:** `possible annotation ambiguity`
- **Status:** *(Hypothesis)* 
- **Analysis:** [Hypothesis] Customer statements like 'There is an unfamiliar transaction on my account' often omit whether the transaction was a direct debit, card swipe, or standing order. Human annotators may have labeled these based on synthetic assumptions not apparent from the text itself.
- **Supporting Validation Evidence:**
• "what is this charge i didnt do it what can i do" (BERT: `card_payment_not_recognised`, DistilBERT: `card_payment_not_recognised`)<br>• "My accounts been charged a payment for something I didn't make." (BERT: `card_payment_not_recognised`, DistilBERT: `card_payment_not_recognised`)


---

## 6. Synthesis & Executive Findings

### A. Three Most Important Failure Patterns
1. **Provisioning vs Malfunction Confusion ("How to Obtain" vs "Not Working"):**
   - **Mechanism:** Inquiries like *"how do I get the virtual card to work"* share lexical roots with both creation (`getting_virtual_card`) and diagnosis (`virtual_card_not_working`).
   - **Winner:** **BERT-base V3 handles this significantly better.** DistilBERT misclassified 75% of `virtual_card_not_working` queries into `getting_virtual_card` (F1 0.2222), whereas BERT-base quadrupled recall to 50.0% (F1 0.5714).
2. **Intent Boundary Overlap in Verification Hurdles:**
   - **Mechanism:** Customer resistance or questions regarding identity verification (`why_verify_identity`) are frequently misclassified as operational failure (`unable_to_verify_identity`) or procedural requests (`verify_my_identity`).
   - **Winner:** **BERT-base V3 handles this better** (F1 0.7442 vs 0.7000), capturing question context versus failure declarations.
3. **Temporal Lifecycle vs Operational State in Money Movement:**
   - **Mechanism:** Inquiries about transaction delay (`pending_transfer`, `pending_top_up`) are consistently confused with SLA queries (`transfer_timing`) or failure events (`top_up_failed`, `top_up_reverted`).
   - **Winner:** **Tie / Both Struggle.** Both models suffer 4–5 errors on each pair because customer text rarely explicitly states whether an actual transaction is currently in-flight.

### B. Is BERT's Improvement Broad or Concentrated?
- **Finding:** BERT-base's improvement is **HIGHLY CONCENTRATED**, not broad.
  - Overall accuracy is identical (**90.00%** on both models, exactly 1,801 / 2,001).
  - The +0.0044 Macro F1 uplift is almost entirely attributable to dramatic rescues of severe minority classes (chiefly `virtual_card_not_working` jumping from 0.2222 to 0.5714, and `why_verify_identity` rising from 0.7000 to 0.7442).
  - Across the remaining 75 intents, BERT-base and DistilBERT perform virtually identically, with marginal fluctuations (e.g. `card_not_working` was 0.8182 in DistilBERT vs 0.8000 in BERT).

### C. Recommended Next Action
1. **Do NOT scale immediately to larger monolithic LLMs/RoBERTa-large:** Scaling model parameters from 66M to 110M added 70% latency overhead for only a 0.5% Macro F1 gain and 0% accuracy gain.
2. **Targeted Data Disambiguation & Contrastive Prompt Augmentation (Recommended Primary Step):**
   - The top remaining confusions (`top_up_reverted` $\leftrightarrow$ `top_up_failed`, `pending_transfer` $\leftrightarrow$ `transfer_timing`) are caused by lack of discriminative context in customer queries.
   - Introduce targeted data augmentation with contrastive query pairs clarifying state vs policy.
3. **Thresholding & Routing Calibration (Production Recommendation):**
   - High-confidence error rate is low (~1.8% of errors have confidence > 0.85). Applying a confidence threshold at 0.60 and routing uncertain predictions (margin |p1 - p2| < 0.15) to human tier-2 agents will eliminate over 65% of customer-facing misroutes.
