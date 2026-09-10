# Baseline Model Evaluation Report (V1: TF-IDF + Logistic Regression)

**Project:** Transformer-Based Banking Support Intelligence  
**Model Architecture:** `TfidfVectorizer(ngram_range=(1,2))` $\rightarrow$ `LogisticRegression(multinomial, C=1.0)`  
**Target Space:** 77-class fine-grained customer intent classification  
**Status:** Benchmark Established  

---

## 1. Executive Performance Summary

| Metric | Score | Benchmark Target Role |
| :--- | :--- | :--- |
| **Macro F1 (PRIMARY)** | **0.8320** (83.20%) | Primary optimization target across 77 imbalanced classes |
| **Weighted F1** | **0.8417** (84.17%) | Frequency-weighted intent accuracy |
| **Accuracy** | **0.8431** (84.31%) | Overall correct intent classification rate |
| **Training Duration** | **11.69 seconds** | Extremely fast CPU iteration baseline |
| **Vocabulary Size** | **8,901 features** | Unigrams + Bigrams with min_df=2 |

> [!NOTE]
> All metrics reported are evaluated strictly on the **Validation Split** (2,001 examples).  
> The **Official Test Split** (3,080 examples) remains **100% untouched**.

---

## 2. Model Architecture & Hyperparameter Rationale

### TF-IDF Vectorizer
| Parameter | Value | Design Rationale |
| :--- | :--- | :--- |
| `ngram_range` | `(1, 2)` | Captures critical multi-word banking expressions (e.g. *"cash machine"*, *"apple pay"*, *"wrong exchange rate"*, *"cancel transfer"*). |
| `min_df` | `2` | Prunes single-occurrence typos while retaining genuine financial domain terminology. |
| `max_df` | `0.95` | Drops terms appearing in >95% of documents to remove corpus-wide stopwords. |
| `sublinear_tf` | `True` | Applies logarithmic frequency scaling ($1 + \log(\text{tf})$) to dampen the impact of repeated words in long tickets. |
| `lowercase` | `True` | Standardizes case variations typical in user chat inquiries. |
| `max_features`| `None` | Preserves all 8,901 valid vocabulary terms without artificial feature truncation. |

### Logistic Regression Classifier
| Parameter | Value | Design Rationale |
| :--- | :--- | :--- |
| `solver` | `lbfgs` | Quasi-Newton optimization algorithm well-suited for high-dimensional sparse representations and multinomial cross-entropy. |
| `C` | `1.0` | Balanced $L_2$ regularization penalty preventing overfitting on 8,901 sparse n-gram dimensions. |
| `max_iter` | `1000` | Ample iteration ceiling ensuring full mathematical convergence across all 77 intent classes. |
| `random_state` | `42` | Explicit seed ensuring bitwise deterministic training across platforms. |

---

## 3. Best-Performing vs. Worst-Performing Intents

### Top 10 Best-Performing Intents (Highest F1)
Intents with unique, unambiguous keyword markers (e.g. card delivery, pin changing, direct debit):

| Intent Name | Precision | Recall | Macro F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| `apple_pay_or_google_pay` | 1.0000 | 1.0000 | **1.0000** | 25 |
| `passcode_forgotten` | 0.9545 | 1.0000 | **0.9767** | 21 |
| `visa_or_mastercard` | 0.9630 | 0.9630 | **0.9630** | 27 |
| `edit_personal_details` | 0.9583 | 0.9583 | **0.9583** | 24 |
| `age_limit` | 0.9545 | 0.9545 | **0.9545** | 22 |
| `cash_withdrawal_not_recognised` | 0.9394 | 0.9688 | **0.9538** | 32 |
| `transaction_charged_twice` | 0.9429 | 0.9429 | **0.9429** | 35 |
| `country_support` | 0.9600 | 0.9231 | **0.9412** | 26 |
| `getting_spare_card` | 1.0000 | 0.8846 | **0.9388** | 26 |
| `verify_top_up` | 0.9583 | 0.9200 | **0.9388** | 25 |

### Top 10 Worst-Performing Intents (Lowest F1)
Intents exhibiting high semantic overlap or sparse support:

| Intent Name | Precision | Recall | Macro F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| `virtual_card_not_working` | 1.0000 | 0.1250 | **0.2222** | 8 |
| `contactless_not_working` | 1.0000 | 0.4286 | **0.6000** | 7 |
| `card_not_working` | 0.6400 | 0.7273 | **0.6809** | 22 |
| `exchange_rate` | 0.9231 | 0.5455 | **0.6857** | 22 |
| `why_verify_identity` | 0.7273 | 0.6667 | **0.6957** | 24 |
| `unable_to_verify_identity` | 0.7368 | 0.7000 | **0.7179** | 20 |
| `get_disposable_virtual_card` | 0.7647 | 0.6842 | **0.7222** | 19 |
| `card_acceptance` | 1.0000 | 0.5833 | **0.7368** | 12 |
| `extra_charge_on_statement` | 0.7027 | 0.7879 | **0.7429** | 33 |
| `topping_up_by_card` | 0.9286 | 0.6190 | **0.7429** | 21 |

---

## 4. Top Misclassification Confusion Pairs

Analysis of off-diagonal errors reveals where n-gram bag-of-words representations struggle without contextual transformer self-attention:

| True Intent | Incorrectly Predicted Intent | Error Count |
| :--- | :--- | :--- |
| `exchange_rate` | `card_payment_wrong_exchange_rate` | **9** |
| `transfer_fee_charged` | `extra_charge_on_statement` | **6** |
| `virtual_card_not_working` | `getting_virtual_card` | **6** |
| `pin_blocked` | `get_physical_card` | **5** |
| `unable_to_verify_identity` | `why_verify_identity` | **5** |
| `direct_debit_payment_not_recognised` | `card_payment_not_recognised` | **4** |
| `pending_cash_withdrawal` | `declined_cash_withdrawal` | **4** |
| `wrong_exchange_rate_for_cash_withdrawal` | `wrong_amount_of_cash_received` | **4** |
| `beneficiary_not_allowed` | `failed_transfer` | **3** |
| `card_payment_wrong_exchange_rate` | `wrong_exchange_rate_for_cash_withdrawal` | **3** |
| `change_pin` | `pin_blocked` | **3** |
| `declined_card_payment` | `reverted_card_payment?` | **3** |
| `disposable_card_limits` | `get_disposable_virtual_card` | **3** |
| `exchange_via_app` | `exchange_charge` | **3** |
| `get_disposable_virtual_card` | `disposable_card_limits` | **3** |

### Key Confusion Clusters Identified:
1. **Cash Withdrawal Ambiguity:** Queries involving ATM issues frequently confuse `declined_cash_withdrawal` $\leftrightarrow$ `wrong_amount_of_cash_received` $\leftrightarrow$ `cash_withdrawal_not_recognised`. All share keywords like *"cash"*, *"atm"*, *"money"*, *"machine"*.
2. **Transfer Status Ambiguity:** Queries about transfers confuse `transfer_not_received_by_recipient` $\leftrightarrow$ `transfer_timing` $\leftrightarrow$ `pending_transfer` because customers use overlapping phrasing (*"when will my transfer arrive"*, *"has money been sent"*).
3. **Card Linking vs Virtual Card Creation:** `card_linking` $\leftrightarrow$ `getting_virtual_card` $\leftrightarrow$ `get_physical_card` share lexical roots (*"card"*, *"link"*, *"add"*, *"get"*).

*Implication for V2 Transformer:* Transformer attention heads will be able to capture syntactic dependency relationships (e.g. *subject/action/temporal modifiers*) that bag-of-ngrams cannot distinguish.

---

## 5. Artifacts and Visualization

- **Fitted Model Artifact:** `models/baseline_tfidf_logreg.joblib`
- **Confusion Matrix Heatmap:** `reports/baseline_confusion_matrix.png`
- **Detailed Confusion Pairs CSV:** `reports/baseline_top_confusions.csv`
- **Structured Experiment Record:** `experiments/baseline_v1_run.json`

---

## 6. Complete 77-Class Per-Class Metrics Table

| Class ID | Intent Name | Precision | Recall | F1 Score | Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | `activate_my_card` | 1.0000 | 0.8438 | 0.9153 | 32 |
| 1 | `age_limit` | 0.9545 | 0.9545 | 0.9545 | 22 |
| 2 | `apple_pay_or_google_pay` | 1.0000 | 1.0000 | 1.0000 | 25 |
| 3 | `atm_support` | 0.8824 | 0.8824 | 0.8824 | 17 |
| 4 | `automatic_top_up` | 1.0000 | 0.8400 | 0.9130 | 25 |
| 5 | `balance_not_updated_after_bank_transfer` | 0.7632 | 0.8529 | 0.8056 | 34 |
| 6 | `balance_not_updated_after_cheque_or_cash_deposit` | 0.8537 | 0.9722 | 0.9091 | 36 |
| 7 | `beneficiary_not_allowed` | 0.9231 | 0.7742 | 0.8421 | 31 |
| 8 | `cancel_transfer` | 0.9333 | 0.9032 | 0.9180 | 31 |
| 9 | `card_about_to_expire` | 0.8000 | 0.9231 | 0.8571 | 26 |
| 10 | `card_acceptance` | 1.0000 | 0.5833 | 0.7368 | 12 |
| 11 | `card_arrival` | 0.6905 | 0.9355 | 0.7945 | 31 |
| 12 | `card_delivery_estimate` | 0.9474 | 0.8182 | 0.8780 | 22 |
| 13 | `card_linking` | 0.7222 | 0.9286 | 0.8125 | 28 |
| 14 | `card_not_working` | 0.6400 | 0.7273 | 0.6809 | 22 |
| 15 | `card_payment_fee_charged` | 0.7692 | 0.7895 | 0.7792 | 38 |
| 16 | `card_payment_not_recognised` | 0.7632 | 0.8529 | 0.8056 | 34 |
| 17 | `card_payment_wrong_exchange_rate` | 0.7297 | 0.8182 | 0.7714 | 33 |
| 18 | `card_swallowed` | 1.0000 | 0.7500 | 0.8571 | 12 |
| 19 | `cash_withdrawal_charge` | 0.8205 | 0.8889 | 0.8533 | 36 |
| 20 | `cash_withdrawal_not_recognised` | 0.9394 | 0.9688 | 0.9538 | 32 |
| 21 | `change_pin` | 1.0000 | 0.8333 | 0.9091 | 24 |
| 22 | `compromised_card` | 0.8571 | 0.7059 | 0.7742 | 17 |
| 23 | `contactless_not_working` | 1.0000 | 0.4286 | 0.6000 | 7 |
| 24 | `country_support` | 0.9600 | 0.9231 | 0.9412 | 26 |
| 25 | `declined_card_payment` | 0.7742 | 0.7742 | 0.7742 | 31 |
| 26 | `declined_cash_withdrawal` | 0.7895 | 0.8571 | 0.8219 | 35 |
| 27 | `declined_transfer` | 0.8929 | 0.9259 | 0.9091 | 27 |
| 28 | `direct_debit_payment_not_recognised` | 0.9394 | 0.8378 | 0.8857 | 37 |
| 29 | `disposable_card_limits` | 0.7917 | 0.7917 | 0.7917 | 24 |
| 30 | `edit_personal_details` | 0.9583 | 0.9583 | 0.9583 | 24 |
| 31 | `exchange_charge` | 0.7333 | 0.9167 | 0.8148 | 24 |
| 32 | `exchange_rate` | 0.9231 | 0.5455 | 0.6857 | 22 |
| 33 | `exchange_via_app` | 0.9048 | 0.7917 | 0.8444 | 24 |
| 34 | `extra_charge_on_statement` | 0.7027 | 0.7879 | 0.7429 | 33 |
| 35 | `failed_transfer` | 0.7429 | 0.9630 | 0.8387 | 27 |
| 36 | `fiat_currency_support` | 0.8519 | 0.9200 | 0.8846 | 25 |
| 37 | `get_disposable_virtual_card` | 0.7647 | 0.6842 | 0.7222 | 19 |
| 38 | `get_physical_card` | 0.8000 | 0.9524 | 0.8696 | 21 |
| 39 | `getting_spare_card` | 1.0000 | 0.8846 | 0.9388 | 26 |
| 40 | `getting_virtual_card` | 0.6667 | 0.9000 | 0.7660 | 20 |
| 41 | `lost_or_stolen_card` | 0.7000 | 0.8750 | 0.7778 | 16 |
| 42 | `lost_or_stolen_phone` | 0.9565 | 0.9167 | 0.9362 | 24 |
| 43 | `order_physical_card` | 0.8095 | 0.7083 | 0.7556 | 24 |
| 44 | `passcode_forgotten` | 0.9545 | 1.0000 | 0.9767 | 21 |
| 45 | `pending_card_payment` | 0.9667 | 0.9062 | 0.9355 | 32 |
| 46 | `pending_cash_withdrawal` | 0.8750 | 0.7241 | 0.7925 | 29 |
| 47 | `pending_top_up` | 0.7576 | 0.8333 | 0.7937 | 30 |
| 48 | `pending_transfer` | 0.7419 | 0.7667 | 0.7541 | 30 |
| 49 | `pin_blocked` | 0.7727 | 0.7391 | 0.7556 | 23 |
| 50 | `receiving_money` | 0.9412 | 0.8421 | 0.8889 | 19 |
| 51 | `Refund_not_showing_up` | 0.9667 | 0.9062 | 0.9355 | 32 |
| 52 | `request_refund` | 0.8684 | 0.9706 | 0.9167 | 34 |
| 53 | `reverted_card_payment?` | 0.7368 | 0.8750 | 0.8000 | 32 |
| 54 | `supported_cards_and_currencies` | 0.8519 | 0.8846 | 0.8679 | 26 |
| 55 | `terminate_account` | 0.9500 | 0.8636 | 0.9048 | 22 |
| 56 | `top_up_by_bank_transfer_charge` | 0.9412 | 0.7273 | 0.8205 | 22 |
| 57 | `top_up_by_card_charge` | 0.7931 | 1.0000 | 0.8846 | 23 |
| 58 | `top_up_by_cash_or_cheque` | 0.8182 | 0.7826 | 0.8000 | 23 |
| 59 | `top_up_failed` | 0.8214 | 0.7931 | 0.8070 | 29 |
| 60 | `top_up_limits` | 0.8824 | 0.7895 | 0.8333 | 19 |
| 61 | `top_up_reverted` | 0.8276 | 0.8276 | 0.8276 | 29 |
| 62 | `topping_up_by_card` | 0.9286 | 0.6190 | 0.7429 | 21 |
| 63 | `transaction_charged_twice` | 0.9429 | 0.9429 | 0.9429 | 35 |
| 64 | `transfer_fee_charged` | 0.8571 | 0.7059 | 0.7742 | 34 |
| 65 | `transfer_into_account` | 0.9091 | 0.8696 | 0.8889 | 23 |
| 66 | `transfer_not_received_by_recipient` | 0.7576 | 0.7353 | 0.7463 | 34 |
| 67 | `transfer_timing` | 0.7931 | 0.8846 | 0.8364 | 26 |
| 68 | `unable_to_verify_identity` | 0.7368 | 0.7000 | 0.7179 | 20 |
| 69 | `verify_my_identity` | 0.8182 | 0.8571 | 0.8372 | 21 |
| 70 | `verify_source_of_funds` | 0.9524 | 0.8696 | 0.9091 | 23 |
| 71 | `verify_top_up` | 0.9583 | 0.9200 | 0.9388 | 25 |
| 72 | `virtual_card_not_working` | 1.0000 | 0.1250 | 0.2222 | 8 |
| 73 | `visa_or_mastercard` | 0.9630 | 0.9630 | 0.9630 | 27 |
| 74 | `why_verify_identity` | 0.7273 | 0.6667 | 0.6957 | 24 |
| 75 | `wrong_amount_of_cash_received` | 0.8718 | 0.9444 | 0.9067 | 36 |
| 76 | `wrong_exchange_rate_for_cash_withdrawal` | 0.8065 | 0.7576 | 0.7812 | 33 |
