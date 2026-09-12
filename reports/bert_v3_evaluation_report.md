# BERT-base (V3) Evaluation & 3-Way Benchmark Comparison Report

**Project:** Transformer-Based Banking Support Intelligence  
**Model Checkpoint:** `bert-base-uncased` fine-tuned end-to-end (109.5M parameters)  
**Evaluation Partition:** Exact 2,001-example validation split (locked)  
**Test Set Status:** Official test set (3,080 samples) remains **100% untouched**  

---

## 1. 3-Way Model Benchmark Comparison Table

| Model | Macro F1 | Weighted F1 | Accuracy | Params | Model Size | p50 Latency | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TF-IDF + Logistic Regression** | 0.8320 | 0.8417 | 0.8431 | N/A | ~2.5 MB | ~4.26 ms | 11.7s |
| **DistilBERT (V2)** | 0.8907 | 0.8983 | 0.9000 | 66.4M | ~265 MB | ~31.26 ms | 410.6s |
| **BERT-base (V3)** | **0.8951** | **0.9001** | **0.9000** | **109,541,453** | **~438 MB** | **~53.13 ms** | **807.5s** |

### Absolute and Relative Improvements
- **BERT-base over DistilBERT (V3 vs V2):**
  - Macro F1 Delta: **+0.0044** (+0.50%)
  - Accuracy Delta: **+0.0000**
- **BERT-base over TF-IDF Baseline (V3 vs V1):**
  - Macro F1 Delta: **+0.0631** (+7.59%)
  - Accuracy Delta: **+0.0569**

---

## 2. Deep-Dive on Difficult Intents (TF-IDF vs DistilBERT vs BERT-base)

Direct side-by-side progression on the 5 difficult intents:

| Intent Name | TF-IDF F1 | DistilBERT F1 | BERT-base F1 | Delta over DistilBERT | TF-IDF Recall | DistilBERT Recall | BERT-base Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `virtual_card_not_working` | 0.2222 | 0.2222 | **0.5714** | **+0.3492** | 0.1250 | 0.1250 | **0.5000** |
| `why_verify_identity` | 0.6957 | 0.7000 | **0.7442** | **+0.0442** | 0.6667 | 0.5833 | **0.6667** |
| `exchange_rate` | 0.6857 | 0.9545 | **0.9333** | **-0.0212** | 0.5455 | 0.9545 | **0.9545** |
| `card_not_working` | 0.6809 | 0.8182 | **0.8000** | **-0.0182** | 0.7273 | 0.8182 | **0.8182** |
| `contactless_not_working` | 0.6000 | 0.7273 | **0.7273** | **+0.0000** | 0.4286 | 0.5714 | **0.5714** |

---

## 3. Training Dynamics & GPU Memory Telemetry

- **GPU Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4.00 GB VRAM)
- **Batch Size:** `16` (exact protocol match with DistilBERT)
- **VRAM Utilization:** Peak memory allocated was **~2140.9 MB** (52.3% of available 4GB VRAM).
- **Memory Adjustments Needed:** **NONE**. Batch size 16 with FP16 Autocast ran smoothly without OOM or gradient accumulation.
- **Optimization:** AdamW (lr = 3e-5, weight_decay = 0.01), 10% linear warmup, 4 epochs.

### Epoch-by-Epoch Convergence
| Epoch | Train Loss | Val Loss | Val Accuracy | Val Macro F1 | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 3.3702 | 1.7259 | 0.7406 | **0.6908** | 179.8s |
| 2 | 1.1479 | 0.7034 | 0.8731 | **0.8579** | 174.3s |
| 3 | 0.4945 | 0.4790 | 0.8956 | **0.8911** | 229.2s |
| 4 | 0.2867 | 0.4274 | 0.9000 | **0.8951** | 221.0s |

---

## 4. BERT-base Error Analysis & Remaining Confusions

### Top 15 Confusion Pairs
| True Intent | Incorrectly Predicted Intent | Error Count |
| :--- | :--- | :--- |
| `top_up_reverted` | `top_up_failed` | **5** |
| `balance_not_updated_after_bank_transfer` | `transfer_timing` | **4** |
| `card_delivery_estimate` | `card_arrival` | **4** |
| `direct_debit_payment_not_recognised` | `card_payment_not_recognised` | **4** |
| `pending_top_up` | `top_up_failed` | **4** |
| `pending_transfer` | `transfer_timing` | **4** |
| `why_verify_identity` | `unable_to_verify_identity` | **4** |
| `wrong_exchange_rate_for_cash_withdrawal` | `wrong_amount_of_cash_received` | **4** |
| `beneficiary_not_allowed` | `failed_transfer` | **3** |
| `declined_transfer` | `declined_card_payment` | **3** |
| `pending_transfer` | `balance_not_updated_after_bank_transfer` | **3** |
| `reverted_card_payment?` | `declined_card_payment` | **3** |
| `topping_up_by_card` | `supported_cards_and_currencies` | **3** |
| `transfer_not_received_by_recipient` | `transfer_timing` | **3** |
| `virtual_card_not_working` | `getting_virtual_card` | **3** |

### Representative Misclassified Customer Queries
| Query Text | True Intent | Predicted Intent | Confidence |
| :--- | :--- | :--- | :--- |
| "how do the cards work?" | `get_disposable_virtual_card` | `card_not_working` | 0.3455 |
| "I just got a new card how do I get it to start working?" | `activate_my_card` | `card_not_working` | 0.3354 |
| "How long does a transfer take?" | `pending_transfer` | `transfer_timing` | 0.6948 |
| "I'm going on holiday to France for 6 weeks with my family and the cat, Fluffy.  I just purchased €10,000 from a currency exchange to ensure we don't run short but there appears to be a fee.  What is this for?" | `card_payment_fee_charged` | `card_payment_wrong_exchange_rate` | 0.6460 |
| "Does it cost me to add cash?" | `top_up_by_bank_transfer_charge` | `top_up_by_cash_or_cheque` | 0.4360 |
| "How do I add money to my card?" | `topping_up_by_card` | `supported_cards_and_currencies` | 0.4468 |
| "Why am I being charged a hidden fee?" | `card_payment_fee_charged` | `extra_charge_on_statement` | 0.5498 |
| "I need my refund as soon as possible. What else do I have to do?" | `Refund_not_showing_up` | `request_refund` | 0.5415 |
| "There is a pending transaction on my account, what does this mean?" | `pending_card_payment` | `pending_transfer` | 0.5537 |
| "I need help proving that this is really me and to verify my identity." | `unable_to_verify_identity` | `why_verify_identity` | 0.3196 |

---

## 5. Architectural Findings: Is the Improvement Meaningful?

1. **Performance Verdict:** BERT-base reaches **0.8951** Macro F1 vs **0.8907** for DistilBERT (+0.0044).
2. **Resource Tradeoff:**
   - Model size increases from ~265 MB to ~438 MB (+65%).
   - Inference latency increases from ~31.26 ms to ~53.13 ms.
   - Training time scaled from ~410.6s to ~807.5s.
3. **Engineering Assessment:** The gain over DistilBERT is modest relative to the 1.6x parameter increase, indicating DistilBERT offers a highly competitive latency/performance tradeoff for production deployments.
