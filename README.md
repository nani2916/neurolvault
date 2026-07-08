# NeuralVault

A loan risk system that decides if someone qualifies for a loan, explains the decision in plain terms, and puts it all inside Salesforce for customers and bank staff to use.

I built this to learn how AI models actually connect to a real business tool like Salesforce — not just sit in a notebook.

---

## The problem I wanted to solve

Banks get thousands of loan applications. Reviewing them by hand is slow, and when someone gets rejected they often don't know why. Rules now say lenders have to be able to explain their decisions.

So I built something that does three things:

- A customer types in their details and immediately sees if they'd be approved, and the reasons if they're not.
- A bank employee can ask questions like "which customers are high risk?" and get a straight answer.
- Behind the scenes, small AI programs scan all the applications and write up risk and compliance reports.

---

## What you can actually see and use

There are two screens inside Salesforce:

**Customer screen** — you fill in loan amount, income, credit score, etc., hit a button, and get a decision with the reasons behind it. This runs on an XGBoost model, and SHAP is what pulls out the reasons.

**Bank staff screen** — a chat box where you ask questions in normal English. It looks through the loan records and answers you clearly. This uses RAG to find the right records and an LLM to write the answer.

---

## How it's put together

```
Loan data (about 148,000 records, cleaned up)
        |
        |- XGBoost          predicts default risk
        |- SHAP             gives the reasons
        |- Attention model  scores risk
        |- Isolation Forest catches unusual applications
        |- Fairlearn        checks the model isn't biased
        |- RAG (FAISS)      searches the records by meaning
                |
          LLM (Llama 3)  turns findings into plain answers
                |
          FastAPI  serves it all
                |
          Salesforce  customer screen + staff screen
```

The AI runs on my machine. Salesforce reaches it over a secure connection. For a real deployment it would live on a server instead.

---

## The pieces inside

- **Default prediction (XGBoost)** — trained on 148,000 records, 94% accurate. I had to remove a few columns that were secretly giving away the answer, which is why the numbers are honest.
- **Explanations (SHAP)** — this is what tells a rejected customer *why*.
- **Risk scoring (Attention model)** — I first tried an LSTM here, but the data wasn't time-based so it didn't work. An attention model fit the data much better.
- **Fraud detection (Isolation Forest)** — flags roughly 1 in 20 applications as unusual.
- **Bias check (Fairlearn)** — makes sure the model treats genders and regions fairly.
- **Chat answers (RAG + LLM)** — finds the relevant records and explains them in plain language.
- **Four small AI agents** — one scans for risk, one writes reports, one checks compliance, one powers the chat.

---

## A note on privacy

This mattered to me while building it.

The data I used doesn't contain names, SSNs, or ID numbers — it only has loan IDs and financial details. So the AI never sees anyone's identity.

If this were a real bank system with actual personal data, I'd tokenize the sensitive fields (swap the real SSN for a code before the AI ever sees it), encrypt them in the database, and use Salesforce's built-in tools to strip out personal info before anything reaches an LLM. The point is: no customer's private data should ever leave the bank's own systems.

API keys and large data files are kept out of the public code.

---

## Tools I used

Python, scikit-learn, XGBoost, TensorFlow, SHAP, Fairlearn, FAISS, LangChain, Groq (Llama 3), FastAPI, and Salesforce (Apex + Lightning Web Components).

---

## Folder layout

```
neurolvault/
├── data_pipeline/     cleaning the data
├── ml_engine/models/  the AI models
├── rag_system/        search + language model
├── agents/            the four AI agents
├── api/               the backend
└── salesforce/lwc/    the two screens
```

---

## Results

| What | Score |
|---|---|
| XGBoost accuracy | 94% |
| XGBoost ROC-AUC | 0.986 |
| Attention model ROC-AUC | 0.925 |
| Bias check (gender, region) | Passed |

---

## What I took away from it

The models were only half the work. The harder and more useful part was connecting them to Salesforce properly — getting the data through securely, showing it in a way a normal person can act on, and thinking about what would break or leak in the real world. That's the part I'm most glad I did.

---

**Rathna Sai Teja Panguluri**
