import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CLEANED_DATA_PATH = DATA_DIR / "Loan_Default_Cleaned.csv"


def load_documents():
    """
    Convert loan dataset rows into text documents for RAG.
    Each row becomes one searchable document.
    """
    logger.info("Loading loan data as documents...")
    df = pd.read_csv(CLEANED_DATA_PATH)

    # Use first 5000 rows for vector store
    # Full 148K would take too long to embed
    df = df.head(5000)

    documents = []
    for idx, row in df.iterrows():
        doc = f"""
Loan Application ID: {row.get('ID', idx)}
Year: {row.get('year', 'N/A')}
Gender: {row.get('Gender', 'N/A')}
Age Group: {row.get('age', 'N/A')}
Region: {row.get('Region', 'N/A')}
Loan Amount: ${row.get('loan_amount', 'N/A'):,.0f}
Rate of Interest: {row.get('rate_of_interest', 'N/A')}%
Property Value: ${row.get('property_value', 'N/A'):,.0f}
Annual Income: ${row.get('income', 'N/A'):,.0f}
Credit Score: {row.get('Credit_Score', 'N/A')}
Debt-to-Income Ratio: {row.get('dtir1', 'N/A')}%
Loan Term: {row.get('term', 'N/A')} months
Loan Status: {'DEFAULT' if row.get('Status', 0) == 1 else 'NO DEFAULT'}
        """.strip()

        documents.append({
            'id': idx,
            'text': doc,
            'metadata': {
                'loan_id': row.get('ID', idx),
                'status': row.get('Status', 0),
                'region': row.get('Region', 'N/A'),
                'gender': row.get('Gender', 'N/A')
            }
        })

    logger.info(f"Created {len(documents)} documents")
    return documents


if __name__ == "__main__":
    docs = load_documents()
    print(docs[0]['text']) 
