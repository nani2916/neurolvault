import { LightningElement, track } from 'lwc';

export default class LoanDecisionCard extends LightningElement {

    @track loanAmount = '';
    @track income = '';
    @track creditScore = '';
    @track rate = '';
    @track dti = '';
    @track term = '';

    @track isLoading = false;
    @track showResult = false;

    @track probability = 0;
    @track riskLevel = '';
    @track reasons = [];

    API_URL = 'https://sprout-tweed-erasable.ngrok-free.dev/api/predictions/predict';

    handleChange(event) {
        const field = event.target.dataset.field;
        this[field] = event.target.value;
    }

    get probabilityPct() {
        return Math.round(this.probability * 100) + '%';
    }

    get decisionText() {
        return this.riskLevel === 'LOW' ? 'Approved' :
               this.riskLevel === 'MEDIUM' ? 'Needs Review' : 'Not Approved';
    }

    get decisionIcon() {
        return this.riskLevel === 'LOW' ? '✓' :
               this.riskLevel === 'MEDIUM' ? '!' : '✕';
    }

    get decisionClass() {
        return this.riskLevel === 'HIGH' ? 'ld-decision-rejected' : 'ld-decision-approved';
    }

    async checkDecision() {
        this.isLoading = true;

        try {
            const response = await fetch(this.API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'ngrok-skip-browser-warning': 'true'
                },
                body: JSON.stringify({
                    loan_amount: parseFloat(this.loanAmount) || 0,
                    income: parseFloat(this.income) || 0,
                    Credit_Score: parseInt(this.creditScore, 10) || 0,
                    rate_of_interest: parseFloat(this.rate) || 0,
                    dtir1: parseFloat(this.dti) || 0,
                    term: parseFloat(this.term) || 0,
                    property_value: (parseFloat(this.loanAmount) || 0) * 1.3
                })
            });

            const data = await response.json();

            this.probability = data.default_probability;
            this.riskLevel = data.risk_level;

            // Turn technical factors into readable reasons
            const factorMap = {
                'rate_of_interest': 'Interest rate affects repayment ability',
                'credit_type': 'Credit history and type',
                'property_value': 'Property value as collateral',
                'loan_amount': 'Requested loan amount',
                'income': 'Income level relative to loan',
                'dtir1': 'Debt-to-income ratio',
                'Credit_Score': 'Credit score'
            };

            this.reasons = (data.top_risk_factors || []).map(f =>
                factorMap[f] || f
            );

            if (this.reasons.length === 0) {
                this.reasons = ['Overall financial profile assessment'];
            }

            this.showResult = true;

        } catch (error) {
            this.riskLevel = 'HIGH';
            this.probability = 0;
            this.reasons = ['Could not reach the decision service. Please try again.'];
            this.showResult = true;
        } finally {
            this.isLoading = false;
        }
    }

    reset() {
        this.showResult = false;
        this.loanAmount = '';
        this.income = '';
        this.creditScore = '';
        this.rate = '';
        this.dti = '';
        this.term = '';
    }
}
