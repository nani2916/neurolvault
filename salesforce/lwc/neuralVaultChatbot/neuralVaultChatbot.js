import { LightningElement, track } from 'lwc';

export default class NeuralVaultChatbot extends LightningElement {

    @track messages = [];
    @track question = '';
    @track isLoading = false;

    API_URL = 'https://sprout-tweed-erasable.ngrok-free.dev/api/chatbot/ask';

    connectedCallback() {
        this.messages = [{
            id: 1,
            text: 'Hi — I can help with loan risk, customer profiles, and portfolio analysis. What would you like to know?',
            isAI: true,
            rowClass: 'nv-row nv-row-ai',
            bubbleClass: 'nv-bubble nv-bubble-ai'
        }];
    }

    handleInput(event) {
        this.question = event.target.value;
        // Auto-expand textarea
        const el = event.target;
        el.style.height = 'auto';
        el.style.height = Math.min(el.scrollHeight, 140) + 'px';
    }

    handleKeyDown(event) {
        // Enter sends, Shift+Enter makes a new line
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendMessage();
        }
    }

    async sendMessage() {
        if (!this.question || this.question.trim() === '') {
            return;
        }

        const userQuestion = this.question.trim();

        this.messages = [...this.messages, {
            id: Date.now(),
            text: userQuestion,
            isAI: false,
            rowClass: 'nv-row nv-row-user',
            bubbleClass: 'nv-bubble nv-bubble-user'
        }];

        this.question = '';
        this.isLoading = true;

        // Reset textarea height
        const input = this.refs.inputBox;
        if (input) {
            input.value = '';
            input.style.height = 'auto';
        }

        this.scrollToBottom();

        try {
            const response = await fetch(this.API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'ngrok-skip-browser-warning': 'true'
                },
                body: JSON.stringify({
                    question: userQuestion
                })
            });

            const data = await response.json();

            this.messages = [...this.messages, {
                id: Date.now() + 1,
                text: data.answer,
                isAI: true,
                rowClass: 'nv-row nv-row-ai',
                bubbleClass: 'nv-bubble nv-bubble-ai'
            }];

        } catch (error) {
            this.messages = [...this.messages, {
                id: Date.now() + 1,
                text: 'I could not reach the AI service. Check that the API is running and try again.',
                isAI: true,
                rowClass: 'nv-row nv-row-ai',
                bubbleClass: 'nv-bubble nv-bubble-ai'
            }];
        } finally {
            this.isLoading = false;
            this.scrollToBottom();
        }
    }

    scrollToBottom() {
        // eslint-disable-next-line @lwc/lwc/no-async-operation
        setTimeout(() => {
            const chat = this.refs.chatArea;
            if (chat) {
                chat.scrollTop = chat.scrollHeight;
            }
        }, 50);
    }
}