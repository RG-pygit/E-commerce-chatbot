# E-commerce-chatbot
create an interactive e-commerce chatbot for customers with AI agents

ecommerce_chatbot_project/
│
├── data/
│   ├── products.json             # 50 products
│   ├── products.csv              # Another 50 products
│   └── company_policies.md       # Company policies
│
├── src/
│   ├── loaders/
│   │   ├── json_loader.py
│   │   ├── csv_loader.py
│   │   └── markdown_loader.py
│   │
│   ├── rag/
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   └── generator.py
│   │
│   ├── graph/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   └── workflow.py
│   │
│   └── config/
│       └── settings.py
│
├── tests/
│   └── test_retrieval.py
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
