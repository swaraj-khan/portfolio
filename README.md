# Swaraj Khan's Portfolio

|  [My Resume](Swaraj Khan Resume.pdf){:target="_blank"} |  [YouTube Channel](https://www.youtube.com/@LumberjackStuds) | [Project Blogs](https://swarajkhan.quarto.pub/testing-blog/) |

## About Me

Computer Science graduate with hands-on expertise in building agentic LLM systems, real-time data pipelines, and AI-powered automation. Experienced in deploying production-grade multi-agent architectures (CrewAI, SmolDev) and working across FastAPI, Redis, GNNs, and Supabase. Passionate about applied ML, graph-based reasoning, and scalable AI infrastructure.

## Publications


**From Nodes to Notables: A Graph Framework to Detect Emerging Influencers**  
*2025*  
Accepted for Publication – IEEE Conference  

- Co-authored a hybrid influencer detection system using graph theory and GNNs, validated on 10,000+ social profiles.  
- Achieved 93.7% precision in identifying top 5% emerging influencers via eigenvector, closeness, and PageRank centralities.  
- Integrated behavioral metrics (engagement >25%, 2,500+ follower growth) and Louvain-based community detection for 3.2× improvement in targeting precision.  
- Improved influencer prediction accuracy by 28.4% over baseline using GNN-based modeling.


**Democratizing Machine Learning: A KNN-Guided Adaptive AutoML Framework**  
*2025*  
Accepted for Publication - IEEE Conference

- Co-authored research introducing a novel AutoML pipeline that automates preprocessing, model selection, and hyperparameter tuning for structured and time-series data.
- Developed a K-Nearest Neighbors approach for neural architecture determination that achieved 91.19% average accuracy across ten CSV datasets and 63.66% loss reduction across time-series datasets.
- Created a flexible framework supporting multiple data types including CSV, time-series, and image datasets with specialized processing pipelines.

**Smart Surveillance: AI-Driven Threat Detection and Women Safety Enhancement**  
*2025*  
Accepted for Publication - IEEE Conference

- Collaborated on an innovative surveillance system that leverages Raspberry Pi 5 for edge computing, enabling real-time threat detection and response.
- Integrated multiple detection models including face emotion recognition, weapon detection, violence detection, and behavior analysis to create a comprehensive security solution.
- Developed the alert management system using Telegram chatbot for instant notification, allowing rapid response to potential security threats.
- Contributed to system evaluation achieving over 90% accuracy in violence detection and 86% accuracy in weapon detection across various test scenarios.

**Automated PDF Q and A Chatbot: Harnessing AI for Efficient Information Retrieval**  
*2024*  
[IRF International Conference](https://digitalxplore.org/proceeding.php?pid=2620)  
Pune, India

- Developed an AI-driven PDF-based Q&A chatbot utilizing text extraction, chunking, and cosine similarity matching for accurate information retrieval without relying on generative AI.
- Implemented advanced data sanitization techniques to prevent XSS attacks, enhancing security while maintaining sub-second response times.
- Created comprehensive logging mechanisms to track interactions and queries, improving system reliability and user experience.
- Presented research findings demonstrating how efficient document parsing and vector representation can outperform conventional NLP techniques for specific Q&A applications.


## Experience


### Chatbot Development | Intern @ Nokia | Hybrid Bangalore | Mar '24 - Jul '24
- Engineered a production-grade AI-powered chatbot for Nokia's ticketing and testing teams, automating log issue resolution and reducing manual workload by 40% while improving operational efficiency.
- Enhanced chatbot query accuracy by 25% through implementation of advanced NLP techniques and BGE3-Large vector embeddings, optimizing response relevance and achieving a 92% user satisfaction rate.
- Integrated real-time database retrieval with optimized caching mechanisms, cutting response times by 30% and improving scalability for enterprise-level deployment.
- Selected to present the project at Nokia Bangalore University Connect (NBUC) program, a prestigious event attended by academic partners and senior technical leadership, receiving recognition for innovative approach to automated support systems.

### Machine Learning | Intern @ Dexian | In-Office Bangalore | Jul '23 - Aug '23
- Designed and implemented a high-accuracy heart stroke prediction system using ensemble methods including Random Forest (94.5%), Gradient Boosting (94.1%), and Logistic Regression (75.4%), enabling early risk assessment for preventative care.
- Optimized data preprocessing pipeline with advanced feature engineering techniques, improving overall model performance by 10% while reducing computational overhead by 15%.
- Developed interactive visualization dashboards using Matplotlib and Seaborn to communicate critical insights to medical stakeholders, facilitating data-driven decision making for patient interventions.

## Projects

### finanalysis: Comprehensive Financial Analysis Python Package (In Development)
- [GitHub Repo](https://github.com/swaraj-khan/finanalysis)
- Currently developing a pure Python PyPI library designed to offer 30+ financial metrics for stock market technical analysis, enabling traders and financial analysts to make data-driven investment decisions.
- Implementing four major analysis categories: candlestick pattern recognition (range, body size, shadow ratios), price action indicators (momentum, acceleration, trend strength), swing metrics (duration, magnitude, pivot points), and options analysis (put-call ratios, implied volatility, open interest).
- Designing a clean, intuitive API focused on performance optimization with vectorized operations for handling large datasets of historical market data with minimal computational overhead.
- Creating comprehensive documentation with practical examples demonstrating how traders can integrate the package into their existing analysis workflows, complete with visualization capabilities.

### World Port Priority Score Predictor
- [GitHub Repo](https://github.com/swaraj-khan/World-Port-Priority-Score-Predictor)
- Developed an interactive web application using Streamlit that analyzes and visualizes global port data, allowing users to evaluate port suitability based on physical characteristics.
- Implemented a machine learning model using joblib to predict priority scores for ports based on critical maritime parameters (overhead limit and tide range).
- Created dynamic geospatial visualizations with GeoPandas, enabling users to filter and display port locations by country on an interactive map interface.
- Designed a robust priority scoring algorithm that evaluates port accessibility factors, providing maritime logistics companies with data-driven insights for route planning and vessel selection.

### RAG Chatbot for Constitution of India
- [GitHub Repo](https://github.com/swaraj-khan/Avatar-James-Cameron-RAG-Chatbot)
- Implemented a RAG chatbot using Gemini 1.5 Flash and Langflow, integrating advanced retrieval-augmented generation capabilities.
- Utilized Chroma DB for efficient and scalable vector storage and retrieval of data.
- Applied chunking and text splitting techniques for optimized data processing and context-aware response generation.
- Engineered prompts and employed AI embeddings to enhance the chatbot's accuracy up to 95% and relevance in answering queries about Constitution of India.

### Auto ML Pipeline
- [GitHub Repo](https://github.com/swaraj-khan/AutoML-Data-Pipeline)
- Created a comprehensive AutoML pipeline that automates machine learning tasks including image segmentation, LSTM prediction, and CSV data analysis.
- Simplified the process of implementing machine learning models for various tasks, providing a seamless experience for researchers and practitioners.

### Autonomous Driving - Car Detection
- [GitHub Repo](https://github.com/swaraj-khan/Deep-Neural-Netwroks/blob/main/Autonomous_driving_application_Car_detection.ipynb)
- Implemented a state-of-the-art YOLO (You Only Look Once) object detection system for autonomous vehicles, achieving 89% accuracy in identifying cars, traffic lights, and other road objects under various lighting conditions.
- Engineered critical algorithms including non-max suppression and intersection over union (IoU) calculation to eliminate redundant detections and improve localization precision.
- Optimized tensor operations using TensorFlow to process the model's 19×19×5×85 dimensional output volume for real-time detection capabilities.
- Developed custom filtering algorithms to extract meaningful predictions from complex neural network outputs, enabling accurate bounding box generation around detected objects.

### Binary Image Classification with CNN
- [GitHub Repo](https://github.com/swaraj-khan/Deep-Neural-Netwroks/blob/main/2.%20Binary_Classification.pdf)
- Developed a robust image classification system using Convolutional Neural Networks (CNN) to distinguish between cat and dog images with 70% validation accuracy.
- Implemented a multi-layer architecture with two convolutional layers (32 and 64 filters), max pooling, and dense connections using TensorFlow and Keras.
- Engineered efficient data preprocessing pipeline for normalizing 100×100×3 pixel images and optimizing model training on 2,000+ samples.
- Achieved consistent performance improvement across training epochs from 55% to 90% accuracy through hyperparameter tuning and architecture optimization.

### Emojify
- [GitHub Repo](https://github.com/swaraj-khan/Deep-Neural-Netwroks/blob/main/Emoji_v3a.ipynb)
- Created an embedding layer in Keras with pre-trained word vectors.
- Explained the advantages and disadvantages of the GloVe algorithm.
- Built a sentiment classifier using word embeddings.
- Built and trained a more sophisticated classifier using an LSTM.
- Achieved an accuracy of 87% on the test set.

## Achievements

**ETL Hackathon (2025)**
- Secured 1st Place out of 300 students at Dayananda Sagar University.
- Developed a data pipeline to extract cricket player statistics from Cricbuzz, transform the data for user comparison, and visualize results through interactive bar and pie charts.

**Discord Bot Creation (2025)**
- Engineered a Discord bot simulating Bitcoin mining, where users solve math problems ranging from simple to intermediate difficulty to earn virtual bitcoins.
- Enhanced user engagement through gamification, creating an educational tool that teaches both mathematical concepts and basic cryptocurrency principles.

## Technical Skills

| Category                   | Skills                                                                                   |
|----------------------------|------------------------------------------------------------------------------------------|
| AI/ML                      | LLM (Anthropic, Hugging Face), RAG Systems, Agentic Framework, PyTorch, Pandas, NumPy   |
| Languages                  | Python, SQL                                                                             |
| Web Technologies           | Chainlit, Streamlit, Web Scraping, Langchain, LlamaIndex                                |
| Developer Tools            | Docker, Git, VS Code, GitHub                                                            |
| Databases & Infrastructure | Redis, PostgreSQL, Grafana, Logfire, Ray, SQS                                           |

## Certificates

- [Deep Neural Networks with PyTorch](https://www.coursera.org/account/accomplishments/verify/3QRQR3ADGLSY) - Coursera
- [Deep Learning Specialization](https://www.coursera.org/account/accomplishments/specialization/LBZ99YFBZLDH) by Andrew Ng (5 Modules) - Coursera

## Contact

- **Name:** P. Swaraj Khan
- **Phone:** +91 8618893815
- **Email:** swarajkhan2003@gmail.com
- **LinkedIn:** [Swaraj Khan](https://www.linkedin.com/in/swaraj-khan/)
- **GitHub:** [swaraj-khan](https://github.com/swaraj-khan)
- **LeetCode:** [swaraj-khan](https://leetcode.com/swaraj-khan/)
- **Hugging Face:** [SicarioOtsutsuki](https://huggingface.co/SicarioOtsutsuki)
