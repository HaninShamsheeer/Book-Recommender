# Dataset

A kaggle dataset with 7k books by Dylan Castillo was used to build this semantic book recommender.

(link: <url>https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata</url>) 

The dataset has columns such as isbn13, isbn10, title, subtitle, authors, categories, thumbnail, description, published_year, average_rating, num_pages and ratings_count

# Data Exploration

**Data Exploration Process:**
* Missing values of all the columns were analyzed using a heatmap in order to check whether there were any patterns on why such values were missing.
* Two new columns: missing_description and age_of_the book were created
- Correlation between num_pages, age_of_the book, missing_description and average rating were observed (no significant correlation)
- Books that had missing description, pages, average rating and published year were extracted (about 4% of the dataset --> removed)
- Descriptions that were not useful were removed (length between 1-24 were too short to be useful). There were 5197 books with description longer than 24 words
- Title and subtitle was merged into one column, if subtitle existed
- New column "tagged_description" was created that combined isbn13 and description (in order to make a vector database)
- Columns that we do not need such as "subtitle", "missing_description", "age_of_book", "words_in_description" were dropped.

**Data Exploration Summary:**

Since there existed no patterns between books with missing values, they were removed. Descriptions shorter than 24 words were removed. New column "tagged_description" was created and columns not in use were dropped.

**Libraries used:** pandas, matplotlib, seaborn, numpy


# Vector Search

**Vector Search Process** 

* Tagged description file was turned into a text file
* The text file was then loaded into TextLoader, split into chunks separated by new line character and loaded into a document file
* Then a vector database was built using Chroma and OpenAI embeddings that suggests recommendations based on query
* Then, the recommendation isbn was matched to the isbn of the original books dataframe to retrieve the title of the book that was recommended

**Packages used:** langchain-community, langchain-text-splitters, langchain-openai, langchain-chroma 

# Text Classification

Used HuggingFace's zero-shot classifier in order to classify books with no simple_categories into one of the 4 categories.

**Text Classification Process**

* Made a simple_category column that maps top 10 categories into either fiction, nonfiction, children's fiction, children's non-fiction,
* There were 1454 books without a simple_category
* Used HuggingFace Zero-Shot Classifier to categories those books into simple categories based on their descriptions
* Tested the model with the actual fiction and non-fiction category vs. predicted category and tested the model. 77% of the predictions matched the actual categories
* Then, the model was used to predict those without a simple_category and a dataframe of predicted category and their isbns were made
* This was then merged to the existing dataframe and the predicted categories replaced simple categories with no values
* The predicted categories column was then dropped 

# Sentiment Analysis

Used HuggingFace transformer to use the fine tuned LLM model to assign a sentiment (out of 7) to a book based on its description

* Predicting the overall emotion of the description does not return the most accurate information
* Broke the description into chunks and ran the classifier through each sentence 
* Obtained the maximum emotion probability for each description broken into chunks
* Created a dataframe with the scores of all the emotions
* Merged into the original books dataframe

# Gradio Dashboard 

A dashboard was created for the books with an option to enter a query/ description, select a categpry, and select an emotional tone. This would generate 16 book recommendations with thumbnails (generic not found cover if no thumbnail)


