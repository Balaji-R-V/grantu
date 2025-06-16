from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.utilities import SQLDatabase
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from dotenv import load_dotenv
import ast
import pickle
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "database": "grandu_db",
    "user": "root",
    "password": "root"
}

# FAISS index configuration
FAISS_INDEX_DIR = "faiss_index"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class SearchCriteria(BaseModel):
    """Search criteria extracted from user query"""
    expertise: Optional[List[str]] = Field(default_factory=list, description="Areas of expertise to search for")
    years_of_experience: Optional[int] = Field(None, description="Minimum years of experience required")
    organization: Optional[List[str]] = Field(default_factory=list, description="Organizations to search for")
    field_of_interest: Optional[List[str]] = Field(default_factory=list, description="Fields of interest to search for")
    requirements: Optional[List[str]] = Field(default_factory=list, description="Specific requirements to search for")

def extract_search_criteria(query: str) -> SearchCriteria:
    """Use LLM to extract structured search criteria from natural language query"""
    llm = ChatOpenAI(
    model="llama3-8b-8192",
    temperature=0.5,
    openai_api_key="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",
    openai_api_base="https://api.groq.com/openai/v1"
)
    
    parser = PydanticOutputParser(pydantic_object=SearchCriteria)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at extracting search criteria from natural language queries.
        Extract the following information:
        - Areas of expertise
        - Years of experience
        - Organizations
        - Fields of interest
        - Specific requirements
        
        {format_instructions}
        
        If a field is not mentioned in the query, leave it as None or empty list."""),
        ("user", "{query}")
    ])
    
    chain = prompt | llm | parser
    
    try:
        criteria = chain.invoke({
            "query": query,
            "format_instructions": parser.get_format_instructions()
        })
        return criteria
    except Exception as e:
        print(f"Error extracting search criteria: {e}")
        return SearchCriteria()

def calculate_perplexity(scores):
    """Calculate perplexity score from similarity scores"""
    if not scores:
        return 0.0
    # Convert scores to probabilities
    probs = np.array(scores) / sum(scores)
    # Calculate perplexity
    perplexity = math.exp(-np.sum(probs * np.log(probs + 1e-10)))
    return perplexity

def calculate_cosine_similarity(query_embedding, doc_embedding):
    """Calculate cosine similarity between query and document embeddings"""
    return cosine_similarity([query_embedding], [doc_embedding])[0][0]

def get_postgres_data():
    """Extract data from MySQL and convert to documents"""
    try:
        # Connect to MySQL
        db = SQLDatabase.from_uri(
            f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )
        
        # Query to fetch expert profiles
        query = """
        SELECT 
            user_id,
            first_name,
            last_name,
            expertise,
            years_of_experience,
            organization_detail,
            field_of_interest,
            requirements
        FROM grandu_user
        """
        results = db.run(query)
        
        # Convert results to documents with structured format
        documents = []
        
        # Parse the results string into a list of tuples
        try:
            # Remove any leading/trailing whitespace and convert to list of tuples
            results_list = ast.literal_eval(results.strip())
            
            for row in results_list:
                # Create a structured text representation
                text = f"""
                Expertise: {row[3]}
                Years of Experience: {row[4]}
                Organization: {row[5]}
                Field of Interest: {row[6]}
                Requirements: {row[7]}
                """
                
                # Create metadata for the document
                metadata = {
                    'user_id': row[0],
                    'first_name': row[1],
                    'last_name': row[2],
                    'expertise': row[3],
                    'years_of_experience': row[4],
                    'organization_detail': row[5],
                    'field_of_interest': row[6],
                    'requirements': row[7]
                }
                
                # Create a Document object with text and metadata
                doc = Document(
                    page_content=text,
                    metadata=metadata
                )
                documents.append(doc)
                
            print(f"Successfully processed {len(documents)} expert profiles")
            return documents
            
        except Exception as e:
            print(f"Error parsing results: {e}")
            print(f"Raw results: {results}")
            return []
            
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return []

def create_or_load_vector_store():
    """Create new vector store or load existing one"""
    # Create directory if it doesn't exist
    if not os.path.exists(FAISS_INDEX_DIR):
        os.makedirs(FAISS_INDEX_DIR)
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    # Check if we have a saved index
    index_path = os.path.join(FAISS_INDEX_DIR, "faiss_index")
    if os.path.exists(index_path):
        print("Loading existing FAISS index...")
        try:
            # Enable pickle deserialization for trusted local files
            vector_store = FAISS.load_local(
                index_path, 
                embeddings,
                allow_dangerous_deserialization=True  # Safe since we created the index
            )
            print("Successfully loaded existing index")
            return vector_store
        except Exception as e:
            print(f"Error loading index: {e}")
            print("Creating new index...")
    
    # If no existing index or error loading, create new one
    print("Creating new FAISS index...")
    documents = get_postgres_data()
    
    if not documents:
        print("No documents to create index")
        return None
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    # Split documents into chunks
    texts = text_splitter.split_documents(documents)
    
    # Create FAISS vector store
    vector_store = FAISS.from_documents(texts, embeddings)
    
    # Save the index
    vector_store.save_local(index_path)
    print(f"Saved FAISS index to {index_path}")
    
    return vector_store

def setup_retriever():
    """Setup the retrieval system"""
    # Create or load vector store
    vector_store = create_or_load_vector_store()
    
    if not vector_store:
        print("Failed to create or load vector store")
        return None
    
    # Create retriever with similarity scores
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5  # Return top 5 most similar documents
        }
    )
    
    return retriever

def calculate_metrics(results):
    """Calculate various metrics for the retrieval results"""
    if not results:
        return {
            'total_results': 0,
            'average_score': 0.0,
            'max_score': 0.0,
            'min_score': 0.0,
            'score_std': 0.0,
            'perplexity': 0.0,
            'average_cosine_similarity': 0.0,
            'score_distribution': {
                'high': 0,
                'medium': 0,
                'low': 0
            }
        }
        
    # Extract similarity scores
    scores = [result.get('similarity_score', 0) for result in results]
    cosine_scores = [result.get('cosine_similarity', 0) for result in results]
    
    metrics = {
        'total_results': len(results),
        'average_score': np.mean(scores),
        'max_score': max(scores),
        'min_score': min(scores),
        'score_std': np.std(scores),
        'perplexity': calculate_perplexity(scores),
        'average_cosine_similarity': np.mean(cosine_scores),
        'score_distribution': {
            'high': len([s for s in scores if s >= 0.8]),
            'medium': len([s for s in scores if 0.5 <= s < 0.8]),
            'low': len([s for s in scores if s < 0.5])
        }
    }
    
    return metrics

def filter_results_by_criteria(results, criteria: SearchCriteria):
    """Filter results based on extracted search criteria and separate into exact and recommended matches"""
    exact_matches = []
    recommended_matches = []
    
    for result in results:
        match_score = 0
        total_criteria = 0
        
        # Check years of experience
        if criteria.years_of_experience and result['years_of_experience']:
            total_criteria += 1
            try:
                if int(result['years_of_experience']) >= criteria.years_of_experience:
                    match_score += 1
            except ValueError:
                pass
        
        # Check expertise
        if criteria.expertise and result['expertise']:
            total_criteria += 1
            if any(exp.lower() in result['expertise'].lower() for exp in criteria.expertise):
                match_score += 1
        
        # Check organization
        if criteria.organization and result['organization']:
            total_criteria += 1
            if any(org.lower() in result['organization'].lower() for org in criteria.organization):
                match_score += 1
        
        # Check field of interest
        if criteria.field_of_interest and result['field_of_interest']:
            total_criteria += 1
            if any(field.lower() in result['field_of_interest'].lower() for field in criteria.field_of_interest):
                match_score += 1
        
        # Check requirements
        if criteria.requirements and result['requirements']:
            total_criteria += 1
            if any(req.lower() in result['requirements'].lower() for req in criteria.requirements):
                match_score += 1
        
        # Calculate match percentage
        match_percentage = (match_score / total_criteria) if total_criteria > 0 else 0
        result['match_percentage'] = match_percentage
        
        # Categorize as exact or recommended match
        if match_percentage == 1.0:  # Perfect match
            exact_matches.append(result)
        elif match_percentage >= 0.5:  # Partial match
            recommended_matches.append(result)
    
    # Sort recommended matches by match percentage
    recommended_matches.sort(key=lambda x: x['match_percentage'], reverse=True)
    
    return exact_matches, recommended_matches

def query_retriever(retriever, query):
    """Query the retriever system"""
    if retriever is None:
        return "Retriever system not properly initialized"
    
    try:
        # Extract search criteria using LLM
        print("\nExtracting search criteria...")
        criteria = extract_search_criteria(query)
        print(f"Extracted criteria: {criteria}")
        
        # Get relevant documents
        docs = retriever.invoke(query)
        
        # Get query embedding for cosine similarity
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL, model_kwargs={'device': 'cpu'})
        query_embedding = embeddings.embed_query(query)
        
        # Format the results
        results = []
        for doc in docs:
            # Get document embedding
            doc_embedding = embeddings.embed_query(doc.page_content)
            
            # Calculate similarity scores
            similarity_score = 0.8  # Default score
            cosine_sim = calculate_cosine_similarity(query_embedding, doc_embedding)
            
            result = {
                'expert': f"{doc.metadata['first_name']} {doc.metadata['last_name']}",
                'expertise': doc.metadata['expertise'],
                'years_of_experience': doc.metadata['years_of_experience'],
                'organization': doc.metadata['organization_detail'],
                'field_of_interest': doc.metadata['field_of_interest'],
                'requirements': doc.metadata['requirements'],
                'similarity_score': similarity_score,
                'cosine_similarity': cosine_sim
            }
            results.append(result)
        
        # Filter results based on extracted criteria
        exact_matches, recommended_matches = filter_results_by_criteria(results, criteria)
        
        # Calculate metrics for both exact and recommended matches
        exact_metrics = calculate_metrics(exact_matches)
        recommended_metrics = calculate_metrics(recommended_matches)
        
        return {
            'exact_matches': {
                'results': exact_matches,
                'metrics': exact_metrics
            },
            'recommended_matches': {
                'results': recommended_matches,
                'metrics': recommended_metrics
            },
            'search_criteria': criteria
        }
    except Exception as e:
        print(f"Error details: {str(e)}")
        return f"Error querying retriever system: {e}"

def print_retrieval_results(query, response):
    """Print formatted retrieval results with metrics"""
    print(f"\nQuery: {query}")
    
    if isinstance(response, str):
        print(f"Error: {response}")
        return
    
    # Print search criteria
    print("\nExtracted Search Criteria:")
    criteria = response['search_criteria']
    if criteria.expertise:
        print(f"Expertise: {', '.join(criteria.expertise)}")
    if criteria.years_of_experience:
        print(f"Years of Experience: {criteria.years_of_experience}+")
    if criteria.organization:
        print(f"Organizations: {', '.join(criteria.organization)}")
    if criteria.field_of_interest:
        print(f"Fields of Interest: {', '.join(criteria.field_of_interest)}")
    if criteria.requirements:
        print(f"Requirements: {', '.join(criteria.requirements)}")
    
    # Print exact matches
    exact_matches = response['exact_matches']
    exact_metrics = exact_matches['metrics']
    
    print("\n=== Exact Matches ===")
    print(f"Total Exact Matches: {exact_metrics['total_results']}")
    if exact_metrics['total_results'] > 0:
        print("\nExact Match Metrics:")
        print(f"Average Similarity Score: {exact_metrics['average_score']:.3f}")
        print(f"Average Cosine Similarity: {exact_metrics['average_cosine_similarity']:.3f}")
        print(f"Perplexity Score: {exact_metrics['perplexity']:.3f}")
        
        print("\nExact Match Experts:")
        for i, result in enumerate(exact_matches['results'], 1):
            print(f"\n{i}. Expert: {result['expert']}")
            print(f"   Similarity Score: {result['similarity_score']:.3f}")
            print(f"   Cosine Similarity: {result['cosine_similarity']:.3f}")
            print(f"   Expertise: {result['expertise']}")
            print(f"   Years of Experience: {result['years_of_experience']}")
            print(f"   Organization: {result['organization']}")
            print(f"   Field of Interest: {result['field_of_interest']}")
            print(f"   Requirements: {result['requirements']}")
    else:
        print("No exact matches found.")
    
    # Print recommended matches
    recommended_matches = response['recommended_matches']
    recommended_metrics = recommended_matches['metrics']
    
    print("\n=== Recommended Matches ===")
    print(f"Total Recommended Matches: {recommended_metrics['total_results']}")
    if recommended_metrics['total_results'] > 0:
        print("\nRecommended Match Metrics:")
        print(f"Average Similarity Score: {recommended_metrics['average_score']:.3f}")
        print(f"Average Cosine Similarity: {recommended_metrics['average_cosine_similarity']:.3f}")
        print(f"Perplexity Score: {recommended_metrics['perplexity']:.3f}")
        
        print("\nRecommended Experts:")
        for i, result in enumerate(recommended_matches['results'], 1):
            print(f"\n{i}. Expert: {result['expert']}")
            print(f"   Match Percentage: {result['match_percentage']*100:.1f}%")
            print(f"   Similarity Score: {result['similarity_score']:.3f}")
            print(f"   Cosine Similarity: {result['cosine_similarity']:.3f}")
            print(f"   Expertise: {result['expertise']}")
            print(f"   Years of Experience: {result['years_of_experience']}")
            print(f"   Organization: {result['organization']}")
            print(f"   Field of Interest: {result['field_of_interest']}")
            print(f"   Requirements: {result['requirements']}")
    else:
        print("No recommended matches found.")

if __name__ == "__main__":
    # Initialize retriever system
    retriever = setup_retriever()
    
    # Example queries
    if retriever:
        # Example 1: Find experts by expertise
        query1 = "i want people who working in the AppGenius Inc. more tha 12 yeasr of exxperinse "
        response1 = query_retriever(retriever, query1)
        print_retrieval_results(query1, response1)
        
        # Example 2: Find experts by years of experience
        query2 = "Find experts in Cloud Computing more than 5 years of experience"
        response2 = query_retriever(retriever, query2)
        print_retrieval_results(query2, response2)
        
        # Example 3: Find experts by organization
        query3 = "Find experts who worked at Google"
        response3 = query_retriever(retriever, query3)
        print_retrieval_results(query3, response3)
