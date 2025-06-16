import streamlit as st
import pandas as pd
from Faiss import setup_retriever, query_retriever, print_retrieval_results
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Expert Search System",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
        color: white;
    }
    .chat-message.assistant {
        background-color: #2b313e; /* Changed to match user message background for consistency */
        color: white;
    }
    .profile-grid {
        display: grid; /* Changed back to grid layout */
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); /* Adjust column width as needed */
        gap: 20px;
        padding: 20px 0;
    }
    .profile-tile {
        display: flex;
        flex-direction: column;
        background-color: #f8f8f8; /* Light background */
        border-radius: 15px; /* Softer rounded corners */
        box-shadow: 0 4px 10px rgba(0,0,0,0.08); /* Softer shadow */
        transition: transform 0.2s;
        overflow: hidden;
        position: relative;
        padding: 20px;
        min-height: 220px; /* Ensure enough height for content */
    }
    .profile-tile:hover {
        transform: translateY(-5px);
    }
    .profile-tile.exact-match {
        border: 2px solid rgba(40,167,69,0.5); /* Lighter border for exact match */
    }
    .profile-tile.recommended-match {
        border: 2px solid rgba(255,193,7,0.5); /* Lighter border for recommended match */
    }
    .profile-avatar-container {
        width: 80px; /* Size for circular image */
        height: 80px;
        border-radius: 50%;
        overflow: hidden;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        background-color: #e0e0e0; /* Placeholder background */
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .profile-avatar {
        width: 100%;
        height: 100%;
        background-size: cover;
        background-position: center;
    }
    .profile-text-content {
        flex-grow: 1;
    }
    .profile-name {
        font-size: 1.2em;
        font-weight: bold;
        margin: 0 0 5px 0;
        color: #333; /* Darker text for readability */
    }
    .profile-description-line {
        font-size: 0.9em;
        line-height: 1.4;
        color: #555;
        margin-bottom: 10px;
    }
    /* Hide elements from previous designs that are no longer needed */
    .profile-image-section,
    .profile-image,
    .profile-info-section,
    .profile-designation,
    .profile-description,
    .profile-detail-line,
    .profile-metrics-inline,
    .profile-tile-image,
    .profile-overlay,
    .profile-overlay-name,
    .profile-overlay-info {
        display: none !important; /* Force hide with !important */
    }
    .metric-badge {
        display: none !important; /* Hide metric badges from tiles */
    }
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 15px;
        padding: 15px;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.2em;
        font-weight: bold;
        color: #2b313e;
    }
    .metric-label {
        font-size: 0.8em;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .example-query {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .example-query:hover {
        background-color: #e0e2e6;
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
        padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Example queries
EXAMPLE_QUERIES = [
    "Find experts in Cloud Computing with more than 5 years of experience",
    "Show me people who worked at Google and have AI expertise",
    "Find experts in Machine Learning with experience in healthcare",
    "Show me experts who worked at Microsoft and have cloud expertise",
    "Find people with more than 10 years of experience in software development"
]

def display_expert_tile(expert, is_exact_match=True):
    """Display an expert profile tile with a circular image and descriptive text."""
    match_class = "exact-match" if is_exact_match else "recommended-match"
    
    # Generate a concise description of what the expert helps with
    help_description_parts = []
    if expert['expertise']:
        help_description_parts.append(f"helps with {expert['expertise']}")
    if expert['requirements']:
        help_description_parts.append(f"requires {expert['requirements']}")
    elif expert['field_of_interest']:
        help_description_parts.append(f"interested in {expert['field_of_interest']}")
    
    if expert['organization']:
        help_description = f"{expert['expert']} works at {expert['organization']} and {' and '.join(help_description_parts)}."
    else:
        help_description = f"{expert['expert']} {' and '.join(help_description_parts)}."

    # Use a generic image placeholder for the profile picture
    image_url = "https://cdn-icons-png.flaticon.com/512/149/149071.png" # Generic user icon

    st.markdown(f"""
        <div class="profile-tile {match_class}">
            <div class="profile-avatar-container">
                <div class="profile-avatar" style="background-image: url('{image_url}');"></div>
            </div>
            <div class="profile-text-content">
                <div class="profile-name">{expert['expert']}</div>
                <div class="profile-description-line">{help_description}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def display_metrics_section(exact_metrics, recommended_metrics):
    """Display metrics as values in a collapsible section"""
    with st.expander("📊 View Search Metrics", expanded=False):
        st.markdown("### Search Metrics")
        
        # Exact matches metrics
        st.markdown("#### Exact Matches Metrics")
        st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{exact_metrics['total_results']}</div>
                <div class="metric-label">Total Results</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{exact_metrics['average_score']:.3f}</div>
                <div class="metric-label">Average Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{exact_metrics['average_cosine_similarity']:.3f}</div>
                <div class="metric-label">Avg Cosine Similarity</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{exact_metrics['perplexity']:.3f}</div>
                <div class="metric-label">Perplexity</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommended matches metrics
        st.markdown("#### Recommended Matches Metrics")
        st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{recommended_metrics['total_results']}</div>
                <div class="metric-label">Total Results</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{recommended_metrics['average_score']:.3f}</div>
                <div class="metric-label">Average Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{recommended_metrics['average_cosine_similarity']:.3f}</div>
                <div class="metric-label">Avg Cosine Similarity</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{recommended_metrics['perplexity']:.3f}</div>
                <div class="metric-label">Perplexity</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def display_chat_message(message, is_user=True):
    """Display a chat message"""
    st.markdown(f"""
        <div class="chat-message {'user' if is_user else 'assistant'}">
            {message}
        </div>
    """, unsafe_allow_html=True)

def main():
    st.title("🔍 Expert Search System")
    
    # Initialize session state
    if 'retriever' not in st.session_state:
        with st.spinner("Initializing search system..."):
            st.session_state.retriever = setup_retriever()
    
    # Store the last query and its response
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
    if 'last_response' not in st.session_state:
        st.session_state.last_response = None

    # Sidebar with example queries
    with st.sidebar:
        st.markdown("### Example Queries")
        for query_text in EXAMPLE_QUERIES:
            if st.button(query_text, key=f"example_{query_text}"):
                st.session_state.last_query = query_text
                st.session_state.last_response = None # Clear previous response
                with st.spinner(f"Searching for experts for: {query_text}"):
                    response = query_retriever(st.session_state.retriever, query_text)
                    if not isinstance(response, str):
                        st.session_state.last_response = response
                    else:
                        st.error(response)
                st.rerun() # Rerun to display the new response

    # Main content area - display current query and response
    st.markdown("---") # Separator for visual clarity

    if st.session_state.last_query:
        display_chat_message(st.session_state.last_query, is_user=True)
        if st.session_state.last_response:
            display_chat_message("Here are your expert search results:", is_user=False)

            response_content = st.session_state.last_response

            # Display search criteria
            criteria = response_content['search_criteria']
            criteria_text = []
            if criteria.expertise:
                criteria_text.append(f"Expertise: {', '.join(criteria.expertise)}")
            if criteria.years_of_experience:
                criteria_text.append(f"Experience: {criteria.years_of_experience}+ years")
            if criteria.organization:
                criteria_text.append(f"Organizations: {', '.join(criteria.organization)}")
            if criteria.field_of_interest:
                criteria_text.append(f"Fields: {', '.join(criteria.field_of_interest)}")
            if criteria.requirements:
                criteria_text.append(f"Requirements: {', '.join(criteria.requirements)}")

            st.markdown("**Search Criteria:** " + " | ".join(criteria_text))

            # Display exact matches in grid
            st.markdown("### Exact Matches")
            if response_content['exact_matches']['metrics']['total_results'] > 0:
                st.markdown('<div class="profile-grid">', unsafe_allow_html=True)
                for expert in response_content['exact_matches']['results']:
                    display_expert_tile(expert, is_exact_match=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No exact matches found.")

            # Display recommended matches in grid
            st.markdown("### Recommended Matches")
            if response_content['recommended_matches']['metrics']['total_results'] > 0:
                st.markdown('<div class="profile-grid">', unsafe_allow_html=True)
                for expert in response_content['recommended_matches']['results']:
                    display_expert_tile(expert, is_exact_match=False)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No recommended matches found.")
            
            # Display metrics in collapsible section
            display_metrics_section(
                response_content['exact_matches']['metrics'],
                response_content['recommended_matches']['metrics']
            )
    
    # Chat input at the bottom
    with st.form(key="query_form", clear_on_submit=True):
        query_input = st.text_input(
            "Ask about experts...",
            placeholder="Example: Find experts in Cloud Computing with more than 5 years of experience",
            key="query_text_input"
        )
        submit_button = st.form_submit_button("Search")

        if submit_button and query_input:
            st.session_state.last_query = query_input
            st.session_state.last_response = None # Clear previous response
            with st.spinner("Searching for experts..."):
                response = query_retriever(st.session_state.retriever, query_input)
                if not isinstance(response, str):
                    st.session_state.last_response = response
                else:
                    st.error(response)
            st.rerun()

if __name__ == "__main__":
    main() 