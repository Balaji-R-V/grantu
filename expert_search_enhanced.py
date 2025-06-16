import streamlit as st
import pandas as pd
from Faiss import setup_retriever, query_retriever, print_retrieval_results
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Expert Search System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
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
        background-color: #2b313e;
        color: white;
    }
    .profile-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
        gap: 25px;
        padding: 20px 0;
    }
    .profile-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        overflow: hidden;
        position: relative;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        min-height: 280px;
    }
    .profile-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }
    .profile-card.exact-match {
        border: 3px solid #28a745;
    }
    .profile-card.recommended-match {
        border: 3px solid #ffc107;
    }
    .profile-header {
        position: relative;
        padding: 25px 20px 15px;
        color: white;
        text-align: center;
    }
    .profile-avatar {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        margin: 0 auto 15px;
        background-image: url('https://cdn-icons-png.flaticon.com/512/149/149071.png');
        background-size: cover;
        background-position: center;
        border: 4px solid rgba(255,255,255,0.3);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .profile-name {
        font-size: 1.4em;
        font-weight: bold;
        margin-bottom: 8px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .profile-title {
        font-size: 0.95em;
        opacity: 0.9;
        margin-bottom: 5px;
        font-weight: 500;
    }
    .profile-organization {
        font-size: 0.85em;
        opacity: 0.8;
        font-style: italic;
    }
    .profile-content {
        background: white;
        padding: 20px;
        margin: 0;
        flex-grow: 1;
    }
    .profile-section {
        margin-bottom: 15px;
    }
    .profile-section-title {
        font-size: 0.8em;
        font-weight: bold;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .profile-section-content {
        font-size: 0.9em;
        color: #333;
        line-height: 1.4;
    }
    .expertise-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 5px;
    }
    .expertise-tag {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.75em;
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 120px;
        display: inline-block;
    }
    .experience-badge {
        background: #28a745;
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: bold;
        display: inline-block;
        margin-top: 5px;
    }
    .match-indicator {
        position: absolute;
        top: 15px;
        right: 15px;
        padding: 5px 10px;
        border-radius: 12px;
        font-size: 0.7em;
        font-weight: bold;
        text-transform: uppercase;
    }
    .match-indicator.exact {
        background: #28a745;
        color: white;
    }
    .match-indicator.recommended {
        background: #ffc107;
        color: #333;
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
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #eee;
    }
    .section-title {
        font-size: 1.5em;
        font-weight: bold;
        color: #333;
        margin: 0;
    }
    .results-count {
        background: #667eea;
        color: white;
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 0.8em;
        font-weight: bold;
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

def display_expert_card(expert, is_exact_match=True):
    """Display an enhanced expert profile card with comprehensive details."""
    match_class = "exact-match" if is_exact_match else "recommended-match"
    match_type = "exact" if is_exact_match else "recommended"
    match_label = "Exact Match" if is_exact_match else "Recommended"
    
    # Extract and format expert details
    name = expert.get('expert', 'Unknown Expert')
    organization = expert.get('organization', '')
    expertise = expert.get('expertise', '')
    experience = expert.get('years_of_experience', '')
    field_of_interest = expert.get('field_of_interest', '')
    requirements = expert.get('requirements', '')
    
    # Create expertise tags
    expertise_list = []
    if expertise:
        expertise_list.extend([skill.strip() for skill in expertise.split(',') if skill.strip()])
    if field_of_interest and field_of_interest not in expertise:
        expertise_list.extend([field.strip() for field in field_of_interest.split(',') if field.strip()])
    
    # Generate color gradient based on name hash for variety
    colors = [
        "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
        "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
        "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)",
        "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
        "linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)",
        "linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%)",
        "linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)"
    ]
    
    color_index = hash(name) % len(colors)
    gradient = colors[color_index]
    
    # Build expertise tags HTML
    expertise_tags_html = ""
    if expertise_list:
        for skill in expertise_list[:4]:
            truncated_skill = skill[:15] + ("..." if len(skill) > 15 else "")
            expertise_tags_html += f'<span class="expertise-tag" title="{skill}">{truncated_skill}</span>'
    
    # Build profile sections
    profile_sections = ""
    
    # Expertise section
    if expertise_list:
        profile_sections += f"""
        <div class="profile-section">
            <div class="profile-section-title">🎯 Expertise</div>
            <div class="expertise-tags">{expertise_tags_html}</div>
        </div>
        """
    
    # Experience section
    if experience:
        profile_sections += f"""
        <div class="profile-section">
            <div class="profile-section-title">💼 Experience</div>
            <div class="experience-badge">{experience}+ Years</div>
        </div>
        """
    
    # Requirements section
    if requirements:
        truncated_requirements = requirements[:100] + ("..." if len(requirements) > 100 else "")
        profile_sections += f"""
        <div class="profile-section">
            <div class="profile-section-title">📋 Requirements</div>
            <div class="profile-section-content">{truncated_requirements}</div>
        </div>
        """
    
    # Specialization section
    if field_of_interest and field_of_interest != expertise:
        truncated_field = field_of_interest[:100] + ("..." if len(field_of_interest) > 100 else "")
        profile_sections += f"""
        <div class="profile-section">
            <div class="profile-section-title">🔬 Specialization</div>
            <div class="profile-section-content">{truncated_field}</div>
        </div>
        """
    
    # Organization display
    org_html = f'<div class="profile-organization">📍 {organization}</div>' if organization else ''
    
    # Complete card HTML
    card_html = f"""
    <div class="profile-card {match_class}">
        <div class="match-indicator {match_type}">{match_label}</div>
        <div class="profile-header" style="background: {gradient};">
            <div class="profile-avatar"></div>
            <div class="profile-name">{name}</div>
            {org_html}
        </div>
        <div class="profile-content">
            {profile_sections}
        </div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)

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
        st.markdown("### 💡 Example Queries")
        st.markdown("*Click any example to try it out*")
        
        for i, query_text in enumerate(EXAMPLE_QUERIES):
            if st.button(query_text, key=f"example_{i}", help="Click to search"):
                st.session_state.last_query = query_text
                st.session_state.last_response = None
                with st.spinner(f"🔍 Searching for experts..."):
                    response = query_retriever(st.session_state.retriever, query_text)
                    if not isinstance(response, str):
                        st.session_state.last_response = response
                    else:
                        st.error(response)
                st.rerun()

    # Main content area
    st.markdown("---")

    if st.session_state.last_query:
        display_chat_message(st.session_state.last_query, is_user=True)
        
        if st.session_state.last_response:
            display_chat_message("Here are your expert search results:", is_user=False)

            response_content = st.session_state.last_response

            # Display search criteria
            criteria = response_content['search_criteria']
            criteria_text = []
            if criteria.expertise:
                criteria_text.append(f"🎯 **Expertise:** {', '.join(criteria.expertise)}")
            if criteria.years_of_experience:
                criteria_text.append(f"⏱️ **Experience:** {criteria.years_of_experience}+ years")
            if criteria.organization:
                criteria_text.append(f"🏢 **Organizations:** {', '.join(criteria.organization)}")
            if criteria.field_of_interest:
                criteria_text.append(f"🔬 **Fields:** {', '.join(criteria.field_of_interest)}")
            if criteria.requirements:
                criteria_text.append(f"📋 **Requirements:** {', '.join(criteria.requirements)}")

            if criteria_text:
                st.markdown("**🔍 Search Criteria:**")
                st.markdown(" | ".join(criteria_text))

            # Display exact matches
            exact_count = response_content['exact_matches']['metrics']['total_results']
            if exact_count > 0:
                st.markdown(f"""
                    <div class="section-header">
                        <h3 class="section-title">✅ Exact Matches</h3>
                        <span class="results-count">{exact_count} Results</span>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="profile-grid">', unsafe_allow_html=True)
                for expert in response_content['exact_matches']['results']:
                    display_expert_card(expert, is_exact_match=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("ℹ️ No exact matches found.")

            # Display recommended matches  
            recommended_count = response_content['recommended_matches']['metrics']['total_results']
            if recommended_count > 0:
                st.markdown(f"""
                    <div class="section-header">
                        <h3 class="section-title">💡 Recommended Matches</h3>
                        <span class="results-count">{recommended_count} Results</span>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="profile-grid">', unsafe_allow_html=True)
                for expert in response_content['recommended_matches']['results']:
                    display_expert_card(expert, is_exact_match=False)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("ℹ️ No recommended matches found.")
            
            # Display metrics
            display_metrics_section(
                response_content['exact_matches']['metrics'],
                response_content['recommended_matches']['metrics']
            )
    
    # Chat input at the bottom
    st.markdown("---")
    with st.form(key="query_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query_input = st.text_input(
                "",
                placeholder="🔍 Example: Find experts in Cloud Computing with more than 5 years of experience",
                key="query_text_input",
                label_visibility="collapsed"
            )
        
        with col2:
            submit_button = st.form_submit_button("🚀 Search", use_container_width=True)

        if submit_button and query_input:
            st.session_state.last_query = query_input
            st.session_state.last_response = None
            with st.spinner("🔍 Searching for experts..."):
                response = query_retriever(st.session_state.retriever, query_input)
                if not isinstance(response, str):
                    st.session_state.last_response = response
                else:
                    st.error(f"❌ Error: {response}")
            st.rerun()

if __name__ == "__main__":
    main()