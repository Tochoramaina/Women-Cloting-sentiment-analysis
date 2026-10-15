import os 
import requests
import streamlit as st 
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title='Womens clothing sentiment and topics', layout="centered")
st.title("Women's Clothing Review Sentiment and Topic Analyser")
st.write("Enter a customer product review below to analyze its sentiment and extract underlying topics")

review_text = st.text_area("Customer Review", placeholder="type or paste customer review here", height=150)
if st.button('Analyze Review', type='Primary'):
    if not review_text.strip():
        st.warning('Please enter som review text before clicking analyze')
    else:
        with st.spinner("Analyzing sentiment and topics ...."):
            try:
                response = requests.post(f'{API_URL}/predict', json={"text": review_text})
                if response.status_code == 200:
                    result = response.json()
                    sentiment = result['sentiment']
                    confidence = result["confidence_percent"]
                    topics = result.get("review_topics", [])

                    st.divider()

                    col1, col2 = st.columns(2)
                    with col1:
                        if sentiment.lower() in ["positive", "1", "good"]:
                            st.success(f"**sentiment:** {sentiment}")
                        else:
                            st.error(f"**Sentiment:** {sentiment}")
                    with col2:
                        st.metric(label="Confidence", values=f"{confidence:.2f}%")
                    with st.expander("View preprocessed text"):
                        st.code(result['processed_review'], language="text")


                    if topics:
                        st.subheader("underlying review topics")
                        for topic in topics:
                            words_str = ", ".join(topic["top_words"])
                            prob_val = topic['probability'] * 100
                            st.markdown(
                               f"- **Topic {topic['topic_id']} ** (Match:" 
                               f"`{prob_val:.1f}%`): *{words_str}*"
                            )
                        else:
                            st.info("No dominant topic thresholds met of this review")
                    else:
                        st.error(
                            f"API Error ({response.status_code}): {response.text}"

                        )
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the FastAPI backend. Ensure your API container", 
                                 "/ or local uvicorn server is running"
                                )
            except Exception as e:
                st.error(f"An expected error occured: {e}")
                    
                    
    