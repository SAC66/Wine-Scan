import os
import io
import json
import base64
import PIL.Image
import streamlit as st
import requests
from pypdf import PdfReader

st.set_page_config(page_title="Wine List Value Scanner", page_icon="🍷")

st.title("🍷 Wine List Value Scanner")
st.write("Take or upload a photo or PDF of a restaurant wine list to spot top values and markup estimates.")

api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("Please add your GEMINI_API_KEY in the Streamlit app settings.")
else:
    uploaded_file = st.file_uploader("Snap or Choose Wine List", type=["jpg", "jpeg", "png", "pdf"])
    
    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1].lower()
        
        prompt = """
        You are an expert sommelier and wine pricing analyst. Analyze this restaurant wine list.
        Extract every wine listed (Producer, Vintage, Name, Varietal/Region, and Restaurant List Price).
        For each wine:
        1. Estimate its average retail market price.
        2. Calculate the restaurant markup multiplier (List Price / Retail Price).
        3. Rate it as 'Top Value / Gem' (markup < 2.2x), 'Fair Value' (markup 2.2x - 3.2x), or 'High Markup' (markup > 3.2x).
        4. Provide a brief 1-sentence note on why it is or isn't a good buy.

        Format the output cleanly with clear headers, bullet points, and value badges for each wine.
        """
        
        parts = [{"text": prompt}]
        
        if file_type in ["jpg", "jpeg", "png"]:
            image = PIL.Image.open(uploaded_file)
            st.image(image, caption="Uploaded Wine List", use_container_width=True)
            
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": img_base64
                }
            })
        elif file_type == "pdf":
            st.success(f"Uploaded PDF: {uploaded_file.name}")
            reader = PdfReader(uploaded_file)
            pdf_text = ""
            for page in reader.pages:
                pdf_text += page.extract_text() or ""
            parts[0]["text"] = f"{prompt}\n\nWine List Content:\n{pdf_text}"

        if st.button("🔍 Analyze Wine List", type="primary"):
            with st.spinner("Analyzing wine list and calculating retail values..."):
                try:
                    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
                    
                    headers = {"Content-Type": "application/json"}
                    # Automatically use Bearer token header for AQ tokens, and API key header for AIza keys
                    if api_key.strip().startswith("AQ."):
                        headers["Authorization"] = f"Bearer {api_key.strip()}"
                    else:
                        headers["x-goog-api-key"] = api_key.strip()
                        
                    payload = {"contents": [{"parts": parts}]}
                    
                    response = requests.post(url, headers=headers, data=json.dumps(payload))
                    result = response.json()
                    
                    if response.status_code == 200:
                        analysis_text = result["candidates"][0]["content"]["parts"][0]["text"]
                        st.markdown("### 📊 Value Analysis")
                        st.markdown(analysis_text)
                    else:
                        st.error(f"API Error: {result}")
                except Exception as e:
                    st.error(f"Error analyzing document: {e}")
