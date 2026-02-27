import streamlit as st
import pandas as pd
import requests
import json
import datetime
import os
import re

# =====================================================================
# Project: Smart IDS for SQL Injection Detection
# Description:
#   This project builds a Smart Intrusion Detection System (Smart IDS)
#   that uses a hybrid approach: rule-based detection + a local LLM
#   (via Ollama) to analyze HTTP inputs for SQL Injection attacks.
#   The system logs all events and provides an admin dashboard.
# =====================================================================


# --- App Settings ---
# I'm using Ollama running locally on port 11434
# The model I chose is qwen2.5:7b-instruct because it's good at structured JSON output
# If it's too heavy on the machine, I left a note to switch to the 3b version
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b-instruct"  # lighter alternative: qwen2.5:3b-instruct

# All logs will go into this CSV file
LOG_FILE = "security_events.csv"

# This is where I store the reference notes for the RAG-lite feature
# The idea is to give the LLM some extra context about SQLi patterns
REFS_FILE = "refs/sqli_notes.txt"


# -----------------------------------------------------------------------
# load_refs()
# This function reads the reference/notes file I prepared about SQL injection.
# It's used to implement a simple RAG (Retrieval-Augmented Generation) approach —
# instead of fine-tuning the model, I just inject relevant knowledge into the prompt.
# If the file doesn't exist, it just returns an empty string and the app still works.
# -----------------------------------------------------------------------
def load_refs():
    try:
        if os.path.exists(REFS_FILE):
            with open(REFS_FILE, "r") as f:
                return f.read()
    except Exception as e:
        st.error(f"Error loading refs: {e}")
    return ""


# -----------------------------------------------------------------------
# log_event()
# I need to keep a persistent record of every request the IDS processes.
# This function appends each event as a new row in a CSV file.
# I'm logging: timestamp, client ID, endpoint, a preview of the input,
# and all the detection results (label, confidence, risk level, decision, tags).
# I had to be careful about newlines in the input text — they break CSV rows,
# so I replace them with spaces in the preview.
# -----------------------------------------------------------------------

def log_event(client_id, endpoint, input_text, result):
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Truncate long inputs so the CSV stays readable
    input_preview = (
        input_text[:100].replace("\n", " ") + "..."
        if len(input_text) > 100
        else input_text.replace("\n", " ")
    )

    # Safely pull values from the result dict — use defaults if missing
    label = str(result.get("label", "unknown"))
    confidence = str(result.get("confidence", 0.0))
    risk_level = str(result.get("risk_level", "unknown"))
    decision = str(result.get("decision", "unknown"))
    # Tags are stored as a list, I join them with | for CSV compatibility
    tags = "|".join(result.get("tags", []))

    new_row = {
        "timestamp_utc": timestamp,
        "client_id": str(client_id),
        "endpoint": str(endpoint),
        "input_preview": input_preview,
        "label": label,
        "confidence": confidence,
        "risk_level": risk_level,
        "decision": decision,
        "tags": tags,
    }

    try:
        # If the file doesn't exist yet, create it with headers
        # Otherwise, append without re-writing the header
        if not os.path.exists(LOG_FILE):
            df = pd.DataFrame([new_row])
            df.to_csv(LOG_FILE, index=False)
        else:
            df = pd.DataFrame([new_row])
            df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    except Exception as e:
        # Don't crash the whole app just because logging failed
        print(f"Logging failed: {e}")


# -----------------------------------------------------------------------
# check_rules()
# Before calling the LLM (which takes time), I run a fast rule-based check.
# This is inspired by traditional IDS tools like Snort/Suricata.
# I use regex patterns to detect the most common SQLi techniques:
#   1. Boolean-based blind: ' OR '1'='1
#   2. SQL comments: -- or /* ... */
#   3. UNION-based injection: UNION SELECT
#   4. Dangerous DDL/DML keywords: DROP TABLE, INSERT INTO, etc.
#   5. Tautologies: 1=1, 2=2, etc.
# Each detection adds to a cumulative score (capped at 1.0)
# and appends a tag for categorization.
# -----------------------------------------------------------------------
def check_rules(input_text):
    score = 0.0
    tags = []

    # Normalize to lowercase for matching, but keep original for output
    text_lower = input_text.lower()

    # Pattern 1: Boolean-based blind injection (e.g. ' OR '1'='1)
    if re.search(r"'\s+or\s+['\d]", text_lower):
        score += 0.5
        tags.append("boolean_based_blind")

    # Pattern 2: SQL comment sequences used to truncate queries
    if "--" in input_text or "/*" in input_text:
        score += 0.3
        tags.append("sql_comment")

    # Pattern 3: UNION SELECT — classic for data extraction
    if re.search(r"union\s+select", text_lower):
        score += 0.8
        tags.append("union_select")

    # Pattern 4: Dangerous SQL keywords that could modify/destroy data
    dangerous_keywords = ["drop table", "insert into", "waitfor delay", "exec(", "shutdown"]
    for kw in dangerous_keywords:
        if kw in text_lower:
            score += 0.7
            tags.append("dangerous_keyword")

    # Pattern 5: Tautologies like 1=1 or 2=2 (always-true conditions)
    if re.search(r"\b(\d+)\s*=\s*\1\b", text_lower):
        score += 0.4
        tags.append("tautology")

    # Make sure score stays in [0, 1] range
    return min(score, 1.0), list(set(tags))


# -----------------------------------------------------------------------
# query_ollama()
# This sends a prompt to the local Ollama API and returns the model's response.
# I set format="json" to force the model to output valid JSON — this is important
# because my analyze_request() function will try to parse the response with json.loads().
# I added a 120s timeout since the model can be slow on CPU.
# If Ollama is offline or the request fails, I return None and handle it upstream.
# -----------------------------------------------------------------------
def query_ollama(prompt):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",  # Forces the model to return valid JSON
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        st.error(f"Ollama Error: {e}. Is Ollama running?")
        return None


# -----------------------------------------------------------------------
# analyze_request()
# This is the main detection pipeline. It combines:
#   Step 1: Rule-based detection (fast, deterministic)
#   Step 2: RAG-lite — optionally inject reference notes into the prompt
#   Step 3: LLM analysis via Ollama (slower, but smarter)
#   Step 4: Policy enforcement — translate the LLM's output into a decision
#
# The decision logic is:
#   - label="normal"                       → ALLOW
#   - label="malicious" + confidence >= 0.8 → BLOCK
#   - label="malicious" + confidence < 0.8  → CHALLENGE (e.g., show CAPTCHA)
#
# If the model is unavailable, we fall back to a safe default (ALLOW + error label).
# We also log every request regardless of outcome.
# -----------------------------------------------------------------------
def analyze_request(input_text, use_refs=False):

    # Step 1: Run fast rule-based pre-filter
    rule_score, rule_tags = check_rules(input_text)

    # Step 2: Optionally load RAG reference notes to enhance the prompt
    refs_content = ""
    if use_refs:
        refs_content = f"\n\nREFERENCE DEFENSIVE NOTES:\n{load_refs()}"

    # Step 3: Build the LLM prompt
    # I use few-shot instruction style with an explicit JSON schema
    # to make the output predictable and easy to parse.
    # I also pass the rule-based score so the LLM can factor it in.
    # Important note: Arabic text in the payload should NOT be flagged —
    # it's valid user input and not inherently malicious.
    prompt = f"""You are a Smart Intrusion Detection System (IDS) expert in SQL Injection. 
    Analyze the following HTTP request input for SQL Injection attacks.
    Return JSON ONLY. No markdown, no prose.
    
    Input: "{input_text}"
    
    Context:
    - Rule-based detection score: {rule_score} (0=safe, 1=danger)
    - Rule flags: {rule_tags}
    {refs_content}
    
    Output Schema:
    {{
      "label": "malicious" | "normal",
      "confidence": float (0.0-1.0),
      "risk_level": "low" | "medium" | "high",
      "reasons": ["reason1", ...],
      "recommended_mitigations": ["mitigation1", ...],
      "tags": ["tag1", ...]
    }}
    
    Strictly follow the JSON format. If Arabic text is present, translate context internally but valid Arabic is NOT malicious by itself.
    """

    # Step 4: Call the LLM
    llm_response = query_ollama(prompt)

    # Default fallback result — used when the model is unavailable
    # We still want the app to run, so we default to ALLOW and tag it as an error
    result = {
        "label": "error",
        "confidence": 0.0,
        "risk_level": "unknown",
        "decision": "ALLOW",
        "reasons": ["Model unavailable"],
        "recommended_mitigations": ["Check Ollama connection"],
        "tags": rule_tags,
    }

    if llm_response:
        try:
            parsed = json.loads(llm_response)
            result.update(parsed)

            # Step 5: Apply decision policy based on LLM output
            label = result.get("label", "normal").lower()
            conf = result.get("confidence", 0.0)

            if label == "normal":
                result["decision"] = "ALLOW"
            elif label == "malicious":
                # High confidence → block immediately
                # Low confidence → challenge the user (e.g., CAPTCHA)
                if conf >= 0.80:
                    result["decision"] = "BLOCK"
                else:
                    result["decision"] = "CHALLENGE"
            else:
                # Unexpected label value — play it safe and challenge
                result["decision"] = "CHALLENGE"

        except json.JSONDecodeError:
            # LLM returned something that wasn't valid JSON — log the issue
            result["reasons"].append("Failed to parse LLM JSON")

    # Log every request (success or failure)
    log_event("unknown_client", "analyze_request", input_text, result)

    return result


# =====================================================================
# Streamlit UI
# I'm using three tabs:
#   Tab 1 - Login Demo: simulates a vulnerable login form
#   Tab 2 - Request Analyzer: test any arbitrary payload manually
#   Tab 3 - Admin Dashboard: view logs, metrics, and use the AI chatbot
# =====================================================================

st.set_page_config(page_title="Smart IDS for SQLi", page_icon="🛡️", layout="wide")

st.title("🛡️ Smart Intrusion Detection System (Smart IDS)")
st.markdown("### Generative AI-Powered SQL Injection Detection")


# Show Ollama connection status in the sidebar
# A quick GET to the root endpoint tells us if the server is alive
try:
    requests.get(OLLAMA_URL.replace("/api/generate", ""), timeout=2)
    st.sidebar.success("🟢 Ollama Online")
except:
    st.sidebar.error("🔴 Ollama Offline")


tab1, tab2, tab3 = st.tabs(["🔐 Login Demo", "🔍 Request Analyzer", "📊 Admin Dashboard"])


# -----------------------------------------------------------------------
# Tab 1: Login Demo
# This simulates a real login form that an attacker might try to inject into.
# Instead of actually authenticating, we run the IDS on the input first
# and decide whether to allow, challenge, or block the attempt.
# -----------------------------------------------------------------------
with tab1:
    st.header("Login Simulation")
    st.info("This form simulates a vulnerable login endpoint. Inputs are analyzed for SQLi attempts.")

    col1, col2 = st.columns(2)
    with col1:
        username = st.text_input("Username", value="admin")
    with col2:
        password = st.text_input("Password", type="password")

    if st.button("Login (Analyze First)"):
        # Combine both fields into a single payload string
        # This mimics how a web server receives form data as query parameters
        payload = f"username={username}&password={password}"
        result = analyze_request(payload)

        decision = result.get("decision", "ALLOW")
        if decision == "ALLOW":
            st.success(f"✅ ALLOWED: Login proceeding for user '{username}'")
        elif decision == "CHALLENGE":
            st.warning("⚠️ CHALLENGE: Suspicious activity detected. CAPTCHA required.")
        else:
            st.error("🚫 BLOCKED: Malicious SQL injection attempt detected.")


# -----------------------------------------------------------------------
# Tab 2: Raw Request Analyzer
# A free-form testing interface where you can paste any payload
# and see exactly what the IDS thinks of it.
# I added a Client ID and endpoint selector to simulate real HTTP context.
# The "Use reference notes" checkbox enables the RAG-lite feature.
# -----------------------------------------------------------------------
with tab2:
    st.header("Raw Request Analyzer")
    st.markdown("Test arbitrary inputs against the Smart IDS pipeline.")

    client_id = st.text_input("Client ID (IP or User)", value="192.168.1.10")
    endpoint = st.selectbox(
        "Target Endpoint",
        ["/login", "/search?q=", "/comment", "/api/v1/user"]
    )

    input_text = st.text_area(
        "Request Input / Payload",
        height=150,
        placeholder="SELECT * FROM users WHERE..."
    )
    use_refs = st.checkbox("Use reference notes (RAG-lite)")

    if st.button("Analyze Request"):
        if input_text:
            result = analyze_request(input_text, use_refs)
            # Note: log_event() is called inside analyze_request() so we don't need to call it again here

            decision = result.get("decision", "ALLOW")
            if decision == "ALLOW":
                st.success(f"DECISION: {decision}")
            elif decision == "CHALLENGE":
                st.warning(f"DECISION: {decision}")
            else:
                st.error(f"DECISION: {decision}")

            # Show the full result dict in an expandable section for debugging
            with st.expander("👮 Admin / Developer View (Details)"):
                st.json(result)
        else:
            st.warning("Please enter some text to analyze.")


# -----------------------------------------------------------------------
# Tab 3: Admin Dashboard
# Reads the CSV log file and shows:
#   - Summary metrics (total events, blocked, challenged)
#   - A table of recent BLOCK/CHALLENGE alerts
#   - Full event log in an expander
#   - An AI chatbot that lets the admin ask questions about specific alerts
#
# The chatbot works by pulling the alert context from the log row
# and injecting it into a new prompt for the LLM.
# -----------------------------------------------------------------------
with tab3:
    st.header("Admin Dashboard")

    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE)

            # Quick summary metrics at the top
            total_events = len(df)
            blocked = len(df[df["decision"] == "BLOCK"])
            challenged = len(df[df["decision"] == "CHALLENGE"])

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Events", total_events)
            m2.metric("Blocked", blocked, delta_color="inverse")
            m3.metric("Challenged", challenged, delta_color="off")

            st.divider()

            # Show the most recent 10 alerts (newest first)
            st.subheader("Recent Alerts (BLOCK/CHALLENGE)")
            alerts_df = df[df["decision"].isin(["BLOCK", "CHALLENGE"])].tail(10).iloc[::-1]
            st.dataframe(alerts_df, use_container_width=True)

            # Full log in a collapsible section to avoid cluttering the page
            with st.expander("All Recent Events Log"):
                st.dataframe(df.tail(20).iloc[::-1], use_container_width=True)

            st.divider()

            # -------------------------------------------------------
            # AI Security Assistant Chatbot
            # The admin can pick any alert from the table and ask the LLM
            # to explain it or suggest mitigation steps.
            # The context (timestamp, payload, tags, risk, confidence)
            # is injected directly into the prompt so the LLM has full info.
            # -------------------------------------------------------
            st.subheader("🤖 Security Assistant Chatbot")

            if not alerts_df.empty:
                # Build a readable label for each alert to show in the dropdown
                options = alerts_df.apply(
                    lambda x: f"{x['timestamp_utc']} - {x['decision']} - {x['input_preview']}",
                    axis=1
                ).tolist()
                selected_option = st.selectbox("Select an alert to analyze with AI:", options)

                # Get the actual row that matches the selected option
                selected_idx = options.index(selected_option)
                selected_row = alerts_df.iloc[selected_idx]

                user_question = st.text_input(
                    "Ask the assistant:",
                    value="Explain this alert and suggest mitigations."
                )

                if st.button("Ask Assistant"):
                    with st.spinner("AI analyzing..."):
                        # Build a context-rich prompt for the chatbot
                        # I ask for JSON output so I can render it cleanly in the UI
                        context_prompt = f"""
                        You are a Security Assistant. An admin is asking about this specific security alert:
                        Timestamp: {selected_row['timestamp_utc']}
                        Input Payload: {selected_row['input_preview']}
                        Decision: {selected_row['decision']}
                        Tags: {selected_row['tags']}
                        Risk: {selected_row['risk_level']}
                        Confidence: {selected_row['confidence']}
                        
                        Admin Question: {user_question}
                        
                        Return JSON ONLY with this format:
                        {{"analysis": "your explanation here", "mitigations": ["step1", "step2", ...]}}
                        Focus on defensive measures.
                        """

                        answer = query_ollama(context_prompt)
                        st.markdown("### 💡 AI Response")

                        if answer:
                            try:
                                parsed_answer = json.loads(answer)
                                if isinstance(parsed_answer, dict):
                                    # Render the analysis text and mitigation list cleanly
                                    st.markdown(f"**Analysis:** {parsed_answer.get('analysis', answer)}")
                                    mitigations = parsed_answer.get("mitigations", [])
                                    if mitigations:
                                        st.markdown("**Recommended Mitigations:**")
                                        for i, m in enumerate(mitigations, 1):
                                            st.markdown(f"{i}. {m}")
                                else:
                                    st.markdown(str(parsed_answer))
                            except json.JSONDecodeError:
                                # If the model didn't return valid JSON, just show raw text
                                st.markdown(answer)
                        else:
                            st.warning("AI assistant could not respond. Please check that Ollama is running.")

            else:
                st.info("No alerts found to analyze.")

        except Exception as e:
            st.error(f"Error reading logs: {e}")
    else:
        st.info("No security events logged yet.")