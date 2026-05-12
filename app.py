from datetime import date
from io import BytesIO
import base64
import json
import os
import tempfile

import requests
import streamlit as st

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


st.set_page_config(
    page_title="Home Inspection AI",
    page_icon=":house:",
    layout="wide",
    initial_sidebar_state="expanded",
)


REPORT_SECTIONS = [
    "Exterior",
    "Roofing",
    "Interior",
    "Plumbing",
    "Electrical",
    "HVAC",
    "Garage",
    "Foundation / Structure",
    "Kitchen",
    "Bathroom",
    "Safety Concerns",
    "Other",
]

SEVERITY_OPTIONS = ["Monitor", "Maintenance", "Repair", "Safety", "Further Evaluation"]

COMMENT_LIBRARY = {
    "crack": {
        "finding": "Cracking was observed at the referenced area.",
        "recommendation": "Recommend monitoring for movement and further evaluation by a qualified professional if the crack widens, shows displacement, or is associated with moisture intrusion.",
        "severity": "Further Evaluation",
    },
    "gutter": {
        "finding": "Debris or improper drainage conditions were observed at the gutter system.",
        "recommendation": "Recommend cleaning and servicing the gutter system so roof runoff drains away from the structure.",
        "severity": "Maintenance",
    },
    "outlet": {
        "finding": "An electrical receptacle concern was observed.",
        "recommendation": "Recommend correction by a qualified electrical contractor.",
        "severity": "Safety",
    },
    "leak": {
        "finding": "Evidence of leakage or moisture staining was observed.",
        "recommendation": "Recommend identifying the source of moisture and repairing as needed before concealed damage develops.",
        "severity": "Repair",
    },
    "roof": {
        "finding": "A roof covering concern was observed.",
        "recommendation": "Recommend repair or further evaluation by a qualified roofing contractor.",
        "severity": "Further Evaluation",
    },
}


st.markdown(
    """
    <style>
        :root {
            --hia-ink: #1f2933;
            --hia-muted: #607080;
            --hia-line: #d9e0e7;
            --hia-panel: #f7f9fb;
            --hia-accent: #176b87;
            --hia-accent-dark: #0d4659;
            --hia-warm: #f2a65a;
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }

        h1, h2, h3 {
            color: var(--hia-ink);
            letter-spacing: 0;
        }

        div[data-testid="stMetric"] {
            background: var(--hia-panel);
            border: 1px solid var(--hia-line);
            border-radius: 8px;
            padding: 0.85rem 1rem;
        }

        section[data-testid="stSidebar"] {
            background-color: #f4f7f9;
            border-right: 1px solid var(--hia-line);
        }

        section[data-testid="stSidebar"] * {
            color: #1d1d1f !important;
        }

        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea,
        section[data-testid="stSidebar"] select {
            background-color: #ffffff !important;
            color: #1d1d1f !important;
            border: 1px solid var(--hia-line) !important;
            border-radius: 8px !important;
        }

        section[data-testid="stSidebar"] label {
            color: #1d1d1f !important;
            font-weight: 600 !important;
        }

        .hia-hero {
            border-bottom: 1px solid var(--hia-line);
            padding-bottom: 1rem;
            margin-bottom: 1rem;
        }

        .hia-hero p {
            color: var(--hia-muted);
            font-size: 1.05rem;
            margin-bottom: 0;
        }

        .hia-callout {
            background: #eef8fb;
            border-left: 4px solid var(--hia-accent);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            color: var(--hia-ink);
            margin: 0.5rem 0 1rem;
        }

        .hia-small {
            color: var(--hia-muted);
            font-size: 0.9rem;
        }

        .hia-report-block {
            border: 1px solid var(--hia-line);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            background: #ffffff;
        }

        .hia-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.15rem 0.55rem;
            font-size: 0.78rem;
            font-weight: 700;
            background: #fff4e6;
            color: #8a4b00;
            border: 1px solid #ffd7a3;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 8px;
            border: 1px solid var(--hia-accent);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_state():
    if "report_items" not in st.session_state:
        st.session_state.report_items = []
    if "last_ai_camera_note" not in st.session_state:
        st.session_state.last_ai_camera_note = ""
    if "current_note" not in st.session_state:
        st.session_state.current_note = ""
    if "last_transcript" not in st.session_state:
        st.session_state.last_transcript = ""


def match_comment_library(raw_note):
    note = raw_note.lower()
    for keyword, comment in COMMENT_LIBRARY.items():
        if keyword in note:
            return comment
    return None


def draft_report_item(section, raw_note, severity, mode, location, photo_name):
    finding_text = raw_note.strip()
    recommendation = "Recommend further evaluation and repair by a qualified professional as needed."
    final_severity = severity
    library_match = match_comment_library(finding_text)

    if mode == "Comment Library" and library_match:
        finding = f"{library_match['finding']} Inspector note: {finding_text}"
        recommendation = library_match["recommendation"]
        final_severity = library_match["severity"]
    elif mode == "Full AI Draft":
        finding = (
            f"{finding_text.rstrip('.')}. This condition should be documented in the "
            f"{section.lower()} section and reviewed with the client."
        )
        recommendation = (
            "Recommend correction, monitoring, or further evaluation based on the inspector's final assessment "
            "and applicable standards of practice."
        )
    else:
        finding = finding_text

    return {
        "section": section,
        "location": location.strip(),
        "finding": finding,
        "recommendation": recommendation,
        "severity": final_severity,
        "source_note": raw_note.strip(),
        "mode": mode,
        "photo_name": photo_name,
        "approved": False,
    }


def get_openai_client(api_key):
    if OpenAI is None:
        raise RuntimeError("The openai package is not installed. Run pip install -r requirements.txt.")
    if not api_key:
        raise RuntimeError("Add an OpenAI API key in the sidebar before using transcription or AI rewrite.")
    return OpenAI(api_key=api_key)


def transcribe_audio_openai(audio_file, api_key, model):
    client = get_openai_client(api_key)
    audio_bytes = audio_file.getvalue()
    file_name = audio_file.name or "inspection_note.wav"

    transcript = client.audio.transcriptions.create(
        model=model,
        file=(file_name, audio_bytes, audio_file.type or "audio/wav"),
        prompt=(
            "Home inspection field note. Preserve measurements, locations, defect names, "
            "materials, and contractor trade terms."
        ),
    )
    return transcript.text.strip()


@st.cache_resource(show_spinner=False)
def load_local_whisper_model(model_name):
    if WhisperModel is None:
        raise RuntimeError("The faster-whisper package is not installed. Run pip install -r requirements.txt.")
    return WhisperModel(model_name, device="cpu", compute_type="int8")


def transcribe_audio_local(audio_file, model_name):
    suffix = os.path.splitext(audio_file.name or "inspection_note.wav")[1] or ".wav"
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            temp_audio.write(audio_file.getvalue())
            temp_path = temp_audio.name

        model = load_local_whisper_model(model_name)
        segments, _ = model.transcribe(
            temp_path,
            beam_size=5,
            vad_filter=True,
            initial_prompt=(
                "Home inspection field note with locations, measurements, defects, materials, "
                "systems, and recommendations."
            ),
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def parse_json_content(content):
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return json.loads(cleaned)


def ollama_chat(ollama_url, model, messages, images=None, force_json=False):
    payload_messages = []
    for message in messages:
        payload_message = dict(message)
        if images and message is messages[-1]:
            payload_message["images"] = images
        payload_messages.append(payload_message)

    payload = {
        "model": model,
        "messages": payload_messages,
        "stream": False,
    }
    if force_json:
        payload["format"] = "json"

    response = requests.post(
        f"{ollama_url.rstrip('/')}/api/chat",
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def rewrite_note_with_openai(raw_note, section, location, severity, api_key, model):
    client = get_openai_client(api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You help a home inspector turn field notes into concise report language. "
                    "Do not overstate certainty. Do not diagnose hidden conditions. "
                    "Recommend specialist evaluation when appropriate. Return only JSON."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "selected_section": section,
                        "location": location,
                        "initial_severity": severity,
                        "raw_note": raw_note,
                        "allowed_sections": REPORT_SECTIONS,
                        "allowed_severities": SEVERITY_OPTIONS,
                        "required_json_shape": {
                            "section": "one allowed section",
                            "severity": "one allowed severity",
                            "finding": "professional inspection finding",
                            "recommendation": "client-facing recommendation",
                        },
                    }
                ),
            },
        ],
    )

    content = response.choices[0].message.content or "{}"
    data = json.loads(content)
    rewritten_section = data.get("section", section)
    rewritten_severity = data.get("severity", severity)

    return {
        "section": rewritten_section if rewritten_section in REPORT_SECTIONS else section,
        "location": location.strip(),
        "finding": data.get("finding", raw_note.strip()).strip(),
        "recommendation": data.get(
            "recommendation",
            "Recommend further evaluation and repair by a qualified professional as needed.",
        ).strip(),
        "severity": rewritten_severity if rewritten_severity in SEVERITY_OPTIONS else severity,
        "source_note": raw_note.strip(),
        "mode": "AI Professional Rewrite",
        "photo_name": "",
        "approved": False,
    }


def rewrite_note_with_ollama(raw_note, section, location, severity, ollama_url, model):
    content = ollama_chat(
        ollama_url,
        model,
        [
            {
                "role": "system",
                "content": (
                    "You help a home inspector turn field notes into concise report language. "
                    "Do not overstate certainty. Do not diagnose hidden conditions. "
                    "Recommend specialist evaluation when appropriate. Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "selected_section": section,
                        "location": location,
                        "initial_severity": severity,
                        "raw_note": raw_note,
                        "allowed_sections": REPORT_SECTIONS,
                        "allowed_severities": SEVERITY_OPTIONS,
                        "required_json_shape": {
                            "section": "one allowed section",
                            "severity": "one allowed severity",
                            "finding": "professional inspection finding",
                            "recommendation": "client-facing recommendation",
                        },
                    }
                ),
            },
        ],
        force_json=True,
    )
    data = parse_json_content(content)
    rewritten_section = data.get("section", section)
    rewritten_severity = data.get("severity", severity)

    return {
        "section": rewritten_section if rewritten_section in REPORT_SECTIONS else section,
        "location": location.strip(),
        "finding": data.get("finding", raw_note.strip()).strip(),
        "recommendation": data.get(
            "recommendation",
            "Recommend further evaluation and repair by a qualified professional as needed.",
        ).strip(),
        "severity": rewritten_severity if rewritten_severity in SEVERITY_OPTIONS else severity,
        "source_note": raw_note.strip(),
        "mode": f"Ollama {model} Rewrite",
        "photo_name": "",
        "approved": False,
    }


def analyze_image_with_ollama(image_file, prompt, ollama_url, model):
    image_b64 = base64.b64encode(image_file.getvalue()).decode("utf-8")
    return ollama_chat(
        ollama_url,
        model,
        [
            {
                "role": "system",
                "content": (
                    "You are assisting a home inspector from a single image. Describe visible concerns only. "
                    "Do not claim hidden defects or make final diagnoses. Suggest what the inspector should verify."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        images=[image_b64],
    )


def build_report_text(inspection_info, report_items, approved_only=False):
    items = [item for item in report_items if item["approved"]] if approved_only else report_items

    lines = [
        "HOME INSPECTION REPORT",
        f"Inspector: {inspection_info['inspector_name'] or 'N/A'}",
        f"Property: {inspection_info['property_address'] or 'N/A'}",
        f"Client: {inspection_info['client_name'] or 'N/A'}",
        f"Date: {inspection_info['inspection_date']}",
        "",
    ]

    if not items:
        lines.append("No approved report items yet." if approved_only else "No report items yet.")
        return "\n".join(lines)

    grouped = {}
    for item in items:
        grouped.setdefault(item["section"], []).append(item)

    for section, section_items in grouped.items():
        lines.append(section.upper())
        for index, item in enumerate(section_items, start=1):
            location = f"Location: {item['location']}" if item["location"] else "Location: Not specified"
            photo = f"Photo: {item['photo_name']}" if item["photo_name"] else "Photo: Not attached"
            approval = "Approved" if item["approved"] else "Needs Review"

            lines.extend(
                [
                    f"{index}. Severity: {item['severity']} ({approval})",
                    location,
                    f"Finding: {item['finding']}",
                    f"Recommendation: {item['recommendation']}",
                    photo,
                    "",
                ]
            )

    return "\n".join(lines).strip()


def render_report_item(item, index):
    status = "Approved" if item["approved"] else "Needs Review"
    location = f" - {item['location']}" if item["location"] else ""
    title = f"{index}. {item['section']}{location} | {item['severity']} | {status}"

    with st.expander(title, expanded=not item["approved"]):
        st.markdown(f"<span class='hia-pill'>{item['mode']}</span>", unsafe_allow_html=True)
        st.markdown("**Finding**")
        new_finding = st.text_area(
            "Finding",
            value=item["finding"],
            key=f"finding_{index}",
            label_visibility="collapsed",
        )

        st.markdown("**Recommendation**")
        new_recommendation = st.text_area(
            "Recommendation",
            value=item["recommendation"],
            key=f"recommendation_{index}",
            label_visibility="collapsed",
        )

        edit_cols = st.columns([1, 1, 1])
        with edit_cols[0]:
            new_severity = st.selectbox(
                "Severity",
                SEVERITY_OPTIONS,
                index=SEVERITY_OPTIONS.index(item["severity"])
                if item["severity"] in SEVERITY_OPTIONS
                else 0,
                key=f"severity_{index}",
            )
        with edit_cols[1]:
            new_section = st.selectbox(
                "Section",
                REPORT_SECTIONS,
                index=REPORT_SECTIONS.index(item["section"])
                if item["section"] in REPORT_SECTIONS
                else 0,
                key=f"section_{index}",
            )
        with edit_cols[2]:
            new_location = st.text_input("Location", value=item["location"], key=f"location_{index}")

        button_cols = st.columns([1, 1, 4])
        with button_cols[0]:
            if st.button("Save", key=f"save_{index}"):
                item["finding"] = new_finding
                item["recommendation"] = new_recommendation
                item["severity"] = new_severity
                item["section"] = new_section
                item["location"] = new_location
                st.success("Saved.")
        with button_cols[1]:
            approved_label = "Unapprove" if item["approved"] else "Approve"
            if st.button(approved_label, key=f"approve_{index}"):
                item["approved"] = not item["approved"]
                st.rerun()


initialize_state()

st.sidebar.header("Inspection Details")
inspection_info = {
    "inspector_name": st.sidebar.text_input("Inspector Name"),
    "property_address": st.sidebar.text_input("Property Address"),
    "client_name": st.sidebar.text_input("Client Name"),
    "inspection_date": st.sidebar.date_input("Inspection Date", date.today()),
}

st.sidebar.divider()
st.sidebar.header("Report Settings")
draft_mode = st.sidebar.radio(
    "Drafting Mode",
    ["Exact Voice Notes", "Comment Library", "Full AI Draft"],
    help="This controls how new notes are converted into report items.",
)
approved_only_export = st.sidebar.toggle("Export approved items only", value=False)

st.sidebar.divider()
st.sidebar.header("AI Services")
ai_provider = st.sidebar.selectbox(
    "AI Provider",
    ["Ollama Local", "OpenAI Backup"],
    help="Use Ollama/Gemma4 for the demo. OpenAI remains available as a cloud backup.",
)
ollama_url = st.sidebar.text_input("Ollama URL", value="http://localhost:11434")
ollama_rewrite_model = st.sidebar.text_input("Ollama Rewrite Model", value="gemma4:latest")
ollama_vision_model = st.sidebar.text_input("Ollama Vision Model", value="gemma4:latest")
local_speech_model = st.sidebar.selectbox(
    "Local Speech Model",
    ["base.en", "small.en", "tiny.en"],
    help="base.en is the recommended demo balance of speed and accuracy on CPU.",
)
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    value=os.getenv("OPENAI_API_KEY", ""),
    type="password",
    help="Only used when AI Provider is set to OpenAI Backup.",
)
transcription_model = st.sidebar.selectbox(
    "OpenAI Speech Model",
    ["gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"],
)
rewrite_model = st.sidebar.text_input("OpenAI Rewrite Model", value="gpt-4o-mini")
use_ai_rewrite = st.sidebar.toggle("Use AI professional rewrite", value=True)

st.sidebar.divider()
if st.sidebar.button("Clear Report", use_container_width=True):
    st.session_state.report_items = []
    st.success("Report cleared.")

st.markdown(
    """
    <div class="hia-hero">
        <h1>Home Inspection AI</h1>
        <p>Capture notes, organize findings, review AI-assisted language, and build a client-ready report while you stay in control.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

total_items = len(st.session_state.report_items)
approved_items = sum(1 for item in st.session_state.report_items if item["approved"])
needs_review = total_items - approved_items

metric_cols = st.columns(3)
metric_cols[0].metric("Report Items", total_items)
metric_cols[1].metric("Approved", approved_items)
metric_cols[2].metric("Needs Review", needs_review)

capture_tab, review_tab, report_tab, camera_tab = st.tabs(
    ["Capture", "Review", "Report", "AI Camera"]
)

with capture_tab:
    left_col, right_col = st.columns([0.95, 1.05])

    with left_col:
        st.subheader("Add Inspection Note")
        section = st.selectbox("Inspection Section", REPORT_SECTIONS)
        location = st.text_input("Location", placeholder="Example: rear bedroom, attic hatch, north exterior wall")
        severity = st.selectbox("Initial Severity", SEVERITY_OPTIONS, index=1)

        st.markdown("**Voice Note**")
        audio_note = st.audio_input("Record inspection note")
        if st.button("Transcribe Voice Note"):
            if audio_note is None:
                st.warning("Record a voice note first.")
            elif ai_provider == "OpenAI Backup" and not openai_api_key:
                st.warning("Add an OpenAI API key in the sidebar to use OpenAI speech-to-text.")
            else:
                with st.spinner("Transcribing voice note..."):
                    try:
                        if ai_provider == "Ollama Local":
                            transcript = transcribe_audio_local(audio_note, local_speech_model)
                        else:
                            transcript = transcribe_audio_openai(audio_note, openai_api_key, transcription_model)
                        st.session_state.current_note = transcript
                        st.session_state.last_transcript = transcript
                        st.success("Voice note transcribed.")
                    except Exception as error:
                        st.error(f"Transcription failed: {error}")

        if st.session_state.last_transcript:
            st.caption("Last transcript is loaded into the note box below.")

        raw_note = st.text_area(
            "Inspector Note",
            placeholder="Example: vertical crack at drywall above right side of window, no active moisture visible.",
            height=155,
            key="current_note",
        )
        photo = st.file_uploader("Attach Photo", type=["jpg", "jpeg", "png"])

        if st.button("Add Draft Item", type="primary"):
            if not raw_note.strip():
                st.warning("Enter an inspection note first.")
            else:
                photo_name = photo.name if photo else ""
                if use_ai_rewrite and ai_provider == "Ollama Local":
                    with st.spinner("Rewriting note with Ollama/Gemma4..."):
                        try:
                            report_item = rewrite_note_with_ollama(
                                raw_note,
                                section,
                                location,
                                severity,
                                ollama_url,
                                ollama_rewrite_model,
                            )
                            report_item["photo_name"] = photo_name
                            st.session_state.report_items.append(report_item)
                            st.success("Ollama rewritten item added for review.")
                        except Exception as error:
                            st.error(f"Ollama rewrite failed: {error}")
                elif use_ai_rewrite and ai_provider == "OpenAI Backup" and openai_api_key:
                    with st.spinner("Rewriting note into professional report language..."):
                        try:
                            report_item = rewrite_note_with_openai(
                                raw_note,
                                section,
                                location,
                                severity,
                                openai_api_key,
                                rewrite_model,
                            )
                            report_item["photo_name"] = photo_name
                            st.session_state.report_items.append(report_item)
                            st.success("AI rewritten item added for review.")
                        except Exception as error:
                            st.error(f"AI rewrite failed: {error}")
                else:
                    st.session_state.report_items.append(
                        draft_report_item(section, raw_note, severity, draft_mode, location, photo_name)
                    )
                    st.success("Draft item added for review.")

    with right_col:
        st.subheader("How This Drafting Mode Works")
        if draft_mode == "Exact Voice Notes":
            st.markdown(
                """
                <div class="hia-callout">
                    New items keep your note language almost exactly as entered. This is best when you want the report to reflect your field wording.
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif draft_mode == "Comment Library":
            st.markdown(
                """
                <div class="hia-callout">
                    New items check a starter comment library for common keywords such as crack, leak, outlet, roof, and gutter.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="hia-callout">
                    New items are rewritten into a more complete AI-style draft. The demo default uses local Ollama/Gemma4, with OpenAI available as a backup.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("**Starter Comment Library**")
        for keyword, comment in COMMENT_LIBRARY.items():
            st.markdown(
                f"""
                <div class="hia-report-block">
                    <strong>{keyword.title()}</strong><br>
                    <span class="hia-small">{comment['finding']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

with review_tab:
    st.subheader("Review Draft Items")
    if not st.session_state.report_items:
        st.info("No draft items yet. Add one from the Capture tab.")
    else:
        for item_index, report_item in enumerate(st.session_state.report_items):
            render_report_item(report_item, item_index + 1)

with report_tab:
    st.subheader("Generated Report")
    report_text = build_report_text(
        inspection_info,
        st.session_state.report_items,
        approved_only=approved_only_export,
    )
    st.text_area("Copyable Report", report_text, height=420)

    report_file = BytesIO(report_text.encode("utf-8"))
    st.download_button(
        "Download Report Text",
        data=report_file,
        file_name="home_inspection_report.txt",
        mime="text/plain",
    )

with camera_tab:
    st.subheader("AI Camera Prototype")
    st.markdown(
        """
        <div class="hia-callout">
            This is the future camera workflow: capture an image frame, ask a vision model what it sees, then convert the answer into a report item after inspector approval.
        </div>
        """,
        unsafe_allow_html=True,
    )

    camera_image = st.camera_input("Capture Field Photo")
    camera_prompt = st.text_area(
        "Question for AI",
        value="Describe visible inspection concerns. Do not diagnose beyond the image. Suggest what the inspector should verify.",
        height=110,
    )

    if st.button("Analyze Frame"):
        if camera_image is None:
            st.warning("Capture a camera image first.")
        else:
            if ai_provider != "Ollama Local":
                st.warning("Set AI Provider to Ollama Local to use Gemma4 vision analysis.")
            else:
                with st.spinner("Analyzing image with Ollama/Gemma4..."):
                    try:
                        st.session_state.last_ai_camera_note = analyze_image_with_ollama(
                            camera_image,
                            camera_prompt,
                            ollama_url,
                            ollama_vision_model,
                        )
                        st.info(st.session_state.last_ai_camera_note)
                    except Exception as error:
                        st.error(f"Ollama vision analysis failed: {error}")

    if st.session_state.last_ai_camera_note:
        if st.button("Send AI Camera Note to Capture"):
            st.session_state.report_items.append(
                draft_report_item(
                    "Other",
                    st.session_state.last_ai_camera_note,
                    "Further Evaluation",
                    "Full AI Draft",
                    "Camera capture",
                    "camera_capture.png",
                )
            )
            st.success("AI camera note added for review.")
