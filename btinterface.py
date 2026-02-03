# import gradio as gr
# import requests
# import json

# # Your FastAPI endpoint (change to your real URL/port)
# API_URL = "http://localhost:8007/bt_res_mant_wdf_radio"   # or https://your-domain.com/bt_res

# def call_your_api(exist_name_1, exist_Pinuse_1, exist_status_1, exist_op1_1, exist_op2_1,
#                   exist_name_2, exist_Pinuse_2, exist_status_2, exist_op1_2, exist_op2_2,
#                   ee_input, threeuk_input,required_freq, required_mimo,
#                   ):
#     try:
#         payload = {
#             "exist_name_1": exist_name_1,
#             "exist_Pinuse_1": exist_Pinuse_1,
#             "exist_status_1": exist_status_1,
#             "exist_op1_1": exist_op1_1,
#             "exist_op2_1": exist_op2_1,
#             "exist_name_2": exist_name_2,
#             "exist_Pinuse_2": exist_Pinuse_2,
#             "exist_status_2": exist_status_2,
#             "exist_op1_2": exist_op1_2,
#             "exist_op2_2": exist_op2_2,
#             "required_mimo": required_mimo,
#             "required_freq": required_freq,
#             "end_of_ee": ee_input,
#             "end_of_3uk": threeuk_input
#         }

#         response = requests.post(API_URL, json=payload, timeout=500)

#         if response.status_code != 200:
#             return f"API Error: {response.status_code}\n{response.text}"

#         result = response.json()

#         # Format nicely
#         data = result.get("data", {})
#         proposal = data.get("proposal", "—")
#         antenna_sel = data.get("antenna_selection", "—")
#         reason = data.get("reason", "—")

#         output_text = f"""
# **Proposal**  
# {proposal}

# **Antenna Selection**  
# {antenna_sel}

# **Reason**  
# {reason}
#         """.strip()

#         return output_text

#     except Exception as e:
#         return f"Error calling API: {str(e)}"


# # ── Gradio Interface ────────────────────────────────────────
# with gr.Blocks(theme=gr.themes.Soft()) as demo:
#     gr.Markdown("# Antenna + Radio Proposal Tool")

#     with gr.Row():
#         with gr.Column():
#             exist_name_1 = gr.Textbox(label="Antenna name 1", placeholder="name")
#             exist_Pinuse_1 = gr.Textbox(label="Ports in use", placeholder="integer")
#             exist_status_1 = gr.Textbox(label="Status", placeholder="Shared or Unilateral")
#             exist_op1_1 = gr.Textbox(label="Operator1 Frequecy", placeholder="MHz")
#             exist_op2_1 = gr.Textbox(label="Operator2 Frequecy", placeholder="MHz")

#         with gr.Column():
#             exist_name_2 = gr.Textbox(label="Antenna name 2", placeholder="name")
#             exist_Pinuse_2 = gr.Textbox(label="Ports in use", placeholder="integer")
#             exist_status_2 = gr.Textbox(label="Status", placeholder="Shared or Unilateral")
#             exist_op1_2 = gr.Textbox(label="Operator1 Frequecy", placeholder="MHz")
#             exist_op2_2 = gr.Textbox(label="Operator2 Frequecy", placeholder="MHz")

#     with gr.Row():
#         ee_input = gr.Textbox(label="End_of_EE", placeholder="null or date")
#         threeuk_input = gr.Textbox(label="End_of_3UK", placeholder="null or date")
        
    
#     with gr.Row():
#         required_freq = gr.Textbox(label="Required Frequency", placeholder="MHz")
#         required_mimo = gr.Textbox(label="Required MIMO", placeholder="2x4")

#     btn = gr.Button("Get Proposal", variant="primary", size="lg")

#     output = gr.Markdown()

#     btn.click(
#         fn=call_your_api,
#         inputs=[
#             exist_name_1, exist_Pinuse_1, exist_status_1, exist_op1_1, exist_op2_1,
#             exist_name_2, exist_Pinuse_2, exist_status_2, exist_op1_2, exist_op2_2,
#             ee_input, threeuk_input, required_freq, required_mimo
#         ],
#         outputs=output
#     )

# # Launch
# demo.launch(
#     server_name="0.0.0.0",
#     server_port=7860,
#     share=True,               # ← creates public link (good for testing)
#     debug=True
# )


import gradio as gr
import pandas as pd
import requests
import json
import os

# Your API endpoint
API_URL = "http://localhost:8007/bt_res_mant_wdf_radio"

def process_excel_and_call_api(excel_file, site_id):
    if not excel_file:
        return "**Error**: Please upload an Excel file first."

    if not site_id.strip():
        return "**Error**: Please enter a Site ID."

    try:
        # Read Excel - force column names since your file has messy/incomplete header
        df = pd.read_excel(excel_file.name, header=None, dtype=str)
        
        # Assign the exact column names you expect
        df.columns = [
            'id', 'exist_name_1', 'exist_Pinuse_1', 'exist_status_1', 'exist_op1_1',
            'exist_op2_1', 'exist_name_2', 'exist_Pinuse_2', 'exist_status_2',
            'exist_op1_2', 'exist_op2_2', 'required_mimo', 'required_freq',
            'end_of_ee', 'end_of_3uk', 'site_type', 'dep_env', 'fencing', 'exist_cabin',
            'num_ers', 'mimo', 'avail_depth', 'avail_width', 'avail_height'
        ]

        # Clean up: strip spaces, handle empty as ''
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df = df.fillna('')

        # Convert id to string for safe matching
        df['id'] = df['id'].astype(str).str.strip()
        site_id = str(site_id).strip()

        # Find the row
        row = df[df['id'] == site_id]
        if row.empty:
            return f"**Error**: Site ID '{site_id}' not found in the Excel file.\n\nAvailable IDs: {', '.join(df['id'].unique())}"

        row = row.iloc[0]

        params = {
            "exist_name_1": row['exist_name_1'],
            "exist_Pinuse_1": row['exist_Pinuse_1'],
            "exist_status_1": row['exist_status_1'],
            "exist_op1_1": row['exist_op1_1'],
            "exist_op2_1": row['exist_op2_1'],
            "exist_name_2": row['exist_name_2'],
            "exist_Pinuse_2": row['exist_Pinuse_2'],
            "exist_status_2": row['exist_status_2'],
            "exist_op1_2": row['exist_op1_2'],
            "exist_op2_2": row['exist_op2_2'],
            "required_mimo": row['required_mimo'],
            "required_freq": row['required_freq'],
            "end_of_ee": row['end_of_ee'],
            "end_of_3uk": row['end_of_3uk'],
            "site_type": row['site_type'],
            "dep_env": row['dep_env'],
            "fencing": row['fencing'],
            "exist_cabin": row['exist_cabin'],
            "num_ers": row['num_ers'],
            "mimo": row['mimo'],
            "avail_depth": row['avail_depth'],
            "avail_width": row['avail_width'],
            "avail_height": row['avail_height']
        }

        # Call your API
        response = requests.post(API_URL, json=params, timeout=360)
        response.raise_for_status()

        result = response.json()
        data = result.get("data", {})

        proposal = data.get("proposal", "—")
        antenna_sel = data.get("antenna_selection", "—")
        full_reason_text = data.get("reason", "—") # Get the raw reason text from the API

        antenna_reason_parsed = "—"
        radio_reason_parsed = "—"
        cabinet_reason_parsed = "—"

        if full_reason_text != "—":
            prefixes = {
                "antenna": "For antenna: ",
                "radio":   "For radio: ",
                "cabinet": "For cabinet: "
            }

            positions = {key: full_reason_text.find(prefix) for key, prefix in prefixes.items()}
            positions = {k: v for k, v in positions.items() if v != -1}  # only existing ones

            # Sort by appearance order
            ordered_keys = sorted(positions, key=positions.get)

            for i, key in enumerate(ordered_keys):
                start = positions[key]
                prefix_len = len(prefixes[key])
                end = positions[ordered_keys[i+1]] if i+1 < len(ordered_keys) else None

                text = full_reason_text[start + prefix_len : end].strip()
                if text.endswith('.'):
                    text = text[:-1].strip()

                if key == "antenna":
                    antenna_reason_parsed = text or "—"
                elif key == "radio":
                    radio_reason_parsed = text or "—"
                elif key == "cabinet":
                    cabinet_reason_parsed = text or "—"

        return f"""
**Site ID**: {site_id}

**Proposal**  
{proposal}

**Antenna Selection**  
{antenna_sel}

**Full Reason**  
**Antenna**: {antenna_reason_parsed}  
**Radio**: {radio_reason_parsed}
**Cabinet**: {cabinet_reason_parsed}
        """.strip()
    except Exception as e:
        return f"**Processing Error**\n{str(e)}"

# ── Gradio Interface ───────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Antenna + Radio Proposal Tool (Excel Upload)")

    with gr.Row():
        excel_upload = gr.File(
            label="Upload Excel File (.xlsx or .xls)",
            file_types=[".xlsx", ".xls"],
            type="filepath"   # important: gives file path
        )

    site_id_input = gr.Textbox(
        label="Site ID",
        placeholder="e.g. 10001",
        info="Enter the ID from the 'id' column"
    )

    btn = gr.Button("Get Proposal", variant="primary", size="lg")

    output = gr.Markdown()

    # When button clicked → process file + site id
    btn.click(
        fn=process_excel_and_call_api,
        inputs=[excel_upload, site_id_input],
        outputs=output
    )

# Launch
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,          # temporary public link for testing
    debug=True
)