# antenna_selection_api.py
# Full standalone version with PDF text extraction
# You can run this file directly or import functions from it

import os
import sys
import uuid
import json
import time
import logging
import re
from datetime import datetime
from contextlib import contextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# ────────────────────────────────────────────────
# Logging setup
# ────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("antenna-api")

REAL_LOGS_ENABLED     = True
TERMINAL_LOGS_ENABLED = True

# ────────────────────────────────────────────────
# Constants / Paths
# ────────────────────────────────────────────────

DOCUMENTS_DIR = "documents/"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)


# ────────────────────────────────────────────────
# PDF text extraction function
# ────────────────────────────────────────────────

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file using PyPDF2."""
    text = ""
    try:
        import PyPDF2  # import here or at top — your choice
        
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page_text = reader.pages[page_num].extract_text()
                if page_text:
                    text += page_text + "\n"
    except FileNotFoundError:
        logger.error(f"PDF file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error extracting text from PDF '{file_path}': {e}")
        raise Exception(f"Error extracting text from PDF '{file_path}': {e}")
    
    return text.strip()


# ────────────────────────────────────────────────
# Timing context manager
# ────────────────────────────────────────────────

@contextmanager
def time_it(description: str):
    """Context manager to log execution time of a block."""
    start_time = time.perf_counter()
    logger.info(f"TIMING: START  '{description}'")
    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        logger.info(f"TIMING: FINISH '{description}' → {duration_ms:8.2f} ms")


# ────────────────────────────────────────────────
# FastAPI application
# ────────────────────────────────────────────────

app = FastAPI(
    title="Antenna & Radio Planning API",
    description="Endpoint for antenna selection and site equipment proposal",
    version="0.1.0"
)


@app.post("/bt_res_mant_wdf_radio")
async def bt_res_mant_wdf_radio(request: dict):
    """
    Choosing one antenna from multiple
    """
     
    req_id = str(uuid.uuid4())
    logger.info(f"Req ID: {req_id} --- /bt_res_mant [POST] endpoint: START ---")
    with time_it(f"Request ID: {req_id} Total /bt_res_mant request procssing"):
        try:
            logger.info(f"Request ID: {req_id} --- START PAYLOAD --- \n {json.dumps(request, indent=2)}\n --- END PAYLOAD ---")
            
            site_type = request.get('site_type')
            dep_env = request.get('dep_env')
            fencing = request.get('fencing')
            exist_cabin = request.get('exist_cabin')
            num_ers = request.get('num_ers')
            avail_spaces = request.get('avail_spaces')
            
            exist_name_1 = request.get('exist_name_1')
            exist_Pinuse_1 = request.get('exist_Pinuse_1')
            exist_status_1 = request.get('exist_status_1')
            exist_op1_1 = request.get('exist_op1_1')
            exist_op2_1 = request.get('exist_op2_1')
            
            if exist_name_1 != "":
                
                det_1 = extract_text_from_pdf(DOCUMENTS_DIR + exist_name_1 + ".pdf")
                
                
                get_antenna_details= f"""
                    You are an information-extraction agent specialized in telecom antenna specifications.

                    You will receive unstructured text containing technical details of an antenna.

                    Input text:
                    {det_1}

                    Your task is to accurately extract and normalize the following parameters only from the given text:

                    frequency

                    hpbw

                    length

                    total_ports

                    weight

                    Extraction & Formatting Rules

                    1. Frequency

                    Extract all operating frequency ranges.

                    Format exactly as:
                    start-end(Ports)

                    Multiple ranges must be comma-separated.

                    Example format:
                    694-960(4 Ports), 1427-2690(4 Ports), 1695-2690(8 Ports)

                    2. HPBW

                    Horizontal beamwidth value must be either 65 or 85 only.

                    Output as an integer (no degree symbol).

                    3. Length

                    Extract antenna length.

                    Convert to meters if given in mm or cm.

                    Output as a decimal value (e.g., 1.59, 1.99).

                    4. Total Ports

                    Extract the total number of ports.

                    Output as a single integer value.

                    5. Weight

                    Extract antenna weight.

                    If weight is in lbs, convert it to kg (1 lb = 0.453592 kg).

                    Output only the weight value in kg, rounded to two decimals if needed.

                    Output Rules
                    Only provide the extracted parameters like HPBW: 65, Length: 1.59.
                    Do not include explanations or extra text.
                    If any parameter is missing, return null for that field.
                """
                
                gemini_api_key = "AIzaSyCMpaP6FiXENsPDkDU8f7G-gg1NsUs5rXk"
                gemini_direct_url = (
                    "https://generativelanguage.googleapis.com/v1beta/"
                    "models/gemini-2.5-flash:generateContent"
                    f"?key={gemini_api_key}"
                )

                direct_api_payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": get_antenna_details
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.1,
                        "topP": 0.9,
                        "maxOutputTokens": 65536
                    },
                    "safetySettings": [
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                    ]
                }

                async with httpx.AsyncClient(timeout=180.0) as client:
                    gemini_response = await client.post(
                        gemini_direct_url,
                        json=direct_api_payload,
                        headers={"Content-Type": "application/json"}
                    )

                if gemini_response.status_code != 200:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Gemini API error: {gemini_response.text}"
                    )

                gemini_json = gemini_response.json()

                try:
                    antenna_det_1 = (
                        gemini_json["candidates"][0]
                        ["content"]["parts"][0]["text"]
                        .strip()
                    )
                    
                    logger.info(f"output of antenna det: {antenna_det_1.strip()}")
                except (KeyError, IndexError):
                    raise HTTPException(
                        status_code=500,
                        detail="Invalid Gemini response format"
                    )
                
                exist_antenna_1 = f"""
                                Name: {exist_name_1}
                                {antenna_det_1.strip()}
                                Ports_in_use: {exist_Pinuse_1}
                                Status: {exist_status_1}
                                Operator_1: {exist_op1_1}
                                Operator_2: {exist_op2_1}
                            """
                            
            if exist_name_1 == "":
                exist_antenna_1 = ""
            
            exist_name_2 = request.get('exist_name_2')
            exist_Pinuse_2 = request.get('exist_Pinuse_2')
            exist_status_2 = request.get('exist_status_2')
            exist_op1_2 = request.get('exist_op1_2')
            exist_op2_2 = request.get('exist_op2_2')
            
            if exist_name_2 != "":
                det_2 = extract_text_from_pdf(DOCUMENTS_DIR + exist_name_2 + ".pdf")
            
            
                get_antenna_details= f"""
                    You are an information-extraction agent specialized in telecom antenna specifications.

                    You will receive unstructured text containing technical details of an antenna.

                    Input text:
                    {det_2}

                    Your task is to accurately extract and normalize the following parameters only from the given text:

                    frequency

                    hpbw

                    length

                    total_ports

                    weight

                    Extraction & Formatting Rules

                    1. Frequency

                    Extract all operating frequency ranges.

                    Format exactly as:
                    start-end(Ports)

                    Multiple ranges must be comma-separated.

                    Example format:
                    694-960(4 Ports), 1427-2690(4 Ports), 1695-2690(8 Ports)

                    2. HPBW

                    Horizontal beamwidth value must be either 65 or 85 only.

                    Output as an integer (no degree symbol).

                    3. Length

                    Extract antenna length.

                    Convert to meters if given in mm or cm.

                    Output as a decimal value (e.g., 1.59, 1.99).

                    4. Total Ports

                    Extract the total number of ports.

                    Output as a single integer value.

                    5. Weight

                    Extract antenna weight.

                    If weight is in lbs, convert it to kg (1 lb = 0.453592 kg).

                    Output only the weight value in kg, rounded to two decimals if needed.

                    Output Rules
                    Only provide the extracted parameters like HPBW: 65, Length: 1.59.
                    Do not include explanations or extra text.
                    If any parameter is missing, return null for that field.
                """
                
                gemini_api_key = "AIzaSyCMpaP6FiXENsPDkDU8f7G-gg1NsUs5rXk"
                gemini_direct_url = (
                    "https://generativelanguage.googleapis.com/v1beta/"
                    "models/gemini-2.5-flash:generateContent"
                    f"?key={gemini_api_key}"
                )

                direct_api_payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": get_antenna_details
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.1,
                        "topP": 0.9,
                        "maxOutputTokens": 65536
                    },
                    "safetySettings": [
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                    ]
                }

                async with httpx.AsyncClient(timeout=180.0) as client:
                    gemini_response = await client.post(
                        gemini_direct_url,
                        json=direct_api_payload,
                        headers={"Content-Type": "application/json"}
                    )

                if gemini_response.status_code != 200:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Gemini API error: {gemini_response.text}"
                    )

                gemini_json = gemini_response.json()

                try:
                    antenna_det_2 = (
                        gemini_json["candidates"][0]
                        ["content"]["parts"][0]["text"]
                        .strip()
                    )
                    
                    logger.info(f"output of antenna det: {antenna_det_2.strip()}")
                except (KeyError, IndexError):
                    raise HTTPException(
                        status_code=500,
                        detail="Invalid Gemini response format"
                    )
                
                
                exist_antenna_2 = f"""
                                Name: {exist_name_2}
                                {antenna_det_2.strip()}
                                Ports_in_use: {exist_Pinuse_2}
                                Status: {exist_status_2}
                                Operator_1: {exist_op1_2}
                                Operator_2: {exist_op2_2}
                            """
                            
            if exist_name_2 == "":
                exist_antenna_2 = ""
                
                          
            req_mimo = request.get('required_mimo')
            req_freq = request.get('required_freq')
            End_of_EE = request.get('end_of_ee')
            End_of_3UK = request.get('end_of_3uk')
            
            
            P_1 = f"""
                You are an AI agent tasked with deciding which of two antennas (Antenna A and Antenna B) should be selected for upgrade based on specific requirements and guidelines. You will receive the following input data:

                Details of Antenna A: {exist_antenna_1}

                Details of Antenna B: {exist_antenna_2}
                Upgrade requirements:
                req_freq: {req_freq}
                req_mimo: {req_mimo}


                Follow these decision rules exactly in order:
                Case 1: Both antennas have status "unilateral" and Operator_2 is null

                If req_freq is 3500:
                If req_mimo is "3500@32x32", return "A" 
                If req_mimo is "3500@8x8" or req_mimo is "3500@32x32 and other frequencies":
                Choose the antenna with the higher weight
                If weights are equal, choose the one with more type of frequencies it is running and return the respective letter.
                If the frequencies are also equal, return the letter A.


                If req_freq is not 3500:
                Choose the antenna with the higher weight
                If weights are equal, choose the one with more type of frequencies it is running and return the respective letter.
                If the frequencies are also equal, return the letter A.

                Case 2: One antenna has status "unilateral" with operator_2 as null and the other has "shared"

                Return the letter A or B of antenna with status "unilateral"

                Case 3: One antenna has status "unilateral" with operator_1 as null and the other has "shared"

                Return the letter A or B of antenna with status "shared"

                Case 4: Both antennas have status "shared"

                choose the antenna which has less number of frequencies for operator_2. For example, if antenna A has operator_2 as "2100@2x2" and antenna B has operator_2 as "800@2x2 and 1800@2x2", then choose antenna A.
                But if in case both antennas have same number of frequencies for operator_2, then choose the antenna which has less number of Total_Ports.
                If both antennas have same number of frequencies for operator_2 and same number of Total_Ports, then return the antenna which is heavier.
                
                Case 5: Both antennas have status "unilateral" with one antenna with operator_1 as null and other with operator_2 as null
                
                Return the letter A or B of antenna with operator_2 as null
                
                Case 6: One antenna fields are empty, that means, on the site there is only 1 antenna, so you will return the letter of other antenna.
                
                For any other case, not covered just return -1.
                
                Your response must explicitly be only a single character: "A", "B", or "-1".
            """
            
            gemini_api_key = "AIzaSyCMpaP6FiXENsPDkDU8f7G-gg1NsUs5rXk"
            gemini_direct_url = (
                "https://generativelanguage.googleapis.com/v1beta/"
                "models/gemini-2.5-flash:generateContent"
                f"?key={gemini_api_key}"
            )

            direct_api_payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": P_1
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "topP": 0.9,
                    "maxOutputTokens": 65536
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            }

            async with httpx.AsyncClient(timeout=180.0) as client:
                gemini_response = await client.post(
                    gemini_direct_url,
                    json=direct_api_payload,
                    headers={"Content-Type": "application/json"}
                )

            if gemini_response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Gemini API error: {gemini_response.text}"
                )

            gemini_json = gemini_response.json()

            try:
                summary = (
                    gemini_json["candidates"][0]
                    ["content"]["parts"][0]["text"]
                    .strip()
                )
            except (KeyError, IndexError):
                raise HTTPException(
                    status_code=500,
                    detail="Invalid Gemini response format"
                )

            response_data = {
                "summary": summary.strip()
            }
            
            logger.info(f"output of antenna selection: {summary.strip()}")
            
            P_2 = f"""
                You are a Telecom RF Planning AI agent responsible for determining whether Operator-2 
                can be consolidated onto a single existing antenna from two available antennas by using 
                only current antenna capabilities. You must collect all frequencies and MIMO requirements
                used by Operator-2 across both antennas and evaluate if either antenna alone can support
                them unilaterally without exceeding port limits. Ignore Operator-1 entirely. Apply 
                strict port-sharing rules: 700 and 800 MHz may run together on the same ports only if 
                the MIMO configuration is identical (700@2x2 with 800@2x2 uses 2 ports; 700@2x4 with 
                800@2x4 uses 4 ports). Similarly, 1800 and 2100 MHz may share ports only when MIMO 
                matches (2x2 with 2x2 or 4x4 with 4x4). Port sharing is allowed only if the antenna 
                supports both frequencies and required MIMO. Swapping or consolidation is permitted only 
                if both antennas have the same HPBW. Do not downgrade MIMO, do not propose new antennas, 
                and do not partially consolidate. Allowed MIMO limits are: 700/800 → 2x2 or 2x4; 
                1800/2100 → 2x2 or 4x4; 2600 → 2x2 or 4x4; 3500 → 8x8 or 32x32. If one antenna can 
                fully support all Operator-2 requirements, return the other antenna's number (1 or 2); 
                otherwise return -9 only. If both antennas can support Operator-2 unilaterally, return the
                antenna number (1 or 2) where disruption of operator_1 is minimized (i.e., where operator_1 frequencies are more in list and if both antennas are running same number of frequencies in the list just return 1)
                
                Your output must explicitly only be a single character which can be: "1" , "2" or "-9".
                No explanation to give at all.
                
                existing_antenna_1: {exist_antenna_1}
                existing_antenna_2: {exist_antenna_2}
                
                Also, if the status of both the antenna is "unilateral", then just return -9.
            """
            
            gemini_api_key = "AIzaSyCMpaP6FiXENsPDkDU8f7G-gg1NsUs5rXk"
            gemini_direct_url = (
                "https://generativelanguage.googleapis.com/v1beta/"
                "models/gemini-2.5-flash:generateContent"
                f"?key={gemini_api_key}"
            )

            direct_api_payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": P_2
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "topP": 0.9,
                    "maxOutputTokens": 65536
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            }

            async with httpx.AsyncClient(timeout=180.0) as client:
                gemini_response = await client.post(
                    gemini_direct_url,
                    json=direct_api_payload,
                    headers={"Content-Type": "application/json"}
                )

            if gemini_response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Gemini API error: {gemini_response.text}"
                )

            gemini_json = gemini_response.json()

            try:
                p2_res = (
                    gemini_json["candidates"][0]
                    ["content"]["parts"][0]["text"]
                    .strip()
                )
                
                logger.info(f"output of operator_2 consolidation: {p2_res.strip()}")
            except (KeyError, IndexError):
                raise HTTPException(
                    status_code=500,
                    detail="Invalid Gemini response format"
                )
            
            Antenna_Options = [
                                """
                                    Name: RVV65B-C3-3XR
                                    Frequency: 694-960(2 Ports), 1695-2690(4 Ports)
                                    HPBW: 65
                                    Length: 1.85
                                    Total_Ports: 6
                                    Weight: 23kg
                                """,
                                """
                                    Name: RZVV-65B-R4-V4
                                    Frequency: 694-960(2 Ports),1427-2690(2 Ports), 1695-2690(4 Ports)
                                    HPBW: 65
                                    Length: 2.0
                                    Total_Ports: 8
                                    Weight: 22.8kg
                                """,
                                """
                                    Name: RZVV-65B-R4-V3
                                    Frequency: 694-960(2 Ports),1427-2690(2 Ports), 1695-2690(4 Ports)
                                    HPBW: 65
                                    Length: 2.0
                                    Total_Ports: 8
                                    Weight: 22.8kg
                                """,
                                """
                                    Name: RRVV-65B-R4-V4
                                    Frequency: 694-960(4 Ports), 1695-2690(4 Ports)
                                    HPBW: 65
                                    Length: 1.828
                                    Total_Ports: 8
                                    Weight: 35.5kg
                                """,
                                """
                                    Name: RRZZVV65BR6N43
                                    Frequency: 694-960(4 Ports),1427-2690(4 Ports), 1695-2690(4 Ports)
                                    HPBW: 65
                                    Length: 2.1
                                    Total_Ports: 12
                                    Weight: 35.6kg
                                """,
                                """
                                    Name: RRZZHHTT-65B-R6H4
                                    Frequency: 694-960(4 Ports),1427-2690(4 Ports), 1695-2690(4 Ports), 2490-2690(4 Ports)
                                    HPBW: 65
                                    Length: 2.1
                                    Total_Ports: 16
                                    Weight: 42.5kg
                                """,
                                """
                                    Name: RRZZV4-65B-R8H4
                                    Frequency: 694-960(4 Ports),1427-2690(4 Ports), 1695-2690(8 Ports)
                                    HPBW: 65
                                    Length: 2.1
                                    Total_Ports: 16
                                    Weight: 42.9kg
                                """,
                                """
                                    Name: RRZZHHTTS4-65B-R7
                                    Frequency: 694-960(4 Ports),2490-2690(4 Ports), 1695-2180(4 Ports), 1427-2690(4 Ports), 3300-3800(8 Ports)
                                    HPBW: 65
                                    Length: 2.1
                                    Total_Ports: 24
                                    Weight: 47kg
                                """,
                                """
                                    Name: AIR3218
                                    Frequency: 694-960(4 Ports),1427-2690(4 Ports), 1695-2690(4 Ports), 3500(32 Ports)
                                    HPBW: 65
                                    Length: 2
                                    Total_Ports: 44
                                    Weight: 59.5kg
                                """,
                                """
                                    Name: AIR3268
                                    Frequency: 3500(32 Ports)
                                    HPBW: 65
                                    Length: 1
                                    Total_Ports: 32
                                    Weight: 12kg
                                """
                                
                            ]
            
            Prompt = "Empty"
            
            
            if summary.strip() == "-1":
                Prompt = """Just return "Invalid input" as the response."""
            
            if summary.strip() == "A" and p2_res.strip() == "-9":
                Prompt = f"""
                You are a radio-network antenna planning expert. Your task is to determine whether the existing antenna can support the requested upgrade or whether a new antenna must be proposed from the provided Antenna_Options. Your decision must strictly follow the rules below and be deterministic.

                        Requirement input format will be like 70/80@2x4, that means frequency required is 700 and 800 MHz and MIMO is 2x4 and also if the existing is 700@2x2 and requirement is of 800@2x4, then it means we have to replace 700@2x2 with 700@2x4 and 800@2x4 as 700@2x2 will become obsolete. Similarly with 1800 and 2100 MHz frequencies.
                        
                        700 / 800 MHz → 2x2 or 2x4

                        1800 / 2100 MHz → 2x2 or 4x4

                        2600 MHz → 2x2 or 4x4

                        3500 MHz → 8x8 or 32x32           
                        
                        Evaluate whether the existing antenna 1 supports the required frequency band. If the frequency is not supported, the existing antenna 1 is not suitable and an upgrade is required.
                        
                        If the current frequency is 700@2x2 and the requirement is of 800@2x2, then no need to upgrade antenna, then the proposed antenna will be the existing one and vice-versa.
                        Similarly, if current is 700@2x4 and required is 800@2x4 and vice-versa.
                        Likewise, if current is 1800@2x2 and required is 2100@2x2 and vice-versa.
                        Finally, if current is 1800@4x4 and required is 2100@4x4 and vice-versa.
                        
                        Also, if in the existing config, if an operator is given with 1800@2x2,2100@2x2, that means it is only using 2 ports to run both 1800 and 2100. Similarly with 700@2x2,800@2x2 or 700@2x4,800@2x4 or 1800@4x4,2100@4x4.
                        The 2600 MHz frequency will run seperately from 1800 and 2100 frequency, it will not be shared with them.
                        From the required_mimo, extract the second number, which represents the required number of ports for that frequency (for example, 2x4 → 4 ports, 4x4 → 4 ports, 8x8 → 8 ports). Check whether the existing antenna 1 has enough free ports for the required frequency after accounting for current usage.

                        Operator separation is mandatory:

                        Only Operator 1 configurations may be combined or modified.

                        Operator 2 must always remain separate, with its existing frequency and MIMO configuration unchanged.


                        Any proposed antenna (existing or upgraded) must retain Operator 2 independently and must not reuse or merge Operator 2 ports with Operator 1.

                        If the existing antenna 1 supports the required frequency, respects the frequency–MIMO rules, has sufficient free ports, and preserves Operator 2 separation, return the existing antenna 1 as the final result.

                        If an upgrade is required, evaluate antennas from Antenna_Options and select only those that:

                        Support the required frequency

                        Support the required MIMO configuration

                        Provide at least the required number of ports

                        Support all existing operator configurations

                        Maintain strict separation of Operator 2

                        Have the same HPBW as the existing antenna 1

                        If multiple antennas satisfy the above conditions, apply the following preference order:

                        Prefer antennas that provide the most efficient port utilization across all their specified frequency bands, meaning the fewest total unallocated ports across all current (Operator 1, Operator 2) and required configurations (required_freq, required_mimo). This includes minimizing completely unused frequency sub-bands or segments.

                        If unavoidable, allow extra ports only when no exact-match option exists

                        If the requirement is of 3500@32x32 then consider the following:
                            ---
                                
                                if the status of both the antenna is unilateral and both have frequencies for operator_1:
                                    then propose 2 antennas, one will be AIR3268 and the other will be choosed from antenna_options and for that you have to consider frequencies of existing operator_1 on both antenna and also the required frequency.
                                Else you have to propose the AIR3218 only.
                            ---
    
                        Choose the lightest antenna

                        Prefer antenna length closest to 2 meters
                        
                        One more important thing, if there is only antenna on the site, and it has status: Shared, we always allocate 2 ports that supports 700/800 MHz and 4 ports that supports 1800/2100/2600 MHz for operator 2 even though operator 2 is not using those frequencies currently. Basically, for future use of operator_2. So consider this also while proposing antenna and if not able to find an antenna which is suitable then drop the required configuration for 700 or 800 only to 2x2 and not 2x4. And in the reasong do write 'To refer to operator for step down'.

                        Do not invent specifications, and do not assume missing data.
                        
                        If the existing antenna to be used then in proposed antenna must write: Existing antenna 'antenna_name'.
                        ***Firstly check if we have values for both antennas. If yes, then in your response, firstly write ""Antenna 1 is selected for the upgrade and no swapping was done."" and only tell the proposed antenna and give a concise summary as why you proposed that antenna according to guidelines provided.If no, then in your response, firstly write ""There is only 1 antenna and no swapping was done.""and only tell the proposed antenna and give a concise summary as why you proposed that antenna according to guidelines provided.***
            
                        Existing Antenna 1: {exist_antenna_1}
                        Existing Antenna 2: {exist_antenna_2}
                        
                        
                        ***Do Not consider frequency of both the operator from Antenna 2 for upgrading Antenna.***
                        
                        Antenna_Option: {Antenna_Options}
                        
                        Upgrade_requirement: 'freq_upgrade': {req_freq}, 'required_mimo': {req_mimo}
                        
                        
                                       
***CRITICAL RESPONSE FORMAT INSTRUCTIONS***
Your entire response MUST be a valid JSON object with exactly these four fields (no additional text, no markdown, no explanations outside the JSON):

{{
  "Antenna selection": "<the required prefix sentence, e.g., 'Antenna 1 is selected for the upgrade and no swapping was done.' or the corresponding one for the case>",
  "Proposed antenna": "<the proposed antenna name, e.g., 'Existing antenna APXVLL4_65-C-A20' or the new model>",
  "reason": "<a concise summary explaining why this antenna was proposed, strictly according to the guidelines>"
  "requirement": "<full requirement frequencies along with their technologies only for operator 1. Existing+Required>"
}}


Do not include any text before or after the JSON. Do not use markdown. Do not escape the JSON. Output only the raw JSON object.
                """
                
            if summary.strip() == "B" and p2_res.strip() == "-9":
                Prompt = f"""
                You are a radio-network antenna planning expert. Your task is to determine whether the existing antenna can support the requested upgrade or whether a new antenna must be proposed from the provided Antenna_Options. Your decision must strictly follow the rules below and be deterministic.

                        
                        Requirement input format will be like 70/80@2x4, that means frequency required is 700 and 800 MHz and MIMO is 2x4 and also if the existing is 700@2x2 and requirement is of 800@2x4, then it means we have to replace 700@2x2 with 700@2x4 and 800@2x4 as 700@2x2 will become obsolete. Similarly with 1800 and 2100 MHz frequencies.
                        
                        700 / 800 MHz → 2x2 or 2x4

                        1800 / 2100 MHz → 2x2 or 4x4

                        2600 MHz → 2x2 or 4x4

                        3500 MHz → 8x8 or 32x32
                        
                        
                        First, evaluate whether the existing antenna 2 supports the required frequency band. If the frequency is not supported, the existing antenna 2 is not suitable and an upgrade is required.
                        
                        If the current frequency is 700@2x2 and the requirement is of 800@2x2, then no need to upgrade antenna, then the proposed antenna will be the existing one and vice-versa.
                        Similarly, if current is 700@2x4 and required is 800@2x4 and vice-versa.
                        Likewise, if current is 1800@2x2 and required is 2100@2x2 and vice-versa.
                        Finally, if current is 1800@4x4 and required is 2100@4x4 and vice-versa.
                        
                        Also, if in the existing config, if an operator is given with 1800@2x2,2100@2x2, that means it is only using 2 ports to run both 1800 and 2100. Similarly with 700@2x2,800@2x2 or 700@2x4,800@2x4 or 1800@4x4,2100@4x4.
                        The 2600 MHz frequency will run seperately from 1800 and 2100 frequency, it will not be shared with them.
                        From the required_mimo, extract the second number, which represents the required number of ports for that frequency (for example, 2x4 → 4 ports, 4x4 → 4 ports, 8x8 → 8 ports). Check whether the existing antenna 2 has enough free ports for the required frequency after accounting for current usage.

                        Operator separation is mandatory:

                        Only Operator 1 configurations may be combined or modified.

                        Operator 2 must always remain separate, with its existing frequency and MIMO configuration unchanged.

                        Any proposed antenna (existing or upgraded) must retain Operator 2 independently and must not reuse or merge Operator 2 ports with Operator 1.

                        If the existing antenna 2 supports the required frequency, respects the frequency–MIMO rules, has sufficient free ports, and preserves Operator 2 separation, return the existing antenna 2 as the final result.

                        If an upgrade is required, evaluate antennas from Antenna_Options and select only those that:

                        Support the required frequency

                        Support the required MIMO configuration

                        Provide at least the required number of ports

                        Support all existing operator configurations

                        Maintain strict separation of Operator 2

                        Have the same HPBW as the existing antenna 2

                        If multiple antennas satisfy the above conditions, apply the following preference order:

                        Prefer antennas that provide the most efficient port utilization across all their specified frequency bands, meaning the fewest total unallocated ports across all current (Operator 1, Operator 2) and required configurations (required_freq, required_mimo). This includes minimizing completely unused frequency sub-bands or segments.

                        If unavoidable, allow extra ports only when no exact-match option exists

                        If the requirement is of 3500@32x32 then consider the following:
                            ---
                                
                                if the status of both the antenna is unilateral and both have frequencies for operator_1:
                                    then propose 2 antennas, one will be AIR3268 and the other will be choosed from antenna_options and for that you have to consider frequencies of existing operator_1 on both antenna and also the required frequency.
                                Else you have to propose the AIR3218 only.
                            ---
    
                        Choose the lightest antenna

                        Prefer antenna length closest to 2 meters
                        
                        One more important thing, if there is only antenna on the site, and it has status: Shared, we always allocate 2 ports that supports 700/800 MHz and 4 ports that supports 1800/2100/2600 MHz for operator 2 even though operator 2 is not using those frequencies currently. Basically, for future use of operator_2. So consider this also while proposing antenna and if not able to find an antenna which is suitable then drop the required configuration for 700 or 800 only to 2x2 and not 2x4. And in the reasong do write 'To refer to operator for step down'.

                        Do not invent specifications, and do not assume missing data.
                        
                        If the existing antenna to be used then in proposed antenna must write: Existing antenna 'antenna_name'.
                        ***Firstly check if we have values for both antennas. If yes, then in your response, firstly write ""Antenna 2 is selected for the upgrade and no swapping was done."" and only tell the proposed antenna and give a concise summary as why you proposed that antenna according to guidelines provided.If no, then in your response, firstly write ""There is only 1 antenna and no swapping was done.""and only tell the proposed antenna and give a concise summary as why you proposed that antenna according to guidelines provided.***
                        
                        Existing Antenna 1: {exist_antenna_1}
                        Existing Antenna 2: {exist_antenna_2}
                        
                        ***Do Not consider frequency of both the operator from Antenna 2 for upgrading Antenna.***
                        
                        
                        Antenna_Option: {Antenna_Options}
                        
                        Upgrade_requirement: 'freq_upgrade': {req_freq}, 'required_mimo': {req_mimo}
                                 
***CRITICAL RESPONSE FORMAT INSTRUCTIONS***
Your entire response MUST be a valid JSON object with exactly these four fields (no additional text, no markdown, no explanations outside the JSON):

{{
  "Antenna selection": "<the required prefix sentence, e.g., 'Antenna 1 is selected for the upgrade and no swapping was done.' or the corresponding one for the case>",
  "Proposed antenna": "<the proposed antenna name, e.g., 'Existing antenna APXVLL4_65-C-A20' or the new model>",
  "reason": "<a concise summary explaining why this antenna was proposed, strictly according to the guidelines>"
  "requirement": "<full requirement frequencies along with their technologies only for operator 1. Existing+Required>"
}}

Do not include any text before or after the JSON. Do not use markdown. Do not escape the JSON. Output only the raw JSON object.
                """
                
            if p2_res.strip() == "1":
                Prompt= f"""
                
                You are a radio-network antenna planning expert. Your task is to determine whether the existing antenna can support the requested upgrade or whether a new antenna must be proposed from the provided Antenna_Options. Your decision must strictly follow the rules below and be deterministic.

                        Requirement input format will be like 70/80@2x4, that means frequency required is 700 and 800 MHz and MIMO is 2x4 and also if the existing is 700@2x2 and requirement is of 800@2x4, then it means we have to replace 700@2x2 with 700@2x4 and 800@2x4 as 700@2x2 will become obsolete. Similarly with 1800 and 2100 MHz frequencies.
                        
                        700 / 800 MHz → 2x2 or 2x4

                        1800 / 2100 MHz → 2x2 or 4x4

                        2600 MHz → 2x2 or 4x4

                        3500 MHz → 8x8 or 32x32           
                        
                        Evaluate whether the existing antenna 1 supports the required frequency band and also the frequency of antenna 2's operator_1. If the frequency is not supported, the existing antenna 1 is not suitable and an upgrade is required.
                        
                        If the current frequency is 700@2x2 and the requirement is of 800@2x2, then no need to upgrade antenna, then the proposed antenna will be the existing one and vice-versa.
                        Similarly, if current is 700@2x4 and required is 800@2x4 and vice-versa.
                        Likewise, if current is 1800@2x2 and required is 2100@2x2 and vice-versa.
                        Finally, if current is 1800@4x4 and required is 2100@4x4 and vice-versa.
                        
                        Also, if in the existing config, if an operator is given with 1800@2x2,2100@2x2, that means it is only using 2 ports to run both 1800 and 2100. Similarly with 700@2x2,800@2x2 or 700@2x4,800@2x4 or 1800@4x4,2100@4x4.
                        The 2600 MHz frequency will run seperately from 1800 and 2100 frequency, it will not be shared with them.
                        From the required_mimo and mimo of existing_antenna_2 for operator_1, extract the second number, which represents the required number of ports for that frequency (for example, 2x4 → 4 ports, 4x4 → 4 ports, 8x8 → 8 ports). Check whether the existing antenna 1 has enough free ports for the required frequency after accounting for current usage.

                        Now in antenna 1, operator_2 is obsolete. We can totally ignore it's frequencies. We only focus on operator_1 frequencies and required frequencies.
                        

                        If the existing antenna 1 supports the required frequency, respects the frequency–MIMO rules, has sufficient free ports, and also accomodates antenna 2's operator_1 specs, return the existing antenna 1 as the final result.

                        If an upgrade is required, evaluate antennas from Antenna_Options and select only those that:

                        Support the required frequency

                        Support the required MIMO configuration

                        Provide at least the required number of ports

                        Support all existing operator configurations

                        Have the same HPBW as the existing antenna 1

                        If multiple antennas satisfy the above conditions, apply the following preference order:

                        Prefer antennas that provide the most efficient port utilization across all their specified frequency bands, meaning the fewest total unallocated ports across all current and required configurations (required_freq, required_mimo). This includes minimizing completely unused frequency sub-bands or segments.

                        If unavoidable, allow extra ports only when no exact-match option exists

                        Choose the lightest antenna

                        Prefer antenna length closest to 2 meters
                        
                        One more important thing, if there is only antenna on the site, and it has status: Shared, we always allocate 2 ports that supports 700/800 MHz and 4 ports that supports 1800/2100/2600 MHz for operator 2 even though operator 2 is not using those frequencies currently. Basically, for future use of operator_2. So consider this also while proposing antenna and if not able to find an antenna which is suitable then drop the required configuration for 700 or 800 only to 2x2 and not 2x4. And in the reasong do write 'To refer to operator for step down'.

                        Return only one antenna (either the existing antenna 1 or one upgraded antenna). Do not suggest multiple antennas, do not invent specifications, and do not assume missing data.
                        
                        
                        If the existing antenna to be used then in proposed antenna must write: Existing antenna 'antenna_name'.
                        ***In your response, firstly write ""Antenna 1 is selected for the upgrade and Antenna 2 will be configured to provide to Operator_2 unilaterally."" and only tell the proposed antenna and give a concise summary as why you proposed that antenna according to guidelines provided.***
            
                        Existing Antenna 1: {exist_antenna_1}
                        Existing Antenna 2: {exist_antenna_2}
                        
                        Antenna_Option: {Antenna_Options}
                        
                        Upgrade_requirement: 'freq_upgrade': {req_freq}, 'required_mimo': {req_mimo}
                                                
                                              
***CRITICAL RESPONSE FORMAT INSTRUCTIONS***
Your entire response MUST be a valid JSON object with exactly these four fields (no additional text, no markdown, no explanations outside the JSON):

{{
  "Antenna selection": "<the required prefix sentence, e.g., 'Antenna 1 is selected for the upgrade and no swapping was done.' or the corresponding one for the case>",
  "Proposed antenna": "<the proposed antenna name, e.g., 'Existing antenna APXVLL4_65-C-A20' or the new model>",
  "reason": "<a concise summary explaining why this antenna was proposed, strictly according to the guidelines>"
  "requirement": "<full requirement frequencies along with their technologies only for operator 1. Existing+Required>"
}}

Do not include any text before or after the JSON. Do not use markdown. Do not escape the JSON. Output only the raw JSON object.
                
                """
                
            if p2_res.strip() == "2":
                Prompt= f"""
                
                You are a radio-network antenna planning expert. Your task is to determine whether the existing antenna can support the requested upgrade or whether a new antenna must be proposed from the provided Antenna_Options. Your decision must strictly follow the rules below and be deterministic.

                        Requirement input format will be like 70/80@2x4, that means frequency required is 700 and 800 MHz and MIMO is 2x4 and also if the existing is 700@2x2 and requirement is of 800@2x4, then it means we have to replace 700@2x2 with 700@2x4 and 800@2x4 as 700@2x2 will become obsolete. Similarly with 1800 and 2100 MHz frequencies.
                        
                        700 / 800 MHz → 2x2 or 2x4

                        1800 / 2100 MHz → 2x2 or 4x4

                        2600 MHz → 2x2 or 4x4

                        3500 MHz → 8x8 or 32x32           
                        
                        Evaluate whether the existing antenna 2 supports the required frequency band and also the frequency of antenna 1's operator_1. If the frequency is not supported, the existing antenna 2 is not suitable and an upgrade is required.
                        
                        If the current frequency is 700@2x2 and the requirement is of 800@2x2, then no need to upgrade antenna, then the proposed antenna will be the existing one and vice-versa.
                        Similarly, if current is 700@2x4 and required is 800@2x4 and vice-versa.
                        Likewise, if current is 1800@2x2 and required is 2100@2x2 and vice-versa.
                        Finally, if current is 1800@4x4 and required is 2100@4x4 and vice-versa.
                        
                        Also, if in the existing config, if an operator is given with 1800@2x2,2100@2x2, that means it is only using 2 ports to run both 1800 and 2100. Similarly with 700@2x2,800@2x2 or 700@2x4,800@2x4 or 1800@4x4,2100@4x4.
                        The 2600 MHz frequency will run seperately from 1800 and 2100 frequency, it will not be shared with them.
                        From the required_mimo and mimo of existing_antenna_1 for operator_1, extract the second number, which represents the required number of ports for that frequency (for example, 2x4 → 4 ports, 4x4 → 4 ports, 8x8 → 8 ports). Check whether the existing antenna 1 has enough free ports for the required frequency after accounting for current usage.

                        Now in antenna 2, operator_2 is obsolete. We can totally ignore it's frequencies. We only focus on operator_1 frequencies and required frequencies.
                        

                        If the existing antenna 2 supports the required frequency, respects the frequency–MIMO rules, has sufficient free ports, and also accomodates antenna 1's operator_1 specs, return the existing antenna 2 as the final result.

                        If an upgrade is required, evaluate antennas from Antenna_Options and select only those that:

                        Support the required frequency

                        Support the required MIMO configuration

                        Provide at least the required number of ports

                        Support all existing operator configurations

                        Have the same HPBW as the existing antenna 2

                        If multiple antennas satisfy the above conditions, apply the following preference order:

                        Prefer antennas that provide the most efficient port utilization across all their specified frequency bands, meaning the fewest total unallocated ports across all current and required configurations (required_freq, required_mimo). This includes minimizing completely unused frequency sub-bands or segments.

                        If unavoidable, allow extra ports only when no exact-match option exists

                        Choose the lightest antenna

                        Prefer antenna length closest to 2 meters
                        
                        One more important thing, if there is only antenna on the site, and it has status: Shared, we always allocate 2 ports that supports 700/800 MHz and 4 ports that supports 1800/2100/2600 MHz for operator 2 even though operator 2 is not using those frequencies currently. Basically, for future use of operator_2. So consider this also while proposing antenna and if not able to find an antenna which is suitable then drop the required configuration for 700 or 800 only to 2x2 and not 2x4. And in the reasong do write 'To refer to operator for step down'.

                        Return only one antenna (either the existing antenna 2 or one upgraded antenna). Do not suggest multiple antennas, do not invent specifications, and do not assume missing data.
                        
                        
                        in the proposed requirement do write the full requirement frequencies along with their technologies.
                        
                        If the existing antenna to be used then in proposed antenna must write: Existing antenna 'antenna_name'.
                        ***In your response, firstly write ""Antenna 2 is selected for the upgrade and Antenna 1 will be configured to provide to Operator_2 unilaterally."" and only tell the proposed antenna and give a concise summary as why you proposed that antenna according to guidelines provided.***
            
                        Existing Antenna 1: {exist_antenna_1}
                        Existing Antenna 2: {exist_antenna_2}
                        
                        Antenna_Option: {Antenna_Options}
                        
                        Upgrade_requirement: 'freq_upgrade': {req_freq}, 'required_mimo': {req_mimo}
                                                
***CRITICAL RESPONSE FORMAT INSTRUCTIONS***
Your entire response MUST be a valid JSON object with exactly these four fields (no additional text, no markdown, no explanations outside the JSON):

{{
  "Antenna selection": "<the required prefix sentence, e.g., 'Antenna 1 is selected for the upgrade and no swapping was done.' or the corresponding one for the case>",
  "Proposed antenna": "<the proposed antenna name, e.g., 'Existing antenna APXVLL4_65-C-A20' or the new model>",
  "reason": "<a concise summary explaining why this antenna was proposed, strictly according to the guidelines>"
  "requirement": "<full requirement frequencies along with their technologies only for operator 1. Existing+Required>"
}}

Do not include any text before or after the JSON. Do not use markdown. Do not escape the JSON. Output only the raw JSON object.
                
                """
                
                
            gemini_api_key = "AIzaSyCMpaP6FiXENsPDkDU8f7G-gg1NsUs5rXk"
            gemini_direct_url = (
                "https://generativelanguage.googleapis.com/v1beta/"
                "models/gemini-2.5-flash:generateContent"
                f"?key={gemini_api_key}"
            )

            direct_api_payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": Prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.5,
                    "topP": 0.9,
                    "maxOutputTokens": 65536
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            }

            async with httpx.AsyncClient(timeout=180.0) as client:
                gemini_response = await client.post(
                    gemini_direct_url,
                    json=direct_api_payload,
                    headers={"Content-Type": "application/json"}
                )

            if gemini_response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Gemini API error: {gemini_response.text}"
                )

            gemini_json = gemini_response.json()

            try:
                gemini_text = (
                    gemini_json["candidates"][0]
                    ["content"]["parts"][0]["text"]
                    .strip()
                )
            except (KeyError, IndexError):
                raise HTTPException(
                    status_code=500,
                    detail="Invalid Gemini response format"
                )

            # Parse the JSON output from Gemini
            try:
                gemini_text = gemini_text.strip()
                gemini_text = re.sub(r'^```(?:json)?\s*', '', gemini_text, flags=re.IGNORECASE | re.MULTILINE)
                gemini_text = re.sub(r'\s*```$', '', gemini_text)
                gemini_text = gemini_text.strip()
                ai_response = json.loads(gemini_text)
                # Validate required fields
                required_keys = {"Antenna selection", "Proposed antenna", "reason", "requirement"}
                if not required_keys.issubset(ai_response.keys()):
                    raise ValueError("Missing required keys")
            except json.JSONDecodeError:
                # Fallback: if not valid JSON, wrap the raw text (for debugging)
                ai_response = {
                    "Antenna selection": "Error: Invalid response format",
                    "Proposed antenna": "None",
                    "reason": f"Gemini returned non-JSON: {gemini_text[:500]}",
                    "requirement": "N/A"
                }
            except ValueError:
                ai_response = {
                    "Antenna selection": "Error: Incomplete response",
                    "Proposed antenna": "None",
                    "reason": "Response missing required JSON fields",
                    "requirement": "N/A"
                }

            response_data = {
                "response": ai_response  # Now a dict, not a string
            }
            
            logger.info(f"AI Response: {response_data}")

            req = ai_response["requirement"]
            
            Prompt_w_BB = f"""
                You are a telecom AI agent specialized in proposing baseband deployments for cellular sites based on the frequencies. Your goal is to analyze the explanation describing the proposed antenna and recommend the appropriate basebands from the available options: BB6621, BB6631, BB6655, and BB6672.
                Follow these strict rules for proposals, based on project guidelines:

                BB6621 is deployed specifically for GSM1800(gsm 18). It must always be paired with either BB6655 or BB6672; never deploy it alone.
                BB6631 is deployed only when the demand is exclusively for 1800MHz (18) and/or 2100MHz (21) frequencies, with no other demands present. In this case, deploy it as a single baseband.
                BB6655 is deployed if there is demand for any of the following frequencies: 700MHz (70), 800MHz (80), 1800MHz (18), 2100MHz (21), or 2600MHz (26), or any combination of these. It can handle these demands standalone or in combination.
                BB6672 is deployed if there is demand for 3500MHz with either 32T32R (@3232) or 8T8R (@88) configurations, and this must be accompanied by any additional demands that would otherwise require BB6655 (i.e., 700/800/1800/2100/2600MHz).

                2G = GSM (1800)
                3G = UMTS (2100)
                4G = LTE (700/800/1800/2100/2600)
                5G = NR (700/800/1800/2100/2600/3500)
                
                6621 Will support only 2G (1800) and will only be deployed when the input is of gsm:18 with other tech also. So, 6621 will be paired either with 6655 or 6672.
                6631 will support 2G(1800) +4G(2100) But only if 1800 & 2100 Bands are required and no other frequency is there. Also, when 1800 and 2100 are present individually.
                6655 will support 4G all bands  (70/80/18/21/26)
                6672 will only be deployed when the input has nr:35


                General deployment patterns:
                If the input demands do not match any rules, respond by asking for clarification or stating that no matching baseband proposal is possible.

                Input format: The user will provide a description of the site's frequencies with tech, e.g., "frequencies: gsm18, lte18/21, lte70/80, lte,nr70/80/18/21/26/35 etc."

                Output format:
                Your output baseband can only be one of the following:
                    if required frequencies are only 1800MHz(18) and/or 2100MHz(21) or both 18/21 → propose BB6631
                    if required frequencies is gsm1800(gsm18) with other frequencies then pair it as below:
                    if required frequencies include 700MHz(70), 800MHz(80), 1800MHz(18), 2100MHz(21), and/or 2600MHz(26) → propose BB6655 + B6621
                    if required frequencies include 3500MHz nr and/or any of 700MHz(70), 800MHz(80), 1800MHz(18), 2100MHz(21), and/or 2600MHz(26) → propose BB6672 + BB6621
                    if gsm18 is not in input then don't propose BB6621
                List the proposed basebands.
                If multiple basebands are proposed, specify the combination (e.g., "Deploy BB6621 + BB6655").
                Do not provide any explanations or justifications in your response.

                Always respond logically, concisely, and professionally. Do not propose basebands outside the four options or violate the pairing rules.
                You have to infer frequencies from the following explanation:
                {req}
            """
            
            gemini_api_key = "AIzaSyCMpaP6FiXENsPDkDU8f7G-gg1NsUs5rXk"
            gemini_direct_url = (
                "https://generativelanguage.googleapis.com/v1beta/"
                "models/gemini-2.5-flash:generateContent"
                f"?key={gemini_api_key}"
            )

            direct_api_payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": Prompt_w_BB
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "topP": 0.9,
                    "maxOutputTokens": 65536
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            }

            async with httpx.AsyncClient(timeout=180.0) as client:
                gemini_response_bb = await client.post(
                    gemini_direct_url,
                    json=direct_api_payload,
                    headers={"Content-Type": "application/json"}
                )

            if gemini_response_bb.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Gemini API error: {gemini_response_bb.text}"
                )

            gemini_json = gemini_response_bb.json()

            try:
                res_bb = (
                    gemini_json["candidates"][0]
                    ["content"]["parts"][0]["text"]
                    .strip()
                )
                
                logger.info(f"output of operator_2 consolidation: {res_bb.strip()}")
            except (KeyError, IndexError):
                raise HTTPException(
                    status_code=500,
                    detail="Invalid Gemini response format"
                )
            

            Prompt_w_Radio = f"""
                
                You are an AI telecom radio-planning assistant. Your task is to generate a radio proposal based on the given site inputs by strictly following these rules.

                You have to infer from 'Requirements' what technology (2G, 3G, 4G, 5G) and which frequency bands are required at the site.

                Technology mapping: 2G = GSM (1899), 3G = UMTS (2100), 4G = LTE (700/800/1800/2100/2600), 5G = NR (700/800/1800/2100/2600/3500).

                Vendor rule: if the existing antenna vendor is Huawei, always propose ERS or RRU radios.

                Special rule for 3G: if the requirement is UMTS 2100, propose “Nokia Radio” instead of ERS radios.

                End-of-life rule: if end_of_ee = "{End_of_EE}" OR end_of_3uk = "{End_of_3UK}" has dates, include in the proposal: “Soft decommissioning required”.
                
                Also, if the requirement is of 700/800@2x2 and 1800/2100@4x4, then you have to propose 2 radios, one for 700/800@2x2 and another for 1800/2100@4x4. So your output will be like Radio: '2262 ERS' and '4490 ERS'.

                Only propose radio for operator 1, totally ignore operator 2.

                Radio selection must follow this logic based on band and MIMO configuration:

                700@2x2 → 2262 ERS

                800@2x2 → 2262 ERS

                700/800@2x2 → 2262 ERS

                700/800@2x4 → 4486 ERS

                1800 GSM only → 2212 ERS

                1800@2x2 with LTE + GSM → 2260 ERS

                2100@2x2 → 2260 ERS

                1800/2100@2x2 → 2260 ERS

                1800/2100@4x4 → 4490 ERS

                2600@2x2 → 4419 ERS

                2600@4x4 → 4419 ERS

                3500@8x8 → 8863 ERS

                Only choose radios from this list: 2262 ERS, 4486 ERS, 2212 ERS, 2260 ERS, 4490 ERS, 4419 ERS, 8863 ERS.

                VERY IMPORTANT - RESPONSE FORMAT:
Return **ONLY** valid JSON, nothing else. No explanations, no markdown, no extra text.
Example:
{{
  "radio": "choosen radio",                  // or "Nokia Radio" for 3G special case or can be multiple for more requirements
  "decommissioning": "yes" or "no",     // "yes" only if End_of_EE or End_of_3UK has dates
  "reason": "short reason for radio choice"
}}
                
                Input parameters:
                End_of_EE: {End_of_EE}
                End_of_3UK: {End_of_3UK}               
                Requirements: {req}
            """
            logger.info("Radio Proposal Prompt: %s", Prompt_w_Radio)
            
            gemini_api_key = "AIzaSyCMpaP6FiXENsPDkDU8f7G-gg1NsUs5rXk"
            gemini_direct_url = (
                "https://generativelanguage.googleapis.com/v1beta/"
                "models/gemini-2.5-flash:generateContent"
                f"?key={gemini_api_key}"
            )

            direct_api_payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": Prompt_w_Radio
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "topP": 0.9,
                    "maxOutputTokens": 65536
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            }

            async with httpx.AsyncClient(timeout=180.0) as client:
                gemini_response = await client.post(
                    gemini_direct_url,
                    json=direct_api_payload,
                    headers={"Content-Type": "application/json"}
                )

            if gemini_response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Gemini API error: {gemini_response.text}"
                )

            gemini_json = gemini_response.json()
            
            try:
                radio_text = gemini_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                # Aggressive cleaning
                radio_text = re.sub(r'^```(?:json)?\s*', '', radio_text, flags=re.I | re.M)
                radio_text = re.sub(r'\s*```$', '', radio_text)
                radio_text = radio_text.strip()

                # If completely empty after cleaning → explicit fallback
                if not radio_text:
                    raise ValueError("Empty response after cleaning")
                radio_json = json.loads(radio_text)

                required_radio_keys = {"radio", "decommissioning", "reason"}
                if not required_radio_keys.issubset(radio_json.keys()):
                    raise ValueError("Missing radio response keys")

            except Exception as e:
                # Fallback
                radio_json = {
                    "radio": "Unknown",
                    "decommissioning": "unknown",
                    "reason": f"Radio proposal parsing failed: {str(e)}"
                }

            prompt_cabin = f"""
                You are an expert in telecom site infrastructure planning, specialized in Ericsson cabinet families for macro sites (Greenfield, Rooftop, Streetworks / SW).

                Your task is to propose **exactly one cabinet solution** (or a named combination when explicitly allowed) that best fits the site constraints and radio requirements.

                Input variables you MUST use:

                - site_type:          {site_type}           # e.g. "Greenfield", "Rooftop", "Streetworks", "Indoor", "SW"
                - dep_env:            {dep_env}            # usually "Indoor" / "Outdoor"
                - fencing:            {fencing}            # "yes" or "no" — CRITICAL decision driver
                - exist_cabin:        {exist_cabin}        # existing cabinet name or "None"/empty
                - num_ers:            {num_ers}            # number of ERS radios that must be accommodated
                - required radios:    {radio_json['radio']}
                - available space:    {avail_spaces}

                If the dep_env is Indoor then fencing will always be yes even if not given.

                Cabinet families — propose ONLY from this list:

                INDOOR / PROTECTED CABINETS / Greenfield & Rooftop

                1. AIRI
                - Size: 2000 × 600 × 600 mm
                - Max ERS: 6
                - No PSU (uses DCDU)
                - Radios: 4490, 4480, 4419, 2460, 4486, 2212, 2260, 2262, 8863

                2. D-AIRI (Double AIRI – side-by-side)
                - Size: 2000 × 1500 × 700 mm
                - Max ERS: 15
                - Radios: same as AIRI

                3. S-AIRI (Slimline AIRI – used when width < 1500 mm)
                - Size: 2000 × 1200 × 700 mm
                - Max ERS: 15
                - Radios: same as AIRI

                OUTDOOR / EXPOSED CABINETS / Greenfield & Rooftop

                4. AIRO
                - Size: 2100 × 750 × 600 mm
                - Max ERS: 3 + BBU + PSU
                - Radios: 4490, 4480, 4419, 2460, 4486, 2212, 2260, 2262   (NO 8863)

                5. D-AIRO (Double AIRO)
                - Size: 2000 × 1800 × 700 mm
                - Max ERS: 12
                - Radios: same as AIRO

                6. S-AIRO (Slimline AIRO – when width < 1800 mm)
                - Size: 2000 × 1500 × 700 mm
                - Max ERS: 12
                - Radios: same as AIRO

                STREETWORKS 

                7. Wiltshire + E6130  → max 9 ERS, radios: 4490,4480,4486
                8. Porter             → max 6 ERS, radios: 4419,2260,2262
                9. Weston             → max 3 ERS, radios: 2260
                10. E6130 standalone  → only BBU + PSU (no radios), GF/RT outdoor, can be used when radios are on top of tower

                Decision logic — follow STRICTLY in this order:

                1. If site_type contains "Streetworks" or "SW" → MUST use Streetworks family
                - Choose Wiltshire / Porter / Weston based on radio type & num_ers
                - If all radios remote/on-tower → Wiltshire + E6130
                - Ignore fencing for Streetworks

                2. If site_type contains "Greenfield" or "GF" → Use AIRI or AIRO baseed on dep_env
                - Choose AIRI,D-AIRI,S-AIRI,AIRO,D-AIRO,S-AIRO based on radio type & num_ers
                - If all radios remote/on-tower → E6130

                3. Existing cabinet logic:
                - ONLY propose new cabinet if exist_cabin is exactly "BTS3900A" or "BTS3900L"

                4. Radio compatibility check:
                - Must do compatibility check that cabinet to be proposed must support the radio

                5. Space constraints:
                - Available space must be more than required for cabinet.
                - There must be 600mm more space for depth
                - If not available consider other cabinet

                Output format — **ONLY valid JSON**, nothing else:

                {{
                "Proposed Cabinet": "AIRI" | "D-AIRI" | "S-AIRI" | "AIRO" | "D-AIRO" | "S-AIRO" | "Wiltshire" | "Porter" | "Weston" | "Wiltshire + E6130" | "Existing: BTS3900A" | "Existing: BTS3900L" | "None - space constrained" | "Refer to site survey",
                "Reasoning": "clear step-by-step explanation why this was chosen (follow decision order)"
                }}

                Return ONLY the JSON object. No extra text, no markdown, no apologies.
                Be conservative: if space is marginal or radio compatibility unclear → output "Refer to site survey"
                
            """
            
            
                
            
            gemini_api_key = "AIzaSyCMpaP6FiXENsPDkDU8f7G-gg1NsUs5rXk"
            gemini_direct_url = (
                "https://generativelanguage.googleapis.com/v1beta/"
                "models/gemini-2.5-flash:generateContent"
                f"?key={gemini_api_key}"
            )

            direct_api_payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": prompt_cabin
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "topP": 0.9,
                    "maxOutputTokens": 65536
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            }

            async with httpx.AsyncClient(timeout=180.0) as client:
                gemini_response = await client.post(
                    gemini_direct_url,
                    json=direct_api_payload,
                    headers={"Content-Type": "application/json"}
                )

            if gemini_response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Gemini API error: {gemini_response.text}"
                )

            gemini_json = gemini_response.json()
            
            logger.info(f"Cabinet Proposal Gemini Response: {gemini_json}")

            try:
                cabin_text = gemini_json["candidates"][0]["content"]["parts"][0]["text"].strip()

                # Step 1: Remove code fences
                cabin_text = re.sub(r'^```(?:json)?\s*', '', cabin_text, flags=re.I | re.M)
                cabin_text = re.sub(r'\s*```$', '', cabin_text)

                # Step 2: Remove common unwanted Markdown that breaks JSON (**, ##, etc.)
                cabin_text = re.sub(r'\*\*(.*?)\*\*', r'\1', cabin_text)          # remove **bold**
                cabin_text = re.sub(r'^#+\s*', '', cabin_text, flags=re.M)       # remove # headers

                # Step 3: Replace literal newlines inside string values with \\n
                # This is a rough but effective fix for most cases
                def escape_newlines_in_strings(m):
                    return m.group(0).replace('\n', '\\n').replace('\r', '')
                
                cabin_text = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', escape_newlines_in_strings, cabin_text)

                cabin_text = cabin_text.strip()

                if not cabin_text:
                    raise ValueError("Empty after cleaning")

                cabin_json = json.loads(cabin_text)

                # Enforce expected keys (use your actual key names)
                required_keys = {"Proposed Cabinet", "Reasoning"}  # note: your example uses space, not underscore
                if not required_keys.issubset(cabin_json.keys()):
                    raise ValueError("Missing required keys")

                # Optional normalization
                cabinet_name = cabin_json.get("Proposed Cabinet", "None").strip()
                if cabinet_name.lower() in ["", "none", "no cabinet"]:
                    cabinet_name = "None"
                

            except Exception as e:
                # Fallback - same style as radio
                cabin_json = {
                    "proposed_cabinet": "None",
                    "reasoning": f"Failed to parse cabinet proposal from Gemini: {str(e)}",
                    "total_footprint_mm": "N/A"
                }




            # Now merge everything
            final_proposal = (
                f"Antenna: {ai_response['Proposed antenna']}   "
                f"Radio: {radio_json['radio']}"
                f"   Baseband: {res_bb}"
                f"   Cabinet: {cabin_json['Proposed Cabinet']}"
            )

            if radio_json['decommissioning'].lower() == "yes":
                final_proposal += " with soft decommissioning"

            final_reason = (
                f"For antenna: {ai_response['reason']} "
                f"For radio: {radio_json['reason']}"
                f" For cabinet: {cabin_json['Reasoning']}"
            )

            # Final structured response
            response_data = {
                "proposal": final_proposal.strip(),
                "antenna_selection": ai_response["Antenna selection"],
                "reason": final_reason.strip()
            }

            
            return JSONResponse(status_code=200, content={
                "status": "success",
                "data": response_data
            })
            
                                      
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Request ID: {req_id} Error in /summary endpoint: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")



# ────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    RELOAD = os.getenv("UVICORN_RELOAD", "true").lower() in ("true", "1", "yes")

    logger.info(f"Starting server → http://{HOST}:{PORT}  (reload={RELOAD})")

    uvicorn.run(
        "antenna_selection_api:app",
        host=HOST,
        port=PORT,
        reload=RELOAD,
        log_level="info",
    )