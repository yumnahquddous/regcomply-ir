import os
from pathlib import Path
from dotenv import load_dotenv

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

load_dotenv()

# Specific file paths
PROCESSED_CORPUS_PATH = DATA_DIR / "processed" / "processed_corpus.csv"
EMBEDDINGS_PATH = DATA_DIR / "processed" / "document_embeddings.npy"
TXT_DOCS_DIR = DATA_DIR / "raw" / "StandartizedRegulatoryDocumentsTXT"

# Model settings
MODEL_NAME = "intfloat/multilingual-e5-base"

# OpenRouter Settings
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# # Using the exact model that won the RIRAG-2025 Shared Task
# LLM_MODEL_NAME = "z-ai/glm-4.5-air:free"

# === GEMINI CONFIGURATION (ACTIVE) ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Using Gemini 1.5 Flash: Blazing fast, highly accurate, and perfect for RAG
LLM_MODEL_NAME = "gemini-2.5-flash"

DOCUMENT_MAPPING = {
    1: "AML_VER09.211223.txt",
    2: "COBS_VER15.150823.txt",
    3: "FUNDS_VER08.040723.txt",
    4: "MKT_VER08.181223.txt",
    5: "GEN_VER08.181223.txt",
    6: "MIR_VER07.181223.txt",
    7: "PRU_VER13.181223.txt",
    8: "GLO_VER19.181223.txt",
    9: "FEES_VER16.181223.txt",
    10: "IFR_VER07.181223.txt",
    11: "VER_VER01.110319.txt", # Placeholder if matching FP
    12: "FP_VER01.110319.txt",
    13: "PIN_VER05.181223.txt",
    14: "CIB_VER04.030220.txt",
    15: "CMC_VER03.270922.txt",
    16: "CONF_VER03.18042019.txt",
    17: "FSMR (Consolidated_December 2023).txt",
    18: "BRR Regulations (December 2018).txt",
    19: "Draft Guidance - FSRA Guiding Principles for Virtual Assets Regulation and Supervision (IA).txt",
    20: "Guidance - Digital Securities Offerings and Virtual  Assets under the Financial Services and Markets Regulations_240220.txt",
    21: "Guidance  Regulation of Digital Securities Activities in ADGM_240224.txt",
    22: "Guidance - Spot Commodities Activities in ADGM (VER02.181223).txt",  # Loose structural match variations
    23: "Guidance - Virtual Asset Activities in ADGM (VER05.181223).txt",
    24: "Guidance Framework for Fund Managers of Venture Capital Funds (VER03.181223).txt",
    25: "Guidance_Regulatory Framework for PFP and Multilateral Trading Facilities dealing with Private Capital Markets (VER02.181223).txt",
    26: "Supplementary Guidance OTCLPs (VER02.181223).txt",
    27: "Supplementary Guidance  Authorisation of Digital Investment Management (Robo-advisory) Activities.txt",
    28: "FinTech RegLab Guidance_VER01.31082016.txt",
    29: "API - Guidance Note_Final 14 October 2019 Eng.txt",
    30: "Guidance - Private Credit Funds_VER01.040523.txt",
    31: "Sustainable Finance Supplementary Guidance_VER01.040723.txt",
    32: "Environmental Social and Governance Disclosures Guidance_VER01.040723.txt",
    33: "Guidance - Continuous Disclosure_VER01.280922.txt",
    34: "Guidance - Disclosure Requirements for Petroleum Reporting Entities_VER01.280922.txt",
    35: "Guidance - Disclosure Requirements for Mining Reporting Entities_VER01.280922.txt",
    36: "ADGM_Guidance_-_Application_of_English_Laws.txt",
    37: "SFWG_Guidance on Principles for the Effective Management of Climate-related Financial Risks.txt",
    38: "CRS Regulations 2017 (Consolidated_October 2023) v6.txt",
    39: "UAE_CRS_Guidance_Notes_17 June 2020 (002).txt",
    40: "Foreign Tax Account Compliance Regulations 2022.txt"
}