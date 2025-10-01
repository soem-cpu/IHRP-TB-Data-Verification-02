import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Dynamic Rule-Based Data Verification", layout="wide")

import numpy as np
from datetime import datetime

# ==================== Utility Functions ====================

def normalize(x):
    if pd.isnull(x): return ""
    return str(x).strip().replace('\u200b', '').replace('\t', '').replace('\n', '')

def is_date(val, min_date="2024-01-01", max_date=None):
    if max_date is None:
        max_date = datetime.today().strftime("%Y-%m-%d")
    try:
        date = pd.to_datetime(val, errors="coerce", dayfirst=True)
        if pd.isnull(date): return False
        if pd.Timestamp(min_date) <= date <= pd.Timestamp(max_date):
            return True
        return False
    except Exception:
        return False

def is_numeric(val, min_val=0, max_val=100):
    try:
        num = float(val)
        return min_val <= num <= max_val
    except Exception:
        return False

def extract_prefix(name):
    name = normalize(name)
    for prefix in ["Ma ", "Daw ", "Mg ", "U ", "Ko "]:
        if name.startswith(prefix):
            return prefix.strip()
    return None

def _find_col_by_names(df, names):
    """Find a matching column in df by trying multiple candidate names."""
    norm_cols = {c: str(c).strip().lower().replace(" ", "").replace("_", "") for c in df.columns}
    for cand in names:
        cand_norm = str(cand).strip().lower().replace(" ", "").replace("_", "")
        for orig, norm in norm_cols.items():
            if norm == cand_norm:
                return orig
    return None

# ==================== Dropdowns ====================
def load_dropdowns(dropdown_df):
    state_regions = set()
    state_township_dict = {}
    var_values_dict = {}
    for _, row in dropdown_df.iterrows():
        variable = normalize(row["Variable"])
        value = normalize(row["Value"])
        if variable == "State_Region":
            state_regions.add(value)
            var_values_dict.setdefault("State_Region", set()).add(value)
        elif variable in state_regions:
            state_township_dict.setdefault(variable, set()).add(value)
        else:
            var_values_dict.setdefault(variable, set()).add(value)
    return state_regions, state_township_dict, var_values_dict

# ==================== General Checks ====================

def check_list(df, column, allowed):
    """
    Check if values in the given column are in the allowed list.  Adds an error message to a new 'error' column if not.
    """
    allowed = set(normalize(i) for i in allowed)
    error_col = f"{column}_Error"
    if column in df.columns:
        df[error_col] = np.where(~df[column].apply(normalize).isin(allowed),
                                  f"Invalid value: not in allowed list",
                                  "")
    else:
        df[error_col] = "Column not found"
    return df

def check_state_region(df, allowed):
    """
    Check if 'State_Region' is in the allowed list.  Adds an error message to 'State_Region_Error' column if not.
    """
    error_col = "State_Region_Error"
    if "State_Region" in df.columns:
        df[error_col] = np.where(~df["State_Region"].isin(allowed),
                                  "Invalid State Region",
                                  "")
    else:
        df[error_col] = "Column 'State_Region' not found"
    return df

def check_township(df, state_township_dict):
    """
    Checks if the combination of 'State_Region' and 'Township' is valid.
    Adds error message to a new 'Township_Error' column if not.
    """
    error_col = "Township_Error"
    if "State_Region" not in df.columns or "Township" not in df.columns:
        df[error_col] = "Missing State_Region or Township column"
        return df

    df["State_Region_Norm"] = df["State_Region"].apply(normalize)
    df["Township_Norm"] = df["Township"].apply(normalize)

    def is_valid_township(row):
        state = row["State_Region_Norm"]
        township = row["Township_Norm"]
        return state in state_township_dict and township in state_township_dict[state]

    df[error_col] = np.where(~df.apply(is_valid_township, axis=1),
                              "Invalid State_Region/Township combination",
                              "")
    
    # Remove the temporary normalized columns
    df.drop(columns=["State_Region_Norm", "Township_Norm"], inplace=True, errors='ignore')
    return df

def check_service_delivery_point(df, df_service_point):
    """
    Checks if the 'Service_delivery_point' and 'Township' combination is valid based on df_service_point.
    Adds an error message to a new 'Service_delivery_point_Error' column if not.
    """
    error_col = "Service_delivery_point_Error"
    col_candidates = ["Service_delivery_point", "Service Delivery Point", "Service_point"]
    svc_col = next((c for c in col_candidates if c in df.columns), None)
    township_col = "Township" if "Township" in df.columns else None

    if svc_col is None or township_col is None:
        df[error_col] = "Missing Service_delivery_point or Township column"
        return df

    sp_col_candidates = ["Service_delivery_point_code", "Service delivery point code", "Service Point Code"]
    sp_code_col = next((c for c in sp_col_candidates if c in df_service_point.columns), None)
    sp_township_col = "Township" if "Township" in df_service_point.columns else None

    if sp_code_col is None or sp_township_col is None:
        df[error_col] = "Missing Service Point code or Township column in Service Point dataframe"
        return df

    allowed_pairs = set(
        (normalize(row[sp_township_col]), normalize(row[sp_code_col]))
        for _, row in df_service_point.iterrows()
        if pd.notnull(row[sp_township_col]) and pd.notnull(row[sp_code_col])
    )
    
    def is_valid_pair(row):
        township = normalize(row.get(township_col, ""))
        svc_point = normalize(row.get(svc_col, ""))
        return (township, svc_point) in allowed_pairs

    df[error_col] = np.where(~df.apply(is_valid_pair, axis=1),
                              "Invalid Service_delivery_point/Township combination",
                              "")
    return df

def check_reporting_month(df, allowed):
    return check_list(df, "Reporting_Month", allowed)

def check_screening_date(df):
    """
    Check if 'Screening_Date' is a valid date. Adds error message to 'Screening_Date_Error' if not.
    """
    error_col = "Screening_Date_Error"
    if "Screening_Date" not in df.columns:
        df[error_col] = "Column 'Screening_Date' not found"
        return df
    
    df[error_col] = df["Screening_Date"].apply(lambda val: "Invalid date" if not is_date(val, min_date="2024-01-01") else "")
    return df

def check_age_year(df):
    """
    Check if 'Age_Year' is a valid numeric value. Adds error message to 'Age_Year_Error' if not.
    """
    error_col = "Age_Year_Error"
    if "Age_Year" not in df.columns:
        df[error_col] = "Column 'Age_Year' not found"
        return df
    
    df[error_col] = df["Age_Year"].apply(lambda val: "Invalid age" if not is_numeric(val, 0, 100) else "")
    return df

def check_sex(df, var_values_dict):
    """
    Checks if the 'Sex' column contains valid values and matches the expected sex based on name prefixes.
    Adds error message to a new 'Sex_Error' column if not.
    """
    error_col = "Sex_Error"
    if "Sex" not in df.columns:
        df[error_col] = "Column 'Sex' not found"
        return df

    allowed = set(x.lower() for x in var_values_dict.get("Sex", []))

    def check_sex_validity(row):
        sex = str(row.get("Sex", "")).strip().lower()
        name = str(row.get("Name", "")).strip() if "Name" in df.columns else ""
        prefix = extract_prefix(name) if name else None

        if sex not in allowed:
            return "Invalid sex value"

        if prefix in ["Ma", "Daw"] and sex not in ["female", "f"]:
            return "Sex mismatch with name prefix (Female)"
        elif prefix in ["Mg", "U", "Ko"] and sex not in ["male", "m"]:
            return "Sex mismatch with name prefix (Male)"

        return ""  # No error

    df[error_col] = df.apply(check_sex_validity, axis=1)
    return df

def check_value_dropdown(df, column, var_values_dict):
    return check_list(df, column, var_values_dict.get(column, []))

def check_registration_number(df, allowed):
    return check_list(df, "Registration_number", allowed)

def check_registration_number_duplicate(df):
    """
    Check for duplicate registration numbers and add a flag to a new 'Duplicate_RegNum' column.
    """
    if "Registration_number" not in df.columns:
        df["Duplicate_RegNum"] = "Column 'Registration_number' not found"
        return df
    dupes = df.duplicated(subset=["Registration_number"], keep=False)
    df["Duplicate_RegNum"] = np.where(dupes, "Duplicate Registration Number", "")
    return df

def check_duplicate(df, cols):
    """
    Check for duplicate rows based on a subset of columns and add a flag to a new 'Duplicate' column.
    """
    df["Duplicate"] = ""
    for col in cols:
        if col not in df.columns:
            df["Duplicate"] = f"Column '{col}' not found"
            return df
    dupes = df.duplicated(subset=cols, keep=False)
    df["Duplicate"] = np.where(dupes, "Duplicate row", "")
    return df

def check_date_col(df, column):
    """
    Check if the values in the given column are valid dates.
    Adds an error message to a new '{column}_Error' column if not.
    """
    error_col = f"{column}_Error"
    if column not in df.columns:
        df[error_col] = f"Column '{column}' not found"
        return df
    df[error_col] = df[column].apply(lambda val: "Invalid date" if not is_date(val, min_date="2024-01-01") else "")
    return df

# ==================== Patient Sheet Specific ====================

def check_patient_data_registration_numbers(df_patient):
    """
    Check the registration numbers in the Patient Data sheet.  Adds a message to 'Registration_number_check' column.
    """
    if df_patient is None or df_patient.empty:
        df_patient = pd.DataFrame({"Error": ["Patient data sheet is empty"]})
        return df_patient
    if "Registration_number" not in df_patient.columns:
        df_patient["Registration_number_check"] = "Column 'Registration_number' not found in Patient data sheet"
        return df_patient
    results = []
    seen = set()
    for idx, reg_num in df_patient["Registration_number"].items():
        if pd.isna(reg_num) or str(reg_num).strip() == "":
            results.append("Missing Registration_number")
        else:
            reg_num_str = str(reg_num).strip()
            if reg_num_str not in seen:
                results.append("First")
                seen.add(reg_num_str)
            else:
                results.append("Repeat")
    df_patient["Registration_number_check"] = results
    return df_patient

def check_transfer_in_date(df):
    """
    Check the Transfer_in and Transfer_in_date columns.  Adds error messages to a new 'Transfer_in_Error' column.
    """
    df["Transfer_in_Error"] = ""
    if "Transfer_in" not in df.columns or "Transfer_in_date" not in df.columns:
        df["Transfer_in_Error"] = "Missing Transfer_in or Transfer_in_date column"
        return df
    min_date = datetime(2024, 1, 1)
    max_date = datetime.today()
    for idx, row in df.iterrows():
        transfer_in = str(row.get("Transfer_in", "")).strip()
        date_val = row.get("Transfer_in_date", "")
        if transfer_in in ["Yes", "Y"]:
            try:
                parsed_date = pd.to_datetime(date_val, errors="raise")
                if not (min_date <= parsed_date <= max_date):
                    df.at[idx, "Transfer_in_Error"] = "Date out of range"
            except Exception:
                df.at[idx, "Transfer_in_Error"] = "Missing or invalid date"
        else:
            if pd.notna(date_val) and str(date_val).strip() != "":
                df.at[idx, "Transfer_in_Error"] = "Date should be blank"
    return df

def check_enrolled_date(df):
    """
    Check if 'Enrolled_Date' is a valid date. Adds error message to 'Enrolled_Date_Error' if not.
    """
    df["Enrolled_Date_Error"] = ""
    if "Enrolled_Date" not in df.columns:
        df["Enrolled_Date_Error"] = "Column 'Enrolled_Date' not found"
        return df
    today = pd.Timestamp.today()
    for idx, row in df.iterrows():
        enrolled = row.get("Enrolled_Date", "")
        treatment_start = row.get("TB_Treatment_Start_Date", "")
        if pd.notnull(treatment_start):
            treatment_start_parsed = pd.to_datetime(treatment_start, errors="coerce", dayfirst=True)
            if pd.notnull(treatment_start_parsed) and treatment_start_parsed.year == 2024:
                continue
        enrolled_parsed = pd.to_datetime(enrolled, errors="coerce", dayfirst=True)
        if pd.isnull(enrolled_parsed) or not (pd.Timestamp("2024-01-01") <= enrolled_parsed <= today):
            df.at[idx, "Enrolled_Date_Error"] = "Invalid enrolled date"
    return df

def check_tb_treatment_start_date(df):
    """
    Check if 'TB_Treatment_Start_Date' is valid when 'TB_Treatment_Regimen' is present.
    Adds error message to 'TB_Treatment_Start_Date_Error' column if not.
    """
    df["TB_Treatment_Start_Date_Error"] = ""
    regimen_col = _find_col_by_names(df, [
        "TB_Treatment_Regimen", "TB Treatment Regimen", "Regimen"
    ])
    start_col = _find_col_by_names(df, [
        "TB_Treatment_Start_Date", "TB Treatment Start Date", "TB_Treatment_Start", "Start Date"
    ])
    if regimen_col is None or start_col is None:
        df["TB_Treatment_Start_Date_Error"] = "Missing TB_Treatment_Regimen or TB_Treatment_Start_Date column"
        return df
    min_date = pd.to_datetime("2024-01-01")
    max_date = pd.to_datetime(datetime.today().date())
    for idx, row in df.iterrows():
        regimen_val = row.get(regimen_col)
        if pd.isna(regimen_val) or (isinstance(regimen_val, str) and regimen_val.strip() == ""):
            continue
        start_val = row.get(start_col)
        if pd.isna(start_val) or (isinstance(start_val, str) and start_val.strip() == ""):
            df.at[idx, "TB_Treatment_Start_Date_Error"] = "Missing start date"
            continue
        parsed = pd.to_datetime(start_val, errors="coerce", dayfirst=True)
        if pd.isna(parsed):
            df.at[idx, "TB_Treatment_Start_Date_Error"] = f"Not parseable -> '{start_val}'"
            continue
        if parsed < min_date or parsed > max_date:
            df.at[idx, "TB_Treatment_Start_Date_Error"] = f"Out of range (01-Jan-2024 to {max_date.strftime('%d-%b-%Y')})"
    return df

def TB_Treatment_Outcome_Date(df):
    """
    Check if 'TB_Treatment_Outcome_Date' is valid when 'TB_Treatment_Outcome' is present.
    Adds error message to 'TB_Treatment_Outcome_Date_Error' column if not.
    """
    df["TB_Treatment_Outcome_Date_Error"] = ""
    outcome_col = _find_col_by_names(df, [
        "TB_Treatment_Outcome", "TB Treatment Outcome", "Outcome"
    ])
    outcomedate_col = _find_col_by_names(df, [
        "TB_Treatment_Outcome_Date", "TB Treatment Outcome Date", "TB_Treatment_Outcome", "Outcome Date"
    ])
    if outcome_col is None or outcomedate_col is None:
        df["TB_Treatment_Outcome_Date_Error"] = "Missing TB_Treatment_Outcome or TB_Treatment_Outcome_Date column"
        return df
    min_date = pd.to_datetime("2024-01-01")
    max_date = pd.to_datetime(datetime.today().date())
    for idx, row in df.iterrows():
        outcome_val = row.get(outcome_col)
        if pd.isna(outcome_val):
            continue
        if isinstance(outcome_val, str):
            val = outcome_val.strip().lower()
        else:
            val = ""
        if val in ["", "na", "n/a", "none", "null", "nan", "missing", "unknown"]:
            continue
        outcomedate_val = row.get(outcomedate_col)
        if pd.isna(outcomedate_val) or (isinstance(outcomedate_val, str) and outcomedate_val.strip() == ""):
            df.at[idx, "TB_Treatment_Outcome_Date_Error"] = "Missing outcome date"
            continue
        parsed = pd.to_datetime(outcomedate_val, errors="coerce", dayfirst=True)
        if pd.isna(parsed):
            df.at[idx, "TB_Treatment_Outcome_Date_Error"] = f"Not parseable -> '{outcomedate_val}'"
            continue
        if parsed < min_date or parsed > max_date:
            df.at[idx, "TB_Treatment_Outcome_Date_Error"] = f"Out of range (01-Jan-2024 to {max_date.strftime('%d-%b-%Y')})"
    return df

def check_tpt_end_date_rule(df):
    """
    Check if TPT_End_date is valid based on TPT_Start_date.
    Adds error messages to a new 'TPT_End_date_Error' column.
    """
    df["TPT_End_date_Error"] = ""
    def _find_col(df, names):
        norm_cols = {c: str(c).strip().lower().replace(" ", "").replace("_", "") for c in df.columns}
        for cand in names:
            cand_norm = cand.strip().lower().replace(" ", "").replace("_", "")
            for orig, norm in norm_cols.items():
                if norm == cand_norm:
                    return orig
        return None
    start_col = _find_col(df, ["TPT_Start_date", "TPT Start Date"])
    end_col = _find_col(df, ["TPT_End_date", "TPT End Date"])
    if start_col is None or end_col is None:
        df["TPT_End_date_Error"] = "Missing TPT_Start_date or TPT_End_date column"
        return df
    min_date = pd.to_datetime("2024-01-01")
    max_date = pd.to_datetime(datetime.today().date())
    six_months_ago = pd.Timestamp.today() - pd.DateOffset(months=6)
    for idx, row in df.iterrows():
        start_val = row.get(start_col)
        if pd.isna(start_val) or (isinstance(start_val, str) and start_val.strip() == ""):
            continue
        start_date = pd.to_datetime(start_val, errors="coerce", dayfirst=True)
        if pd.isna(start_date):
            continue
        if start_date <= six_months_ago:
            end_val = row.get(end_col)
            end_date = pd.to_datetime(end_val, errors="coerce", dayfirst=True)
            if pd.isna(end_date) or (isinstance(end_val, str) and end_val.strip() == ""):
                df.at[idx, "TPT_End_date_Error"] = "TPT_Start_date ≥ 6 months ago but TPT_End_date missing/invalid"
            elif end_date < min_date or end_date > max_date:
                df.at[idx, "TPT_End_date_Error"] = f"TPT_End_date out of allowed range (01-Jan-2024 to {max_date.strftime('%d-%b-%Y')})"
    return df

# ==================== Add Columns ====================

def add_presumptive_tb_referred(df):
    df["Presumptive_TB_referred"] = 0
    for idx, row in df.iterrows():
        has_exam = any([pd.notnull(row.get(col, None)) for col in [
            "Examination_results_Sputum", "Examination_results_Gene_Xpert", "Examination_results_Truenet", "Examination_results_CXR"
        ] if col in df.columns])
        result = normalize(row.get("Result", ""))
        if has_exam or result in ["Clinically diagnosed TB", "Bact confirmed TB"]:
            df.at[idx, "Presumptive_TB_referred"] = 1
    return df

def add_tb_detected(df):
    df["TB_Detected"] = 0
    for idx, row in df.iterrows():
        result = normalize(row.get("Result", ""))
        presumptive = row.get("Presumptive_TB_referred", 0)
        if result in ["Clinically diagnosed TB","Clinically diagnosed", "Bact confirmed", "Bact confirmed TB"] and presumptive == 1:
            df.at[idx, "TB_Detected"] = 1
    return df

def add_bact_confirmed_tb(df):
    df["Bact_confirmed_TB"] = 0
    for idx, row in df.iterrows():
        result = normalize(row.get("Result", ""))
        presumptive = row.get("Presumptive_TB_referred", 0)
        if result in ["Bact confirmed TB", "Bact confirmed"] and presumptive == 1:
            df.at[idx, "Bact_confirmed_TB"] = 1
    return df

def add_result_check(df):
    """
    Adds a 'Result_check' column to the DataFrame to verify if the 'Result' is consistent with examination results.
    """
    df["Result_check"] = "T"
    for idx, row in df.iterrows():
        sputum = str(row.get("Examination_results_Sputum", "")).strip()
        gene = str(row.get("Examination_results_Gene_Xpert", "")).strip()
        truenet = str(row.get("Examination_results_Truenet", "")).strip()
        result = str(row.get("Result", "")).strip()

        exam_positive = (
            sputum == "Positive" or
            gene in ["T", "TT", "TI", "RR"] or
            truenet in ["VT", "RR", "TI"]
        )

        if exam_positive and result not in ["Bact confirmed TB", "Bact confirmed"]:
            df.at[idx, "Result_check"] = "F"
    return df

def add_ongoing_tb_case(df_screen, df_patient):
    df_screen["Ongoing_TB_case_check"] = ""
    if "Registration_number" not in df_screen.columns or "Enrolled_Date" not in df_patient.columns or "Screening_Date" not in df_screen.columns:
        return df_screen
    patient_reg_dates = df_patient.set_index("Registration_number")["Enrolled_Date"].to_dict()
    for idx, row in df_screen.iterrows():
        reg_num = normalize(row.get("Registration_number", ""))
        screen_date = row.get("Screening_Date", "")
        enrolled_date = patient_reg_dates.get(reg_num, None)
        if enrolled_date and is_date(screen_date) and is_date(enrolled_date):
            if pd.to_datetime(screen_date) > pd.to_datetime(enrolled_date):
                df_screen.at[idx, "Ongoing_TB_case_check"] = "Ongoing TB case"
    return df_screen

def add_duplicate_check(df, cols):
    for col in cols:
        if col not in df.columns:
            df["Duplicate_check"] = f"Column '{col}' not found"
            return df
    df["Duplicate_check"] = ""
    dupes = df.duplicated(subset=cols, keep=False)
    df.loc[dupes, "Duplicate_check"] = "To recheck for duplication"
    return df

def merge_screening_columns(df_patient, df_screen):
    """
    Add calculated fields into Patient Data sheet:
      - Channel_Screening: copy 'Channel' from Screening sheet when Registration_number matches
      - TB_Detected_Screening: copy 'TB_Detected' from Screening sheet
      - BC_Screening: set to '1' if 'Bact_confirmed_TB' == 1 in Screening sheet
    """

    if df_patient is None or df_patient.empty:
        return df_patient

    if df_screen is None or df_screen.empty:
        return df_patient

    # Normalize Registration_number column names
    def _find_col_by_names(df, names):
        norm_cols = {c: str(c).strip().lower().replace(" ", "").replace("_", "") for c in df.columns}
        for cand in names:
            cand_norm = str(cand).strip().lower().replace(" ", "").replace("_", "")
            for orig, norm in norm_cols.items():
                if norm == cand_norm:
                    return orig
        return None

    reg_patient = _find_col_by_names(df_patient, ["Registration_number", "Registration number"])
    reg_screen  = _find_col_by_names(df_screen, ["Registration_number", "Registration number"])
    tbdet_col   = _find_col_by_names(df_screen, ["TB_Detected", "TB Detected"])
    bc_col      = _find_col_by_names(df_screen, ["Bact_confirmed_TB", "Bact confirmed TB"])
    channel_col = _find_col_by_names(df_screen, ["Channel"])
    

    if not reg_patient or not reg_screen:
        return df_patient

    # Build lookup dicts from screening
    tbdet_map   = dict(zip(df_screen[reg_screen], df_screen[tbdet_col])) if tbdet_col else {}
    bc_map      = dict(zip(df_screen[reg_screen], df_screen[bc_col])) if bc_col else {}
    channel_map = dict(zip(df_screen[reg_screen], df_screen[channel_col])) if channel_col else {}
    

    # Add new columns in patient dataframe
    df_patient["TB_Detected_Screening"] = df_patient[reg_patient].map(tbdet_map).fillna(0)
    df_patient["BC_Screening"] = df_patient[reg_patient].map(bc_map).fillna(0)
    df_patient["Channel_Screening"] = df_patient[reg_patient].map(channel_map).fillna("")
    
    return df_patient


def add_TBDT_1(df):
    df["TBDT_1"] = 0
    type_col = _find_col_by_names(df, ["TB_Type_of_patient", "Type_of_patient", "TB Type of patient"])
    if type_col is None:
        return df
    for idx, row in df.iterrows():
        patient_type = str(row.get(type_col, "")).strip().lower()
        if patient_type in ["new", "relapse"]:
            df.at[idx, "TBDT_1"] = 1
    return df

def add_TBDT_3c(df):
    df["TBDT_3c"] = 0
    for idx, row in df.iterrows():
        tbdt_1 = row.get("TBDT_1", 0)
        channel = str(row.get("Channel_Screening", "")).strip().lower()
        if tbdt_1 == 1 and channel in ["volunteer", "ichv"]:
            df.at[idx, "TBDT_3c"] = 1
    return df

def add_TBP_1(df):
    """
    Set TBP-1 = 1 when:
      - TPT_Treatment_Regimen is one of {6H, 3HP, 3HR} (space/case tolerant)
      - AND TPT_Start_date is present (not blank/NaN/'nan'/ '-', etc.)
    Otherwise TBP-1 = 0.
    """


    df["TBP-1"] = 0
    def _find_col(df, names):
        norm_cols = {c: str(c).strip().lower().replace(" ", "").replace("_", "") for c in df.columns}
        for cand in names:
            cand_norm = str(cand).strip().lower().replace(" ", "").replace("_", "")
            for orig, norm in norm_cols.items():
                if norm == cand_norm:
                    return orig
        return None

    regimen_col = _find_col(df, ["TPT_Treatment_Regimen", "TPT Treatment Regimen", "TPTRegimen", "TPT_TreatmentRegimen"])
    start_col = _find_col(df, ["TPT_Start_date", "TPT Start Date", "TPT_Start_Date", "TPTStartDateate"])

    if regimen_col is None or start_col is None:
        return df

    valid_regimens = {"6H", "3HP", "3HR"}

    BLANK_TOKENS = {"", "nan", "na", "n/a", "none", "-", "--"}

    for idx, row in df.iterrows():
        reg_val = row.get(regimen_col)
        start_val = row.get(start_col)
        if pd.isna(reg_val):
            continue
        reg_s = str(reg_val).strip()
        if reg_s == "":
            continue
        reg_norm = reg_s.upper().replace(" ", "").replace("-", "")
        if reg_norm not in valid_regimens:
            continue
        if pd.isna(start_val):
            continue
        if isinstance(start_val, str):
            s = start_val.strip().replace("\u200b", "").replace("\u00a0", "")
            if s.lower() in BLANK_TOKENS or s == "":
                continue
        df.at[idx, "TBP-1"] = 1

    return df



def add_TBHIV_5(df):
    df["TBHIV_5"] = 0
    for idx, row in df.iterrows():
        tbdt_1 = row.get("TBDT_1", 0)
        hiv_status = str(row.get("HIV_status", "")).strip().lower()
        if tbdt_1 == 1 and hiv_status in ["positive", "negative"]:
            df.at[idx, "TBHIV_5"] = 1
    return df

def add_TBO2a_N(df):
    df["TBO2a_N"] = 0
    valid_outcomes = ["cure", "complete", "cured", "completed", "treatment completed"]
    for idx, row in df.iterrows():
        tbo2a_d = row.get("TBO2a_D", 0)
        outcome = str(row.get("TB_Treatment_Outcome", "")).strip().lower()
        if tbo2a_d == 1 and outcome in valid_outcomes:
            df.at[idx, "TBO2a_N"] = 1
    return df

def add_patient_checks(df_patient, df_screen):
    """
    Add validation/check columns to Patient Data sheet:
      - Regimen_check
      - Type_of_Disease_check
      - Outcome_check
      - Tin/Refer_from_check
    """

    if df_patient is None or df_patient.empty:
        return df_patient

    # ---- Regimen_check ----
    df_patient["Regimen_check"] = ""
    for idx, row in df_patient.iterrows():
        regimen = str(row.get("TB_Treatment_Regimen", "")).strip().upper()
        type_patient = str(row.get("TB_Type_of_patient", "")).strip().title()
        age_year = row.get("Age_Year", None)

        if regimen == "IR":
            # must be "New" or blank
            if not (type_patient in ["New", ""]):
                df_patient.at[idx, "Regimen_check"] = "Invalid"

        elif regimen == "CR":
            # must be < 15 years
            try:
                if age_year is not None and age_year != "" and float(age_year) >= 15:
                    df_patient.at[idx, "Regimen_check"] = "Invalid"
            except:
                df_patient.at[idx, "Regimen_check"] = "Invalid"

    # ---- Type_of_Disease_check (robust replacement) ----
    import numbers
    import pandas as pd

    def _is_one_flag(v):
        """
        Return True if v represents a '1' / present flag.
        Handles: int/float (1, 1.0), bool True, and strings: "1","1.0","yes","y","true","t"
        """
        if pd.isna(v):
            return False
        if isinstance(v, bool):
            return bool(v)
        if isinstance(v, numbers.Number):
            try:
                return int(v) == 1
            except Exception:
                return False
        s = str(v).strip().lower()
        return s in {"1", "1.0", "yes", "y", "true", "t"}

    # ensure column exists
    df_patient["Type_of_Disease_check"] = ""

    allowed_disease = {"p", "pulmonary tb", "pulmonary"}

    for idx, row in df_patient.iterrows():
        bc_screen_val = row.get("BC_Screening", None)
        bc_val = row.get("BC", None)
        disease = str(row.get("TB_Type_of_Disease", "") or "").strip().lower()

        if _is_one_flag(bc_screen_val) or _is_one_flag(bc_val):
            if disease not in allowed_disease:
                df_patient.at[idx, "Type_of_Disease_check"] = "Invalid"
            else:
                df_patient.at[idx, "Type_of_Disease_check"] = ""


    # ---- Outcome_check ----
    def _is_one_val(v):
        """Return True for values that mean '1' / present (handles numeric, bool, strings like '1','1.0','yes')."""
        import pandas as pd
        if pd.isna(v):
            return False
        if isinstance(v, (int, float, bool)):
            try:
                return int(v) == 1
            except Exception:
                return False
        s = str(v).strip().lower()
        return s in {"1", "1.0", "yes", "y", "true", "t"}

    df_patient["Outcome_check"] = ""
    for idx, row in df_patient.iterrows():
        outcome = str(row.get("TB_Treatment_Outcome", "")).strip().lower()
        bc_val = row.get("BC", None)
        bc_screen_val = row.get("BC_Screening", None)

        bc_flag = _is_one_val(bc_val) or _is_one_val(bc_screen_val)

        # If outcome is Cure/Cured but neither BC nor BC_Screening indicates confirmed TB -> Invalid
        if outcome in {"cure", "cured"} and not bc_flag:
            df_patient.at[idx, "Outcome_check"] = "Invalid"
        else:
            df_patient.at[idx, "Outcome_check"] = ""




    # ---- Tin/Refer_from_check ----
    df_patient["Tin/Refer_from_check"] = "No"
    if df_screen is not None and not df_screen.empty:
        reg_patient = "Registration_number" if "Registration_number" in df_patient.columns else None
        reg_screen = "Registration_number" if "Registration_number" in df_screen.columns else None
        if reg_patient and reg_screen:
            screening_set = set(df_screen[reg_screen].astype(str))
            df_patient["Tin/Refer_from_check"] = df_patient[reg_patient].astype(str).apply(
                lambda x: "Yes" if x not in screening_set else "No"
            )

    return df_patient


def create_vs_update(df_patient, df_visit):
    """
    Pivot Visit data (rows -> columns) and add patient columns:
      - Tx_started_date (from Patient TB_Treatment_Start_Date)
      - BC (from Patient BC)
      - Outcome_PD (from Patient TB_Treatment_Outcome)
     
    Includes error checking during the process and flags issues in new columns.
    """

    if df_visit is None or df_visit.empty:
        # still return patient info if available
        if df_patient is None or df_patient.empty:
            return pd.DataFrame()
        # try to return minimal patient base
        reg_p = _find_col_by_names(df_patient, ["Registration_number", "Registration number"])
        tx_col = _find_col_by_names(df_patient, ["TB_Treatment_Start_Date", "Tx_started_date", "TB Treatment Start Date"])
        bc_col = _find_col_by_names(df_patient, ["BC", "Bact_confirmed_TB", "Bact confirmed TB"])
        out_col = _find_col_by_names(df_patient, ["TB_Treatment_Outcome", "Outcome_PD", "TB Treatment Outcome"])
        base = pd.DataFrame()
        base["Registration_number"] = df_patient[reg_p] if reg_p else df_patient.index.astype(str)
        base["Tx_started_date"] = df_patient.get(tx_col, "")
        base["BC"] = df_patient.get(bc_col, "")
        base["Outcome_PD"] = df_patient.get(out_col, "")
        base["Outcome"] = ""

        base["Error"] = "No visit data available" # Add error flag

        return base


    # find registration column in visit sheet
    reg_col = _find_col_by_names(df_visit, ["Registration_number", "Registration number", "Reg No", "Reg"])
    if reg_col is None:
        # fallback to first column
        reg_col = df_visit.columns[0]
        df_visit["Reg_Col_Error"] = "Registration column not found, using first column"  # Add error flag

    # identify a visit-date column if present (used for sensible ordering)
    visit_date_col = _find_col_by_names(df_visit, ["Visit_date", "Visit date", "Date", "VisitDate"])
    
    # we will pivot all other visit columns (so we preserve everything)
    visit_value_cols = [c for c in df_visit.columns if c != reg_col]

    # sort by reg and date (if a date column exists) to give meaningful visit order
    df_v = df_visit.copy()
    if visit_date_col:
        try:
            df_v[visit_date_col] = pd.to_datetime(df_v[visit_date_col], errors="coerce", dayfirst=True)
            df_v = df_v.sort_values([reg_col, visit_date_col])
        except Exception:
            df_v["Date_Error"] = "Visit date column could not be parsed"  # Add error flag
            df_v = df_v.sort_values([reg_col])
    else:
        df_v["Date_Error"] = "Visit date column not found"  # Add error flag
        df_v = df_v.sort_values([reg_col])


    # create visit index per registration (1..n)
    df_v["visit_idx"] = df_v.groupby(reg_col).cumcount() + 1

    # pivot everything (except reg_col and visit_idx)
    pivot_value_cols = [c for c in df_v.columns if c not in [reg_col, "visit_idx"]]
    if not pivot_value_cols:
        df_wide = pd.DataFrame()  # Empty DataFrame
        df_wide["Pivot_Error"] = "No columns to pivot"  # Add error flag
    else:
        try:
            df_wide = df_v.set_index([reg_col, "visit_idx"])[pivot_value_cols].unstack(level="visit_idx")
            # flatten columns: e.g. ('Sputum_Result', 1) -> 'Sputum_Result_1'
            flat_cols = []
            for col, idx in df_wide.columns.to_flat_index():
                # normalize column name to safe string
                col_name = str(col).strip().replace(" ", "_")
                flat_cols.append(f"{col_name}_{idx}")
            df_wide.columns = flat_cols
            df_wide = df_wide.reset_index().rename(columns={reg_col: "Registration_number"} if reg_col != "Registration_number" else {})
        except Exception as e:
            df_wide = pd.DataFrame()
            df_wide["Pivot_Error"] = f"Error during pivot: {str(e)}"  # Add error flag

    # prepare patient subset (Registration_number + required patient columns)
    df_p = df_patient.copy() if (df_patient is not None) else pd.DataFrame()
    reg_p_col = _find_col_by_names(df_p, ["Registration_number", "Registration number"]) if not df_p.empty else None
    if reg_p_col and reg_p_col != "Registration_number":
        df_p = df_p.rename(columns={reg_p_col: "Registration_number"})
    elif not reg_p_col and not df_p.empty:
        # if patient sheet exists but has no recognizable reg column, create one from index as fallback
        df_p["Registration_number"] = df_p.index.astype(str)
        df_p["Reg_Num_Error"] = "Registration number column not found, using index"  # Add error flag

    tx_col = _find_col_by_names(df_p, ["TB_Treatment_Start_Date", "TB Treatment Start Date", "Tx_started_date"])
    bc_col = _find_col_by_names(df_p, ["BC", "Bact_confirmed_TB", "Bact confirmed TB"])
    outcome_pd_col = _find_col_by_names(df_p, ["TB_Treatment_Outcome", "TB Treatment Outcome", "Outcome_PD"])

    patient_select = ["Registration_number"]
    if tx_col: patient_select.append(tx_col)
    if bc_col: patient_select.append(bc_col)
    if outcome_pd_col: patient_select.append(outcome_pd_col)

    if not df_p.empty:
        patient_subset = df_p[patient_select].drop_duplicates(subset="Registration_number")
        # rename to standard names
        rename_map = {}
        if tx_col: rename_map[tx_col] = "Tx_started_date"
        if bc_col: rename_map[bc_col] = "BC"
        if outcome_pd_col: rename_map[outcome_pd_col] = "Outcome_PD"
        patient_subset = patient_subset.rename(columns=rename_map)
    else:
        patient_subset = pd.DataFrame(columns=["Registration_number", "Tx_started_date", "BC", "Outcome_PD"])
        patient_subset["Patient_Data_Error"] = "Patient data sheet is empty"  # Add error flag

    # merge pivoted visits with patient columns
    if df_wide.empty:
        df_vs = patient_subset  # Assign patient_subset to df_vs
        df_vs["Merge_Error"] = "Visit data could not be merged due to a pivoting error"
    else:
        df_vs = df_wide.merge(patient_subset, on="Registration_number", how="left")

    # ensure added columns exist
    for c in ["Tx_started_date", "BC", "Outcome_PD"]:
        if c not in df_vs.columns:
            df_vs[c] = ""

    # Build a mapping of visit_idx -> (sputum_col_name, date_col_name, any other columns)
    # We expect flattened columns like 'Sputum_Result_1', 'Visit_date_1', etc.
    visit_map = {}
    for col in df_vs.columns:
        if col in ["Registration_number", "Tx_started_date", "BC", "Outcome_PD", "Outcome"]:
            continue
        # try to split suffix index
        parts = col.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            base = parts[0].lower()
            idx = int(parts[1])
            visit_map.setdefault(idx, {})[base] = col

    # helper to test BC truthy (1/yes etc.)
    def _is_bc_flag(val):
        if pd.isna(val):
            return False
        s = str(val).strip().lower()
        return s in ["1", "yes", "y", "true"]



    # final: ensure columns order - SN, Registration_number, Tx_started_date, BC, Outcome_PD, <visit cols...>, Outcome
    drop_bases = {"sn", "tb_or_tpt", "lab_number"}  # columns to exclude (case-insensitive base name)
    visit_cols = [
        c for c in df_vs.columns
        if c not in ["Registration_number", "Tx_started_date", "BC", "Outcome_PD", "Outcome"]
        and all(not c.lower().startswith(base) for base in drop_bases)
    ]

    ordered = ["Registration_number", "Tx_started_date", "BC", "Outcome_PD"] + visit_cols + ["Outcome"]
    df_vs = df_vs.loc[:, [c for c in ordered if c in df_vs.columns]]

    # ---- Format all date columns as yyyy-mm-dd ----
    for col in df_vs.columns:
        if "date" in col.lower():  # crude check, works for Tx_started_date, Outcome_PD, visit date cols
            df_vs[col] = pd.to_datetime(df_vs[col], errors="coerce").dt.strftime("%Y-%m-%d")

    # ---- Add SN as first column, starting at 1 ----
    df_vs.insert(0, "SN", range(1, len(df_vs) + 1))

    return df_vs

def clean_dates_and_sn(df, add_sn=False):
    """Format all date-like columns as yyyy-mm-dd, optionally add SN column."""
    if df is None or df.empty:
        return df

    # Format all date-like columns
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
            df[col] = df[col].fillna("")  # keep blanks

    return df



# ==================== Main Rule Check Function ====================

def check_rules(excel_file, output_file="output.xlsx"):
    """
    Processes data from an Excel file, performs various checks, and writes the results to a new Excel file.

    Args:
        excel_file (str): Path to the input Excel file.
        output_file (str): Path to the output Excel file where results will be written.
    """

    xls = pd.ExcelFile(excel_file)
    sheets = xls.sheet_names
    get_sheet = lambda name: xls.parse(name) if name in sheets else pd.DataFrame()
    df_screen = get_sheet("Screening")
    df_patient = get_sheet("Patient data")
    df_service = get_sheet("Service Point")
    df_visit = get_sheet("Visit data")
    df_dropdown = get_sheet("Dropdown")

    state_regions, state_township_dict, var_values_dict = load_dropdowns(df_dropdown) if not df_dropdown.empty else (set(), {}, {})
    allowed_registration_numbers = set(df_patient["Registration_number"].apply(normalize)) if "Registration_number" in df_patient.columns else set()

    # Screening
    df_screen = check_state_region(df_screen, state_regions)
    df_screen = check_township(df_screen, state_township_dict)
    df_screen = check_service_delivery_point(df_screen, df_service)
    df_screen = check_reporting_month(df_screen, var_values_dict.get("Reporting_Month", []))
    df_screen = check_screening_date(df_screen)
    df_screen = check_age_year(df_screen)
    df_screen = check_sex(df_screen, var_values_dict)
    for col in [
        "Referred_from", "TB_contact", "Symptoms_Cough≥2wk", "Symptoms_Fever",
        "Symptoms_Weight_Loss", "Symptoms_Night_Sweat",
        "Examination_results_Sputum", "Examination_results_Gene_Xpert",
        "Examination_results_Truenet", "Result", "TPT _history", "Channel"
    ]:
        df_screen = check_value_dropdown(df_screen, col, var_values_dict)
    df_screen = check_registration_number(df_screen, allowed_registration_numbers)
    df_screen = add_presumptive_tb_referred(df_screen)
    df_screen = add_tb_detected(df_screen)
    df_screen = add_bact_confirmed_tb(df_screen)
    df_screen = add_result_check(df_screen)
    df_screen = add_duplicate_check(df_screen, ["Service_delivery_point", "Name", "Age_Year", "Sex", "Screening_Date"])
    df_screen = add_ongoing_tb_case(df_screen, df_patient)

    # Patient
    df_patient = check_state_region(df_patient, state_regions)
    df_patient = check_township(df_patient, state_township_dict)
    df_patient = check_service_delivery_point(df_patient, df_service)
    for col in [
        "Transfer_in", "HIV_status", "TB_Type_of_patient",
        "TB_Type_of_Disease", "TB_Treatment_Regimen", "TB_Treatment_Outcome",
        "TPT_Treatment_Regimen"
    ]:
        df_patient = check_value_dropdown(df_patient, col, var_values_dict)
    for col in [
        "TB_Treatment_Outcome_Date", "TPT_Start_date", "TPT_End_date", "HIV_testing_date"
    ]:
        df_patient = check_date_col(df_patient, col)
    df_patient = check_enrolled_date(df_patient)
    df_patient = check_tb_treatment_start_date(df_patient)
    df_patient = check_tpt_end_date_rule(df_patient)
    df_patient = check_registration_number_duplicate(df_patient)
    df_patient = check_age_year(df_patient)
    df_patient = check_sex(df_patient, var_values_dict)
    df_patient = check_transfer_in_date(df_patient)
    df_patient = add_TBDT_1(df_patient)
    df_patient = add_TBHIV_5(df_patient)
    df_patient = add_TBP_1(df_patient)
    df_patient = add_TBO2a_N(df_patient)
    df_patient = merge_screening_columns(df_patient, df_screen)
    df_patient = add_TBDT_3c(df_patient)
    df_patient = add_patient_checks(df_patient, df_screen)

    # Visit
    if "Visit_date" in df_visit.columns:
        df_visit = check_date_col(df_visit, "Visit_date")
    for col in [
        "Sputum_Result", "Gene_Xpert_Result", "Truenet_Result"
    ]:
        df_visit = check_value_dropdown(df_visit, col, var_values_dict)

    # Service Point
    df_service = check_state_region(df_service, state_regions)
    df_service = check_township(df_service, state_township_dict)


    # ---- VS_Update sheet ----
    vs_update_df = create_vs_update(df_patient, df_visit)
    vs_update_df = vs_update_df.loc[:, ~vs_update_df.columns.str.startswith("TB__or_TPT")]


    # Output final sheets
    results = {
        "Screening": df_screen,
        "Patient Data": df_patient,
        "Service Point": df_service,
        "Visit Data": df_visit,
        "VS_Update": vs_update_df
    }

    # ✅ Apply formatting before export
    df_patient = clean_dates_and_sn(df_patient)
    df_screen = clean_dates_and_sn(df_screen)
    df_visit = clean_dates_and_sn(df_visit)
    results["VS_Update"] = clean_dates_and_sn(results["VS_Update"])

    # ✅ Combine all *_Error etc. into one Comment column
    def combine_errors(df, sheet_name=None):
        # Find error-like columns
        error_cols = [c for c in df.columns if "Error" in c or "Duplicate" in c or "_check" in c]

        # --- Patient Data special handling ---
        if sheet_name == "Patient Data":
            if "Tin/Refer_from_check" in df.columns:
                df["Tin_Refer_from"] = df["Tin/Refer_from_check"]
                df = df.drop(columns=["Tin/Refer_from_check"])
                if "Tin/Refer_from_check" in error_cols:
                    error_cols.remove("Tin/Refer_from_check")

        if not error_cols:
            return df

        def row_errors(row):
            msgs = []
            for col in error_cols:
                val = row.get(col)
                if pd.isna(val) or str(val).strip() == "":
                    continue

                # --- Screening: skip Result_check = T ---
                if sheet_name == "Screening" and col == "Result_check" and str(val).strip().upper() == "T":
                    continue

                # Clean variable name (remove _Error/_check/_Check)
                clean_name = (
                    col.replace("_Error", "")
                    .replace("_error", "")
                    .replace("_Check", "")
                    .replace("_check", "")
                )
                msgs.append(f"{clean_name}: {val}")
            return "; ".join(msgs)

        df["Comment"] = df.apply(row_errors, axis=1)

        # Drop old error columns
        df = df.drop(columns=error_cols)
        return df


    results = {
        "Screening": combine_errors(df_screen, "Screening"),
        "Patient Data": combine_errors(df_patient, "Patient Data"),
        "Service Point": combine_errors(df_service, "Service Point"),
        "Visit Data": combine_errors(df_visit, "Visit Data"),
        "VS_Update": combine_errors(results["VS_Update"], "VS_Update")
    }


    # Write DataFrames to Excel file
    with pd.ExcelWriter(output_file) as writer:
        for sheet_name, df in results.items():

            if isinstance(df, pd.DataFrame):
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # Truncate sheet name to 31 characters
            else:
                print(f"Sheet '{sheet_name}' is not a DataFrame and will not be written.")

    
    print(f"Results written to {output_file}")
    return results

if __name__ == "__main__":
    results = check_rules("TB_TestA.xlsx", "output.xlsx")  # Specify both input and output files
    for key, df in results.items():
        print(f"\n--- {key} ---")
        print(df.head())


# ----------------- Streamlit UI -----------------
st.image("TB image2.jpg", width=200)  # TB logo/image
st.title("📊 IHRP: TB Data Verification App")

st.markdown("""
Upload your **Excel file** for TB data verification.  
The app will apply built-in rules and show validation results.  
You can download all results as a single Excel file with multiple sheets.
""")

# Upload Excel data file only
data_file = st.file_uploader("📂 Upload Excel file to verify", type=["xlsx", "csv"])

if data_file:
    try:
        results = check_rules(data_file)
        excel_output = io.BytesIO()
        sheet_count = 0

        with pd.ExcelWriter(excel_output, engine="xlsxwriter") as writer:
            if isinstance(results, dict):
                st.markdown("## 📑 Validation Results:")
                for k, v in results.items():
                    st.write(f"**{k}**")
                    if isinstance(v, pd.DataFrame):
                        if not v.empty:
                            st.dataframe(v, use_container_width=True)
                        else:
                            st.success(f"No issues found in {k}! ✅")
                        v.to_excel(writer, index=False, sheet_name=k[:31])
                        sheet_count += 1
                    else:
                        st.write(v)

        if sheet_count > 0:
            st.download_button(
                label="⬇️ Download ALL Results as Excel (multi-sheet)",
                data=excel_output.getvalue(),
                file_name="all_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"❌ Error running rules: {e}")

st.markdown("---")
st.markdown("🩺 Created with Streamlit")
