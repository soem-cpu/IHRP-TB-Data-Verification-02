import pandas as pd
import numpy as np
from datetime import datetime

# ==================== Utility Functions ====================

def normalize(x):
    if pd.isnull(x):
        return ""
    return str(x).strip().replace("\u200b", "").replace("\t", "").replace("\n", "").replace("\u00a0", " ")

def _norm_text(x):
    """
    Normalize text for robust comparisons:
    - lower
    - remove common punctuation
    - collapse spaces
    """
    s = normalize(x).lower()
    for ch in [":", "-", "_", ";", ","]:
        s = s.replace(ch, " ")
    return " ".join(s.split())

def is_date(val, min_date="2024-01-01", max_date=None):
    if max_date is None:
        max_date = datetime.today().strftime("%Y-%m-%d")
    try:
        date = pd.to_datetime(val, errors="coerce", dayfirst=True)
        if pd.isnull(date):
            return False
        return pd.Timestamp(min_date) <= date <= pd.Timestamp(max_date)
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
    if df is None or df.empty:
        return None
    norm_cols = {c: str(c).strip().lower().replace(" ", "").replace("_", "") for c in df.columns}
    for cand in names:
        cand_norm = str(cand).strip().lower().replace(" ", "").replace("_", "")
        for orig, norm in norm_cols.items():
            if norm == cand_norm:
                return orig
    return None

def _is_one_flag(v):
    """True if v represents 1/yes/true."""
    if pd.isna(v):
        return False
    s = str(v).strip().lower()
    return s in {"1", "1.0", "yes", "y", "true", "t"}

def _is_negative_sputum(v):
    if pd.isna(v):
        return False
    s = str(v).strip().lower()
    return s in {"neg", "negative"}

def _is_positive_sputum(v):
    if pd.isna(v):
        return False
    s = str(v).strip().lower()
    return s in {"pos", "positive"}

def check_symptom_yes_no(df):
    """
    Add 'to recheck symptom' ONLY IF
    all symptom-related columns are blank or NO/N.
    If ANY one is YES/Y → no comment.
    """

    cols = [
        "TB_contact",
        "Symptoms_Cough≥2wk",
        "Symptoms_Fever",
        "Symptoms_Weight_Loss",
        "Symptoms_Night_Sweat",
    ]

    YES_SET = {"y", "yes"}
    NO_SET = {"n", "no", ""}

    df["Symptom_YN_check"] = ""

    for idx, row in df.iterrows():
        values = []

        for c in cols:
            if c not in df.columns:
                continue
            v = str(row.get(c, "")).strip().lower()
            values.append(v)

        # if no symptom columns exist → skip
        if not values:
            continue

        # ANY YES → OK (no comment)
        if any(v in YES_SET for v in values):
            continue

        # ALL blank or NO → recheck
        if all(v in NO_SET for v in values):
            df.at[idx, "Symptom_YN_check"] = "to recheck symptom"

    return df


def month_diff_from_tx(tx_date, visit_date, clamp_to_zero=True):
    """
    Month difference between tx_date and visit_date using calendar month difference.
    """
    tx = pd.to_datetime(tx_date, errors="coerce", dayfirst=True)
    vd = pd.to_datetime(visit_date, errors="coerce", dayfirst=True)
    if pd.isna(tx) or pd.isna(vd):
        return pd.NA

    diff = (vd.year - tx.year) * 12 + (vd.month - tx.month)
    if clamp_to_zero and diff < 0:
        diff = 0
    return int(diff)

# ==================== Dropdowns ====================

def load_dropdowns(dropdown_df):
    state_regions = set()
    state_township_dict = {}
    var_values_dict = {}
    for _, row in dropdown_df.iterrows():
        variable = normalize(row.get("Variable", ""))
        value = normalize(row.get("Value", ""))
        if variable == "State_Region":
            state_regions.add(value)
            var_values_dict.setdefault("State_Region", set()).add(value)
        elif variable in state_regions:
            state_township_dict.setdefault(variable, set()).add(value)
        else:
            var_values_dict.setdefault(variable, set()).add(value)
    var_values_dict = {k: sorted(list(v)) for k, v in var_values_dict.items()}
    return state_regions, state_township_dict, var_values_dict

# ==================== General Checks ====================

def check_list(df, column, allowed):
    allowed = set(normalize(i) for i in allowed)
    error_col = f"{column}_Error"

    if column in df.columns:
        df[error_col] = np.where(
            ~df[column].apply(normalize).isin(allowed),
            "not in allowed list",
            ""
        )
    else:
        df[error_col] = "Column not found"
    return df

def check_value_dropdown(df, column, var_values_dict):
    allowed = var_values_dict.get(column, None)
    if not allowed:
        err_col = f"{column}_Error"
        if err_col in df.columns:
            df[err_col] = ""
        return df
    return check_list(df, column, allowed)

def check_state_region(df, allowed):
    error_col = "State_Region_Error"
    if "State_Region" in df.columns:
        df[error_col] = np.where(~df["State_Region"].apply(normalize).isin(set(allowed)), "Invalid State Region", "")
    else:
        df[error_col] = "Column 'State_Region' not found"
    return df

def check_township(df, state_township_dict):
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

    df[error_col] = np.where(~df.apply(is_valid_township, axis=1), "Invalid State_Region/Township combination", "")
    df.drop(columns=["State_Region_Norm", "Township_Norm"], inplace=True, errors="ignore")
    return df

def check_service_delivery_point(df, df_service_point):
    error_col = "Service_delivery_point_Error"

    svc_col = _find_col_by_names(df, ["Service_delivery_point", "Service Delivery Point", "Service_point"])
    township_col = _find_col_by_names(df, ["Township"])

    if svc_col is None or township_col is None:
        df[error_col] = "Missing Service_delivery_point or Township column"
        return df

    sp_code_col = _find_col_by_names(df_service_point, ["Service_delivery_point_code", "Service delivery point code", "Service Point Code"])
    sp_township_col = _find_col_by_names(df_service_point, ["Township"])

    if sp_code_col is None or sp_township_col is None:
        df[error_col] = "Missing Service Point code or Township column in Service Point dataframe"
        return df

    allowed_pairs = set(
        (normalize(r[sp_township_col]), normalize(r[sp_code_col]))
        for _, r in df_service_point.iterrows()
        if pd.notnull(r.get(sp_township_col)) and pd.notnull(r.get(sp_code_col))
    )

    def is_valid_pair(row):
        township = normalize(row.get(township_col, ""))
        svc_point = normalize(row.get(svc_col, ""))
        return (township, svc_point) in allowed_pairs

    df[error_col] = np.where(~df.apply(is_valid_pair, axis=1), "Invalid Service_delivery_point/Township combination", "")
    return df

def check_screening_date(df):
    error_col = "Screening_Date_Error"
    if "Screening_Date" not in df.columns:
        df[error_col] = "Column 'Screening_Date' not found"
        return df
    df[error_col] = df["Screening_Date"].apply(lambda v: "Invalid date" if not is_date(v, min_date="2024-01-01") else "")
    return df

def check_age_year(df):
    error_col = "Age_Year_Error"
    if "Age_Year" not in df.columns:
        df[error_col] = "Column 'Age_Year' not found"
        return df
    df[error_col] = df["Age_Year"].apply(lambda v: "Invalid age" if not is_numeric(v, 0, 100) else "")
    return df

def check_sex(df, var_values_dict):
    error_col = "Sex_Error"
    if "Sex" not in df.columns:
        df[error_col] = "Column 'Sex' not found"
        return df

    allowed = set(x.lower() for x in var_values_dict.get("Sex", []))

    def check_sex_validity(row):
        sex = str(row.get("Sex", "")).strip().lower()
        name = str(row.get("Name", "")).strip() if "Name" in df.columns else ""
        prefix = extract_prefix(name) if name else None

        if allowed and sex not in allowed:
            return "Invalid sex value"

        if prefix in ["Ma", "Daw"] and sex not in ["female", "f"]:
            return "Sex mismatch with name prefix (Female)"
        elif prefix in ["Mg", "U", "Ko"] and sex not in ["male", "m"]:
            return "Sex mismatch with name prefix (Male)"

        return ""

    df[error_col] = df.apply(check_sex_validity, axis=1)
    return df

def check_registration_number(df, allowed):
    """
    FIX: skip validation if allowed set is empty
    (prevents all rows becoming invalid when Patient sheet is missing/empty)
    """
    if not allowed:
        err_col = "Registration_number_Error"
        if err_col in df.columns:
            df[err_col] = ""
        return df
    return check_list(df, "Registration_number", allowed)

def check_registration_number_duplicate(df):
    if "Registration_number" not in df.columns:
        df["Duplicate_RegNum"] = "Column 'Registration_number' not found"
        return df
    dupes = df.duplicated(subset=["Registration_number"], keep=False)
    df["Duplicate_RegNum"] = np.where(dupes, "Duplicate Registration Number", "")
    return df

def check_date_col(df, column):
    error_col = f"{column}_Error"
    if column not in df.columns:
        df[error_col] = f"Column '{column}' not found"
        return df
    df[error_col] = df[column].apply(lambda v: "Invalid date" if not is_date(v, min_date="2024-01-01") else "")
    return df

# ==================== Add/Map Columns (Service Point Level) ====================

def add_level_from_service_point(df, df_service_point):
    """
    Add 'Level' column into df (Screening / Patient data) by mapping from Service Point sheet:
      df[Service_delivery_point] == df_service_point[Service_delivery_point_code]
    """
    if df is None or df.empty:
        return df
    if df_service_point is None or df_service_point.empty:
        if "Level" not in df.columns:
            df["Level"] = ""
        return df

    svc_col = _find_col_by_names(df, ["Service_delivery_point", "Service Delivery Point", "Service_point"])
    sp_code_col = _find_col_by_names(df_service_point, ["Service_delivery_point_code", "Service delivery point code", "Service Point Code"])
    level_col = _find_col_by_names(df_service_point, ["Level", "LEVEL", "Service_Level", "Service level"])

    if svc_col is None or sp_code_col is None or level_col is None:
        if "Level" not in df.columns:
            df["Level"] = ""
        return df

    sp_map = {
        normalize(k): normalize(v)
        for k, v in zip(df_service_point[sp_code_col], df_service_point[level_col])
        if normalize(k) != ""
    }

    df["Level"] = df[svc_col].apply(lambda x: sp_map.get(normalize(x), ""))
    return df

# ==================== Screening: Added Columns / Rules ====================

def add_presumptive_tb_referred(df):
    """
    Presumptive_TB_referred = 1 if:
      - any Examination_results_* column has data, OR
      - Result is present and NOT unknown (unk/unknown/unkown)
    """
    df["Presumptive_TB_referred"] = 0

    exam_cols = [
        "Examination_results_Sputum",
        "Examination_results_Gene_Xpert",
        "Examination_results_Truenet",
        "Examination_results_CXR",
    ]
    UNKNOWN = {"unk", "unknown", "unkown"}

    for idx, row in df.iterrows():
        exam_present = any(c in df.columns and normalize(row.get(c, "")) != "" for c in exam_cols)
        res = _norm_text(row.get("Result", ""))
        result_present = (res != "" and res not in UNKNOWN)

        if exam_present or result_present:
            df.at[idx, "Presumptive_TB_referred"] = 1

    return df

def add_tb_detected(df):
    """
    TB_Detected = 1 if:
      - Presumptive_TB_referred == 1
      - AND Result indicates TB detected (bact-confirmed or clinical or TB/positive)
    """
    df["TB_Detected"] = 0

    TB_DETECTED = {
        # bacteriological confirmed
        "bact confirmed tb",
        "bact confirmed",
        "bacteriological confirmed",
        "bc",
        "2",
        # clinical diagnosed
        "clinically diagnosed tb",
        "clinically diagnosed",
        "clinically dx tb",
        "1",
        # generic
        "tb",
        "positive",
    }

    for idx, row in df.iterrows():
        presumptive_flag = str(row.get("Presumptive_TB_referred", "")).strip().lower() in {"1", "yes", "y", "true", "t"}
        if not presumptive_flag:
            continue

        res = _norm_text(row.get("Result", ""))
        if res in TB_DETECTED:
            df.at[idx, "TB_Detected"] = 1

    return df

def add_bact_confirmed_tb(df):
    """
    Bact_confirmed_TB = 1 if:
      - Presumptive_TB_referred == 1
      - AND Result indicates bacteriological confirmation
    """
    df["Bact_confirmed_TB"] = 0

    BACT_CONFIRMED = {
        "bact confirmed tb",
        "bact confirmed",
        "bacteriological confirmed",
        "bc",
        "2",
    }

    for idx, row in df.iterrows():
        presumptive_flag = str(row.get("Presumptive_TB_referred", "")).strip().lower() in {"1", "yes", "y", "true", "t"}
        if not presumptive_flag:
            continue

        res = _norm_text(row.get("Result", ""))
        if res in BACT_CONFIRMED:
            df.at[idx, "Bact_confirmed_TB"] = 1

    return df

def add_result_check(df):
    """
    Result_check = 'To recheck' if any exam indicates TB-positive
    BUT Result is NOT in allowed positive result set.
    Otherwise 'T'
    """
    df["Result_check"] = "T"

    GENE_POS = {
        "t", "tt", "ti", "rr",
        "mtb detected rr not detected",
        "mtb detected trace",
        "mtb detected rr detected",
        "mtb detected rr indeterminate",
    }

    TRUENET_POS = {
        "vt", "rr", "ti",
        "valid mtb detected",
        "mtb detected rr detected",
        "mtb detected rr indeterminate",
    }

    ALLOWED_POS_RESULTS = {
        "bact confirmed tb",
        "bact confirmed",
        "bacteriological confirmed",
        "bc",
        "2",
        "tb",
        "positive",
    }

    for idx, row in df.iterrows():
        sputum = _norm_text(row.get("Examination_results_Sputum", ""))
        gene = _norm_text(row.get("Examination_results_Gene_Xpert", ""))
        truen = _norm_text(row.get("Examination_results_Truenet", ""))
        result = _norm_text(row.get("Result", ""))

        sputum_pos = ("positive" in sputum) or ("+" in sputum)

        gene_pos = (gene in GENE_POS) or any(tok in gene for tok in GENE_POS if len(tok) > 2)
        truen_pos = (truen in TRUENET_POS) or any(tok in truen for tok in TRUENET_POS if len(tok) > 2)

        if sputum_pos or gene_pos or truen_pos:
            if result not in ALLOWED_POS_RESULTS:
                df.at[idx, "Result_check"] = "To recheck"

    return df

def add_duplicate_check(df, cols):
    for col in cols:
        if col not in df.columns:
            df["Duplicate_check"] = f"Column '{col}' not found"
            return df

    df["Duplicate_check"] = ""
    dupes = df.duplicated(subset=cols, keep=False)
    df.loc[dupes, "Duplicate_check"] = "To recheck for duplication"
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
            if pd.to_datetime(screen_date, dayfirst=True) > pd.to_datetime(enrolled_date, dayfirst=True):
                df_screen.at[idx, "Ongoing_TB_case_check"] = "Ongoing TB case?"

    return df_screen

# ==================== Patient Sheet Specific Checks / Adds ====================

def check_transfer_in_date(df):
    df["Transfer_in_Error"] = ""
    if "Transfer_in" not in df.columns or "Transfer_in_date" not in df.columns:
        df["Transfer_in_Error"] = "Missing Transfer_in or Transfer_in_date column"
        return df

    min_date = datetime(2024, 1, 1)
    max_date = datetime.today()

    for idx, row in df.iterrows():
        transfer_in = str(row.get("Transfer_in", "")).strip()
        date_val = row.get("Transfer_in_date", "")

        if transfer_in in ["Yes", "Y", "yes", "y", "1", 1]:
            try:
                parsed_date = pd.to_datetime(date_val, errors="raise", dayfirst=True)
                if not (min_date <= parsed_date <= max_date):
                    df.at[idx, "Transfer_in_Error"] = "Date out of range"
            except Exception:
                df.at[idx, "Transfer_in_Error"] = "Missing or invalid date"
        else:
            if pd.notna(date_val) and str(date_val).strip() != "":
                df.at[idx, "Transfer_in_Error"] = "Date should be blank"

    return df

def check_enrolled_date(df):
    df["Enrolled_Date_Error"] = ""
    if "Enrolled_Date" not in df.columns:
        df["Enrolled_Date_Error"] = "Column 'Enrolled_Date' not found"
        return df

    today = pd.Timestamp.today()

    for idx, row in df.iterrows():
        enrolled = row.get("Enrolled_Date", "")
        treatment_start = row.get("TB_Treatment_Start_Date", "")

        if pd.notnull(treatment_start):
            ts = pd.to_datetime(treatment_start, errors="coerce", dayfirst=True)
            if pd.notnull(ts) and ts.year == 2024:
                continue

        enrolled_parsed = pd.to_datetime(enrolled, errors="coerce", dayfirst=True)
        if pd.isnull(enrolled_parsed) or not (pd.Timestamp("2024-01-01") <= enrolled_parsed <= today):
            df.at[idx, "Enrolled_Date_Error"] = "Invalid enrolled date"

    return df

def check_tb_treatment_start_date(df):
    df["TB_Treatment_Start_Date_Error"] = ""

    regimen_col = _find_col_by_names(df, ["TB_Treatment_Regimen", "TB Treatment Regimen", "Regimen"])
    start_col = _find_col_by_names(df, ["TB_Treatment_Start_Date", "TB Treatment Start Date", "TB_Treatment_Start", "Start Date"])

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

def check_tb_treatment_outcome_date(df):
    """
    Outcome date check (if TB_Treatment_Outcome exists and is not blank-like, then outcome date must exist & valid).
    """
    df["TB_Treatment_Outcome_Date_Error"] = ""

    outcome_col = _find_col_by_names(df, ["TB_Treatment_Outcome", "TB Treatment Outcome", "Outcome"])
    outcomedate_col = _find_col_by_names(df, ["TB_Treatment_Outcome_Date", "TB Treatment Outcome Date", "Outcome Date"])

    if outcome_col is None or outcomedate_col is None:
        df["TB_Treatment_Outcome_Date_Error"] = "Missing TB_Treatment_Outcome or TB_Treatment_Outcome_Date column"
        return df

    min_date = pd.to_datetime("2024-01-01")
    max_date = pd.to_datetime(datetime.today().date())
    BLANK_LIKE = {"", "na", "n/a", "none", "null", "nan", "missing", "unknown"}

    for idx, row in df.iterrows():
        outcome_val = row.get(outcome_col)
        val = normalize(outcome_val).strip().lower()
        if val in BLANK_LIKE:
            continue

        outcomedate_val = row.get(outcomedate_col)
        if pd.isna(outcomedate_val) or normalize(outcomedate_val) == "":
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
    df["TPT_End_date_Error"] = ""

    start_col = _find_col_by_names(df, ["TPT_Start_date", "TPT Start Date", "TPT_Start_Date"])
    end_col = _find_col_by_names(df, ["TPT_End_date", "TPT End Date", "TPT_End_Date"])

    if start_col is None or end_col is None:
        df["TPT_End_date_Error"] = "Missing TPT_Start_date or TPT_End_date column"
        return df

    min_date = pd.to_datetime("2024-01-01")
    max_date = pd.to_datetime(datetime.today().date())
    six_months_ago = pd.Timestamp.today() - pd.DateOffset(months=6)

    for idx, row in df.iterrows():
        start_val = row.get(start_col)
        if pd.isna(start_val) or normalize(start_val) == "":
            continue

        start_date = pd.to_datetime(start_val, errors="coerce", dayfirst=True)
        if pd.isna(start_date):
            continue

        if start_date <= six_months_ago:
            end_val = row.get(end_col)
            end_date = pd.to_datetime(end_val, errors="coerce", dayfirst=True)

            if pd.isna(end_date) or normalize(end_val) == "":
                df.at[idx, "TPT_End_date_Error"] = "TPT_Start_date ≥ 6 months ago but TPT_End_date missing/invalid"
            elif end_date < min_date or end_date > max_date:
                df.at[idx, "TPT_End_date_Error"] = f"TPT_End_date out of allowed range (01-Jan-2024 to {max_date.strftime('%d-%b-%Y')})"

    return df

def merge_screening_columns(df_patient, df_screen):
    """
    Add calculated fields into Patient Data sheet:
      - Channel_Screening
      - TB_Detected_Screening
      - BC_Screening
    """
    if df_patient is None or df_patient.empty or df_screen is None or df_screen.empty:
        return df_patient

    reg_patient = _find_col_by_names(df_patient, ["Registration_number", "Registration number"])
    reg_screen = _find_col_by_names(df_screen, ["Registration_number", "Registration number"])
    tbdet_col = _find_col_by_names(df_screen, ["TB_Detected", "TB Detected"])
    bc_col = _find_col_by_names(df_screen, ["Bact_confirmed_TB", "Bact confirmed TB"])
    channel_col = _find_col_by_names(df_screen, ["Channel"])

    if not reg_patient or not reg_screen:
        return df_patient

    tbdet_map = dict(zip(df_screen[reg_screen], df_screen[tbdet_col])) if tbdet_col else {}
    bc_map = dict(zip(df_screen[reg_screen], df_screen[bc_col])) if bc_col else {}
    channel_map = dict(zip(df_screen[reg_screen], df_screen[channel_col])) if channel_col else {}

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

def add_TBP_1(df):
    """
    TBP-1 = 1 ONLY when:
      - TPT_Treatment_Regimen in {6H, 3HP, 3HR}
      - AND TPT_Start_date present
      - AND Age_Year <= 5
    """
    df["TBP-1"] = 0

    regimen_col = _find_col_by_names(df, ["TPT_Treatment_Regimen", "TPT Treatment Regimen", "TPTRegimen", "TPT_TreatmentRegimen"])
    start_col = _find_col_by_names(df, ["TPT_Start_date", "TPT Start Date", "TPT_Start_Date", "TPTStartDate"])
    age_col = _find_col_by_names(df, ["Age_Year", "Age Year", "Age"])

    if regimen_col is None or start_col is None or age_col is None:
        return df

    valid_regimens = {"6H", "3HP", "3HR"}
    BLANK_TOKENS = {"", "nan", "na", "n/a", "none", "-", "--"}

    for idx, row in df.iterrows():
        # age <= 5
        age_ok = False
        age_val = row.get(age_col)
        try:
            if pd.notna(age_val) and str(age_val).strip() != "":
                age_ok = float(age_val) <= 5
        except Exception:
            age_ok = False
        if not age_ok:
            continue

        # regimen valid
        reg_s = normalize(row.get(regimen_col, ""))
        if reg_s == "":
            continue
        reg_norm = reg_s.upper().replace(" ", "").replace("-", "")
        if reg_norm not in valid_regimens:
            continue

        # start date present
        start_val = row.get(start_col)
        if pd.isna(start_val):
            continue
        s = normalize(start_val).strip()
        if s.lower() in BLANK_TOKENS or s == "":
            continue

        df.at[idx, "TBP-1"] = 1

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
            if not (type_patient in ["New", ""]):
                df_patient.at[idx, "Regimen_check"] = "Invalid"
        elif regimen == "CR":
            try:
                if age_year is not None and age_year != "" and float(age_year) >= 15:
                    df_patient.at[idx, "Regimen_check"] = "Invalid"
            except Exception:
                df_patient.at[idx, "Regimen_check"] = "Invalid"

    # ---- Type_of_Disease_check ----
    def _is_one_flag_local(v):
        if pd.isna(v):
            return False
        if isinstance(v, bool):
            return bool(v)
        try:
            if isinstance(v, (int, float)):
                return int(v) == 1
        except Exception:
            pass
        s = str(v).strip().lower()
        return s in {"1", "1.0", "yes", "y", "true", "t"}

    df_patient["Type_of_Disease_check"] = ""
    allowed_disease = {"p", "ptb", "pulmonary tb", "pulmonary"}

    for idx, row in df_patient.iterrows():
        bc_screen_val = row.get("BC_Screening", None)
        bc_val = row.get("BC", None)
        disease = str(row.get("TB_Type_of_Disease", "") or "").strip().lower()

        if _is_one_flag_local(bc_screen_val) or _is_one_flag_local(bc_val):
            if disease not in allowed_disease:
                df_patient.at[idx, "Type_of_Disease_check"] = "Invalid"

    # ---- Outcome_check ----
    def _is_one_val(v):
        if pd.isna(v):
            return False
        try:
            if isinstance(v, (int, float, bool)):
                return int(v) == 1
        except Exception:
            pass
        s = str(v).strip().lower()
        return s in {"1", "1.0", "yes", "y", "true", "t"}

    df_patient["Outcome_check"] = ""
    for idx, row in df_patient.iterrows():
        outcome = str(row.get("TB_Treatment_Outcome", "")).strip().lower()
        bc_val = row.get("BC", None)
        bc_screen_val = row.get("BC_Screening", None)
        bc_flag = _is_one_val(bc_val) or _is_one_val(bc_screen_val)

        if outcome in {"cure", "cured"} and not bc_flag:
            df_patient.at[idx, "Outcome_check"] = "Invalid"

    # ---- Tin/Refer_from_check ----
    df_patient["Tin/Refer_from_check"] = "No"
    if df_screen is not None and not df_screen.empty:
        if "Registration_number" in df_patient.columns and "Registration_number" in df_screen.columns:
            screening_set = set(df_screen["Registration_number"].astype(str))
            df_patient["Tin/Refer_from_check"] = df_patient["Registration_number"].astype(str).apply(
                lambda x: "Yes" if x not in screening_set else "No"
            )

    return df_patient

# ==================== VS Update ====================

def create_vs_update(df_patient, df_visit):
    """
    Pivot Visit data (rows -> columns) and add patient columns:
      - Tx_started_date
      - BC
      - Outcome_PD

    Expansion key:
      - If Visit data contains Patient_name (or Name), expand by (Registration_number + Patient_name)
      - Else expand by Registration_number only

    Adds:
      - Visit_Month_1, Visit_Month_2, ... based on Visit_date_1, Visit_date_2, ...
      - Outcome (Treatment failed / Cure / LTFU / Complete)
    """
    if df_visit is None or df_visit.empty:
        if df_patient is None or df_patient.empty:
            return pd.DataFrame()

        reg_p = _find_col_by_names(df_patient, ["Registration_number", "Registration number"])
        tx_col = _find_col_by_names(df_patient, ["TB_Treatment_Start_Date", "Tx_started_date", "TB Treatment Start Date"])
        bc_col = _find_col_by_names(df_patient, ["BC", "Bact_confirmed_TB", "Bact confirmed TB"])
        out_col = _find_col_by_names(df_patient, ["TB_Treatment_Outcome", "Outcome_PD", "TB Treatment Outcome"])
        name_col_p = _find_col_by_names(df_patient, ["Patient_name", "Patient name", "Name", "Patient_Name"])
        township_col_p = _find_col_by_names(df_patient, ["Township"])

        base = pd.DataFrame()
        base["Registration_number"] = df_patient[reg_p] if reg_p else df_patient.index.astype(str)

        if name_col_p:
            base["Patient_name"] = df_patient.get(name_col_p, "")
        if township_col_p:
            base["Township"] = df_patient.get(township_col_p, "")

        base["Tx_started_date"] = df_patient.get(tx_col, "")
        base["BC"] = df_patient.get(bc_col, "")
        base["Outcome_PD"] = df_patient.get(out_col, "")
        base["Outcome"] = ""
        base["Error"] = "No visit data available"
        base.insert(0, "SN", range(1, len(base) + 1))
        return base

    # --- Find reg column in visit ---
    reg_col = _find_col_by_names(df_visit, ["Registration_number", "Registration number", "Reg No", "Reg"])
    if reg_col is None:
        reg_col = df_visit.columns[0]  # fallback
        df_visit["Reg_Col_Error"] = "Registration column not found, using first column"

    visit_date_col = _find_col_by_names(df_visit, ["Visit_date", "Visit date", "Date", "VisitDate"])

    # Optional: patient name in visit
    visit_name_col = _find_col_by_names(df_visit, ["Patient_name", "Patient name", "Name", "Patient_Name"])

    df_v = df_visit.copy()

    # Normalize / standardize visit name column to "Patient_name"
    if visit_name_col and visit_name_col != "Patient_name":
        df_v = df_v.rename(columns={visit_name_col: "Patient_name"})
        visit_name_col = "Patient_name"

    # Sort by key(s) and date if possible
    if visit_date_col:
        try:
            df_v[visit_date_col] = pd.to_datetime(df_v[visit_date_col], errors="coerce", dayfirst=True)
            sort_keys = [reg_col]
            if visit_name_col:
                sort_keys.append(visit_name_col)
            sort_keys.append(visit_date_col)
            df_v = df_v.sort_values(sort_keys)
        except Exception:
            df_v = df_v.sort_values([reg_col] + ([visit_name_col] if visit_name_col else []))
    else:
        df_v = df_v.sort_values([reg_col] + ([visit_name_col] if visit_name_col else []))

    # --- visit index per expansion group ---
    group_keys = [reg_col]
    if visit_name_col and visit_name_col in df_v.columns:
        group_keys.append(visit_name_col)

    df_v["visit_idx"] = df_v.groupby(group_keys).cumcount() + 1

    # --- Exclude meta columns from pivot (prevents Visit_date_Error_1 etc.) ---
    def _is_meta_col(c: str) -> bool:
        cl = str(c).strip().lower()
        return (
            cl.endswith("_error")
            or cl.endswith("_check")
            or "duplicate" in cl
            or cl == "comment"
            or cl.endswith("_warning")
        )

    pivot_value_cols = [
        c for c in df_v.columns
        if c not in [reg_col, "visit_idx"]
        and not _is_meta_col(c)
    ]

    # --- Pivot ---
    if not pivot_value_cols:
        df_wide = pd.DataFrame()
        df_wide["Pivot_Error"] = "No columns to pivot"
    else:
        pivot_index = [reg_col]
        if visit_name_col and visit_name_col in df_v.columns:
            pivot_index.append(visit_name_col)
        pivot_index.append("visit_idx")

        try:
            df_wide = df_v.set_index(pivot_index)[pivot_value_cols].unstack(level="visit_idx")

            flat_cols = []
            for col, idx in df_wide.columns.to_flat_index():
                col_name = str(col).strip().replace(" ", "_")
                flat_cols.append(f"{col_name}_{idx}")

            df_wide.columns = flat_cols
            df_wide = df_wide.reset_index()

            if reg_col != "Registration_number":
                df_wide = df_wide.rename(columns={reg_col: "Registration_number"})

        except Exception as e:
            df_wide = pd.DataFrame()
            df_wide["Pivot_Error"] = f"Error during pivot: {str(e)}"

    # --- Prepare patient subset for merge ---
    df_p = df_patient.copy() if (df_patient is not None) else pd.DataFrame()

    reg_p_col = _find_col_by_names(df_p, ["Registration_number", "Registration number"]) if not df_p.empty else None
    if reg_p_col and reg_p_col != "Registration_number":
        df_p = df_p.rename(columns={reg_p_col: "Registration_number"})
    elif not reg_p_col and not df_p.empty:
        df_p["Registration_number"] = df_p.index.astype(str)

    # Standardize patient name and township in patient sheet
    name_col_p = _find_col_by_names(df_p, ["Patient_name", "Patient name", "Name", "Patient_Name"])
    if name_col_p and name_col_p != "Patient_name":
        df_p = df_p.rename(columns={name_col_p: "Patient_name"})

    township_col_p = _find_col_by_names(df_p, ["Township"])
    if township_col_p and township_col_p != "Township":
        df_p = df_p.rename(columns={township_col_p: "Township"})

    tx_col = _find_col_by_names(df_p, ["TB_Treatment_Start_Date", "TB Treatment Start Date", "Tx_started_date"])
    bc_col = _find_col_by_names(df_p, ["BC", "Bact_confirmed_TB", "Bact confirmed TB"])
    outcome_pd_col = _find_col_by_names(df_p, ["TB_Treatment_Outcome", "TB Treatment Outcome", "Outcome_PD"])

    patient_select = ["Registration_number"]

    if "Patient_name" in df_p.columns:
        patient_select.append("Patient_name")
    if "Township" in df_p.columns:
        patient_select.append("Township")

    if tx_col:
        patient_select.append(tx_col)
    if bc_col:
        patient_select.append(bc_col)
    if outcome_pd_col:
        patient_select.append(outcome_pd_col)

    rename_map = {}
    if tx_col: rename_map[tx_col] = "Tx_started_date"
    if bc_col: rename_map[bc_col] = "BC"
    if outcome_pd_col: rename_map[outcome_pd_col] = "Outcome_PD"

    if not df_p.empty:
        # drop duplicates based on reg + optional patient_name (if present)
        dedup_keys = ["Registration_number"]
        if "Patient_name" in df_p.columns:
            dedup_keys.append("Patient_name")
        patient_subset = df_p[patient_select].drop_duplicates(subset=dedup_keys).rename(columns=rename_map)
    else:
        patient_subset = pd.DataFrame(columns=["Registration_number", "Patient_name", "Township", "Tx_started_date", "BC", "Outcome_PD"])

    # --- Merge wide + patient subset ---
    if df_wide.empty:
        df_vs = patient_subset.copy()
        df_vs["Merge_Error"] = "Visit data could not be merged due to a pivoting error"
    else:
        merge_keys = ["Registration_number"]
        if ("Patient_name" in df_wide.columns) and ("Patient_name" in patient_subset.columns):
            merge_keys.append("Patient_name")

        df_vs = df_wide.merge(patient_subset, on=merge_keys, how="left")

    # Ensure required columns exist
    for c in ["Tx_started_date", "BC", "Outcome_PD"]:
        if c not in df_vs.columns:
            df_vs[c] = ""

    # --- Format all date-like columns as date only ---
    for col in df_vs.columns:
        if "date" in str(col).lower():
            dt = pd.to_datetime(df_vs[col], errors="coerce", dayfirst=True)
            df_vs[col] = dt.dt.strftime("%Y-%m-%d").fillna("")

    # --- Add Visit_Month_X columns ---
    if "Tx_started_date" in df_vs.columns:
        visit_date_cols = []
        for c in df_vs.columns:
            parts = str(c).rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit() and parts[0].lower() == "visit_date":
                visit_date_cols.append((int(parts[1]), c))
        visit_date_cols.sort(key=lambda x: x[0])

        for idx, visit_col in visit_date_cols:
            new_col = f"Visit_Month_{idx}"
            df_vs[new_col] = df_vs.apply(
                lambda r: month_diff_from_tx(r.get("Tx_started_date", ""), r.get(visit_col, ""), clamp_to_zero=True),
                axis=1
            ).astype("Int64")

    # -------------------- Outcome rules --------------------
    today = pd.Timestamp.today().normalize()

    # detect Visit_date_* columns for outcome
    visit_date_cols = []
    for c in df_vs.columns:
        parts = str(c).rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit() and parts[0].lower() == "visit_date":
            visit_date_cols.append((int(parts[1]), c))
    visit_date_cols.sort(key=lambda x: x[0])

    def compute_outcome(row):
        tx = pd.to_datetime(row.get("Tx_started_date", ""), errors="coerce", dayfirst=True)
        if pd.isna(tx):
            return ""

        bc_flag = _is_one_flag(row.get("BC", ""))

        visits = []
        for idx, vd_col in visit_date_cols:
            vd = pd.to_datetime(row.get(vd_col, ""), errors="coerce", dayfirst=True)
            if pd.isna(vd):
                continue
            m = month_diff_from_tx(tx, vd, clamp_to_zero=False)
            sp_col = f"Sputum_Result_{idx}"
            sp_val = row.get(sp_col, "")
            visits.append((m, vd, sp_val))

        months_since_tx = month_diff_from_tx(tx, today, clamp_to_zero=False)
        months_since_tx_int = None
        if months_since_tx is not pd.NA:
            try:
                months_since_tx_int = int(months_since_tx)
            except Exception:
                months_since_tx_int = None

        if not visits:
            if months_since_tx_int is not None and months_since_tx_int >= 6:
                return "LTFU"
            return ""

        visit_months_nonneg = sorted({m for (m, _, _) in visits if isinstance(m, (int, np.integer)) and m >= 0})
        max_month = max(visit_months_nonneg) if visit_months_nonneg else -999
        has_6m_or_more_visit = (max_month >= 6)

        # 1) Treatment failed: any sputum positive at month >=5
        for (m, _, sp) in visits:
            if isinstance(m, (int, np.integer)) and m >= 5 and _is_positive_sputum(sp):
                return "Treatment failed"

        # 2) Cure: >=6m visit, BC==1, 2 NEG after >=2m, >=7 days apart
        if has_6m_or_more_visit and bc_flag:
            neg_dates = []
            for (m, vd, sp) in visits:
                if isinstance(m, (int, np.integer)) and m >= 2 and _is_negative_sputum(sp):
                    neg_dates.append(vd)
            neg_dates = sorted(neg_dates)
            for i in range(len(neg_dates)):
                for j in range(i + 1, len(neg_dates)):
                    if (neg_dates[j] - neg_dates[i]).days >= 7:
                        return "Cure"

        # 3) LTFU: today >=6m and missing visits for two consecutive months, and no >=6m visit
        if months_since_tx_int is not None and months_since_tx_int >= 6 and not has_6m_or_more_visit:
            present = set(visit_months_nonneg)
            upper = min(6, months_since_tx_int)
            for m in range(0, upper):
                if (m not in present) and ((m + 1) not in present):
                    return "LTFU"

        # 4) Complete: has visit at >=6m and not failed/cure/ltfu
        if has_6m_or_more_visit:
            return "Complete"

        return ""

    df_vs["Outcome"] = df_vs.apply(compute_outcome, axis=1)
    # ------------------------------------------------------

    # --- Column ordering (Township in front of Registration_number) ---
    drop_bases = {"sn", "tb_or_tpt", "lab_number"}

    base_cols = []
    if "Township" in df_vs.columns:
        base_cols.append("Township")

    base_cols.append("Registration_number")

    if "Patient_name" in df_vs.columns:
        base_cols.append("Patient_name")

    base_cols += ["Tx_started_date", "BC", "Outcome_PD"]

    df_vs["Outcome"] = df_vs.get("Outcome", "")

    exclude_cols = set(base_cols + ["Outcome"])
    visit_cols = [
        c for c in df_vs.columns
        if c not in exclude_cols
        and all(not str(c).lower().startswith(b) for b in drop_bases)
    ]

    ordered = base_cols + visit_cols + ["Outcome"]
    df_vs = df_vs.loc[:, [c for c in ordered if c in df_vs.columns]]

    # SN first
    df_vs.insert(0, "SN", range(1, len(df_vs) + 1))
    return df_vs

# ==================== Cleaning (date only) ====================

def clean_dates_and_sn(df):
    """
    Format any column whose name contains 'date' OR is datetime dtype to YYYY-MM-DD (no time).
    """
    if df is None or df.empty:
        return df

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s) or ("date" in str(col).lower()):
            dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
            df[col] = dt.dt.strftime("%Y-%m-%d").fillna("")
    return df


# ==================== Main Rule Check Function ====================

def check_rules(excel_file, output_file="output.xlsx"):
    xls = pd.ExcelFile(excel_file)
    sheets = xls.sheet_names
    get_sheet = lambda name: xls.parse(name) if name in sheets else pd.DataFrame()

    df_screen = get_sheet("Screening")
    df_patient = get_sheet("Patient data")
    df_service = get_sheet("Service Point")
    df_visit = get_sheet("Visit data")
    df_dropdown = get_sheet("Dropdown")

    state_regions, state_township_dict, var_values_dict = (
        load_dropdowns(df_dropdown) if not df_dropdown.empty else (set(), {}, {})
    )

    allowed_registration_numbers = (
        set(df_patient["Registration_number"].apply(normalize))
        if "Registration_number" in df_patient.columns
        else set()
    )

    # -------------------- Screening --------------------
    df_screen = check_state_region(df_screen, state_regions)
    df_screen = check_township(df_screen, state_township_dict)
    df_screen = check_service_delivery_point(df_screen, df_service)
    df_screen = add_level_from_service_point(df_screen, df_service)

    # Reporting_Month check removed (as requested)

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

    df_screen = check_symptom_yes_no(df_screen)
    df_screen = check_registration_number(df_screen, allowed_registration_numbers)

    df_screen = add_presumptive_tb_referred(df_screen)
    df_screen = add_tb_detected(df_screen)
    df_screen = add_bact_confirmed_tb(df_screen)
    df_screen = add_result_check(df_screen)

    df_screen = add_duplicate_check(
        df_screen,
        ["Service_delivery_point", "Name", "Age_Year", "Sex", "Screening_Date"]
    )

    df_screen = add_ongoing_tb_case(df_screen, df_patient)

    # -------------------- Patient data --------------------
    df_patient = check_state_region(df_patient, state_regions)
    df_patient = check_township(df_patient, state_township_dict)
    df_patient = check_service_delivery_point(df_patient, df_service)
    df_patient = add_level_from_service_point(df_patient, df_service)

    for col in [
        "Transfer_in", "HIV_status", "TB_Type_of_patient",
        "TB_Type_of_Disease", "TB_Treatment_Regimen", "TB_Treatment_Outcome",
        "TPT_Treatment_Regimen"
    ]:
        df_patient = check_value_dropdown(df_patient, col, var_values_dict)

    for col in ["TB_Treatment_Outcome_Date", "TPT_Start_date", "TPT_End_date", "HIV_testing_date"]:
        df_patient = check_date_col(df_patient, col)

    df_patient = check_enrolled_date(df_patient)
    df_patient = check_tb_treatment_start_date(df_patient)
    df_patient = check_tb_treatment_outcome_date(df_patient)
    df_patient = check_tpt_end_date_rule(df_patient)

    df_patient = check_registration_number_duplicate(df_patient)
    df_patient = check_age_year(df_patient)
    df_patient = check_sex(df_patient, var_values_dict)
    df_patient = check_transfer_in_date(df_patient)

    df_patient = add_TBDT_1(df_patient)
    df_patient = add_TBHIV_5(df_patient)
    df_patient = add_TBP_1(df_patient)       # ✅ updated rule includes Age<=5
    df_patient = add_TBO2a_N(df_patient)

    df_patient = merge_screening_columns(df_patient, df_screen)
    df_patient = add_TBDT_3c(df_patient)
    df_patient = add_patient_checks(df_patient, df_screen)

    # -------------------- Visit data --------------------
    if "Visit_date" in df_visit.columns:
        df_visit = check_date_col(df_visit, "Visit_date")

    for col in ["Sputum_Result", "Gene_Xpert_Result", "Truenet_Result"]:
        df_visit = check_value_dropdown(df_visit, col, var_values_dict)

    # -------------------- Service Point --------------------
    df_service = check_state_region(df_service, state_regions)
    df_service = check_township(df_service, state_township_dict)

    # -------------------- VS_Update --------------------
    vs_update_df = create_vs_update(df_patient, df_visit)

    # -------------------- Clean dates (no time) --------------------
    df_patient = clean_dates_and_sn(df_patient)
    df_screen = clean_dates_and_sn(df_screen)
    df_service = clean_dates_and_sn(df_service)
    df_visit = clean_dates_and_sn(df_visit)
    vs_update_df = clean_dates_and_sn(vs_update_df)

    # -------------------- Combine errors into Comment --------------------
    def combine_errors(df, sheet_name=None):
        if df is None or df.empty:
            return df

        error_cols = [
            c for c in df.columns
            if str(c).endswith("_Error")
            or "Duplicate" in str(c)
            or str(c).endswith("_check")
            or str(c).endswith("_Check")
        ]

        # Patient Data special handling
        if sheet_name == "Patient Data":
            if "Tin/Refer_from_check" in df.columns:
                df["Tin_Refer_from"] = df["Tin/Refer_from_check"]
                df = df.drop(columns=["Tin/Refer_from_check"], errors="ignore")
                error_cols = [c for c in error_cols if c != "Tin/Refer_from_check"]

        if not error_cols:
            if "Comment" not in df.columns:
                df["Comment"] = ""
            return df

        def row_errors(row):
            msgs = []
            for col in error_cols:
                val = row.get(col)
                if pd.isna(val) or str(val).strip() == "":
                    continue

                # Screening: skip Result_check == T
                if sheet_name == "Screening" and col == "Result_check" and str(val).strip().upper() == "T":
                    continue

                clean_name = (
                    str(col).replace("_Error", "")
                    .replace("_error", "")
                    .replace("_Check", "")
                    .replace("_check", "")
                )
                msgs.append(f"{clean_name}: {val}")
            return "; ".join(msgs)

        df["Comment"] = df.apply(row_errors, axis=1)
        df = df.drop(columns=error_cols, errors="ignore")
        return df

    results = {
        "Screening": combine_errors(df_screen, "Screening"),
        "Patient Data": combine_errors(df_patient, "Patient Data"),
        "Service Point": combine_errors(df_service, "Service Point"),
        "Visit Data": combine_errors(df_visit, "Visit Data"),
        "VS_Update": combine_errors(vs_update_df, "VS_Update"),
    }

    # -------------------- Export --------------------
    with pd.ExcelWriter(output_file) as writer:
        for sheet_name, df in results.items():
            if isinstance(df, pd.DataFrame):
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    return results


excel_output.seek(0)


# ----------------- Streamlit UI -----------------
# ----------------- Streamlit UI -----------------
st.image("TB image2.jpg", width=200)  # TB logo/image
st.title("📊 IHRP: TB data verification & indicator calculation")

st.markdown("""
Upload your **Excel file** for TB data verification.  
The app will apply built-in rules and show validation results.  
You can download all results as a single Excel file with multiple sheets.
""")

# Upload Excel data file only
data_file = st.file_uploader("📂 Upload Excel file to verify", type=["xlsx"])

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

        # IMPORTANT: reset pointer
        excel_output.seek(0)

        if sheet_count > 0:
            st.download_button(
                label="⬇️ Download ALL Results as Excel (multi-sheet)",
                data=excel_output,
                file_name="all_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"❌ Error running rules: {e}")

st.markdown("---")
st.markdown("🩺 Created with Streamlit")
