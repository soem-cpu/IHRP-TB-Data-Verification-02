import streamlit as st
import pandas as pd
import importlib.util
import io

st.set_page_config(page_title="Dynamic Rule-Based Data Verification", layout="wide")

# Show TB image at top
st.image("TB image2.jpg", width=200)  # adjust width as needed

st.title("📊 IHRP: TB Data Verification App")

st.markdown("""
Upload your **Python rules file** and the **Excel file** you want to verify.
The app will dynamically apply the rules and show validation tables.
You can download all results as a single Excel file with multiple sheets.
""")

# Upload rules file (.py)
rules_file = st.file_uploader("Upload your Python rules file (.py)", type=["py"])
if rules_file:
    with open("rules_temp.py", "wb") as f:
        f.write(rules_file.getbuffer())
    spec = importlib.util.spec_from_file_location("rules_module", "rules_temp.py")
    rules_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules_module)
    st.success("✅ Rules file loaded!")

# Upload Excel data file
data_file = st.file_uploader("Upload Excel file to verify", type=["xlsx", "csv"])
if data_file and rules_file:


    # Apply rules
    try:
        results = rules_module.check_rules(data_file)
        excel_output = io.BytesIO()
        sheet_count = 0

        # Prepare multi-sheet Excel file
        with pd.ExcelWriter(excel_output, engine='xlsxwriter') as writer:
            if isinstance(results, dict):
                st.markdown("## Validation Results:")
                for k, v in results.items():
                    st.write(f"**{k}**")
                    if isinstance(v, pd.DataFrame):
                        if not v.empty:
                            st.dataframe(v)
                        else:
                            st.success(f"No issues found in {k}!")
                        v.to_excel(writer, index=False, sheet_name=k[:31])  # Excel sheet names max 31 chars
                        sheet_count += 1
                    else:
                        st.write(v)
            elif isinstance(results, pd.DataFrame):
                if results.empty:
                    st.success("✅ No validation issues found!")
                else:
                    st.write("Validation results:")
                    st.dataframe(results)
                results.to_excel(writer, index=False, sheet_name="Validation")
                sheet_count += 1
            else:
                st.write(results)

        if sheet_count > 0:
            st.download_button(
                label="Download ALL Results as Excel (multi-sheet)",
                data=excel_output.getvalue(),
                file_name="all_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    except Exception as e:
        st.error(f"Error running rules: {e}")

st.markdown("---")
st.markdown("Created with Streamlit")
