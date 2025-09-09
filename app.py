import streamlit as st
import os
import json
import re
import io
import PyPDF2
import camelot
from datetime import datetime
from fpdf import FPDF
import uuid
import time
import tempfile
from transformers import pipeline

class PDFProcessor:
    """Simplified PDF processor for digital PDFs only"""
    
    def __init__(self, output_dir: str = "extracted_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Updated parameter keywords for food analysis
        self.parameter_keywords = {
            "aflatoxin_B1": ["aflatoxin b1", "aflatoxin b₁"],
            "total_aflatoxins": ["total aflatoxins"],
            "fumonisin_B1": ["fumonisin b1"],
            "zearalenone": ["zearalenone"],
            "salmonella": ["salmonella"],
            "e_coli": ["e. coli", "coli"],
            "total_coliforms": ["total coliforms"],
        }

        # Pre-compile a regex for numeric value extraction
        units = r"µg/kg|ppb|mg/kg|ppm|cfu/g|absence|present"
        self.value_re = re.compile(rf"(\d+\.?\d*)\s*({units})", re.IGNORECASE)

    def _contains_keyword(self, text: str, param: str) -> bool:
        """Case-insensitive check if any keyword for *param* occurs in *text*."""
        lower = text.lower()
        return any(k in lower for k in self.parameter_keywords[param])

    def _all_params_found(self, found: set) -> bool:
        return len(found) == len(self.parameter_keywords)

    def _extract_from_digital(self, pdf_bytes: bytes):
        """Extract data from digital PDF using PyPDF2 and Camelot"""
        out = {"text": "", "tables": []}
        found_params = set()

        # Extract text using PyPDF2
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages:
                text = page.extract_text() or ""
                for param in self.parameter_keywords:
                    if (param not in found_params and self._contains_keyword(text, param)):
                        out["text"] += text + "\n"
                        found_params.add(param)
                if self._all_params_found(found_params):
                    break
        except Exception as e:
            st.warning(f"PyPDF2 text extraction error: {e}")

        # Extract tables using Camelot
        try:
            # Write PDF to temporary file for Camelot
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_bytes)
                tmp_path = tmp_file.name

            tables = camelot.read_pdf(tmp_path, flavor="stream", pages="all")
            os.unlink(tmp_path)  # Clean up temp file

            for i, tbl in enumerate(tables):
                tbl_str = tbl.df.to_string().lower()
                keep = False
                for param in self.parameter_keywords:
                    if (param not in found_params and 
                        any(k in tbl_str for k in self.parameter_keywords[param])):
                        keep = True
                        found_params.add(param)
                if keep:
                    out["tables"].append({
                        "table_id": f"table{i+1}", 
                        "data": tbl.df.values.tolist()
                    })
                if self._all_params_found(found_params):
                    break
        except Exception as e:
            st.warning(f"Camelot table extraction error: {e}")

        return out

    def _extract_parameters(self, doc_data: dict):
        """Extract parameters from document data"""
        params = {}
        txt = doc_data.get("text", "").lower()

        # From text context
        for param in self.parameter_keywords:
            if any(k in txt for k in self.parameter_keywords[param]):
                params.setdefault(param, {"contexts": [], "raw_values": []})
                # Capture ±100 chars around each hit
                for m in re.finditer("|".join(self.parameter_keywords[param]), txt):
                    start = max(m.start() - 100, 0)
                    end = min(m.end() + 100, len(txt))
                    params[param]["contexts"].append(txt[start:end])

        # From tables
        for tbl in doc_data.get("tables", []):
            tbl_str = str(tbl["data"]).lower()
            for param in self.parameter_keywords:
                if any(k in tbl_str for k in self.parameter_keywords[param]):
                    params.setdefault(param, {"contexts": [], "raw_values": []})
                    params[param]["raw_values"].append(tbl["data"])

        # Numeric value extraction
        for param, dat in params.items():
            vals = []
            for raw in dat.get("raw_values", []) + dat.get("contexts", []):
                for val, unit in self.value_re.findall(str(raw)):
                    vals.append(f"{val} {unit}")
            if vals:
                dat["values"] = vals

        return params

    def process_pdf(self, pdf_bytes):
        """Process PDF bytes and extract parameters"""
        st.info("Processing digital PDF...")
        
        doc_data = self._extract_from_digital(pdf_bytes)
        doc_data["parameters"] = self._extract_parameters(doc_data)
        
        return doc_data


class DocumentVerifier:
    def __init__(self):
        """Initialize the document verifier with food safety standards"""
        self.standard = {
  "aflatoxin_B1": "not more than 15 µg/kg in raw peanuts; not more than 5 µg/kg in dairy cattle feed; for human foods generally part of total aflatoxin limit (≤ 10 µg/kg in ready-to-eat foods in some Codex standards)",
  "total_aflatoxins": "not more than 20 µg/kg in foods (Codex CXS 193-1995); stricter limits (10 µg/kg) for certain commodities such as ready-to-eat nuts and dried fruits",
  "fumonisin_B1": "not more than 2,000 µg/kg (2 mg/kg) in raw maize; not more than 1,000 µg/kg (1 mg/kg) in maize-based foods for direct human consumption",
  "zearalenone": "not more than 100 µg/kg in unprocessed cereals other than maize; not more than 350 µg/kg in unprocessed maize; lower limits (50 µg/kg) for baby foods and cereal-based infant products",
  "salmonella": "absence in 25 g of product (n=5, c=0, m=0) for ready-to-eat foods, dairy, infant formula, spices, poultry; presence in any 25 g renders product unacceptable",
  "e_coli": "not more than 10 CFU/g in ready-to-eat foods; absence in 1 g for powdered infant formula and follow-up formula intended for infants under 6 months",
  "total_coliforms": "generally not more than 100 CFU/g in ready-to-eat foods; absence in 1 g required for powdered infant formula"
}

        
        self.nli_pipeline = None
        try:
            # Initialize NLI pipeline with progress bar
            with st.spinner("Loading AI verification model..."):
                self.nli_pipeline = pipeline("zero-shot-classification", 
                                           model="facebook/bart-large-mnli")
            st.success("AI model loaded successfully!")
        except Exception as e:
            st.warning(f"AI model failed to load: {e}. Using fallback verification.")

    def _extract_numeric_values(self, text):
        """Extract numerical values with their units from text"""
        patterns = [
            r'(\d+\.?\d*)\s*(µg/kg|ppb)',  # For mycotoxins
            r'(\d+\.?\d*)\s*(mg/kg|ppm)',  # General contaminants
            r'(\d+\.?\d*)\s*(cfu/g|cfu/ml)',  # For bacteria
            r'(absence|present)\s*(in\s*\d+\s*g)?',  # For qualitative tests
            r'(\d+\.?\d*)\s?([a-zA-Z]+/[a-zA-Z]+|[a-zA-Z]+)?',  # Generic pattern
        ]
        
        values = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 1:
                    value = match.group(1)
                    unit = match.group(2) if len(match.groups()) > 1 and match.group(2) else ""
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(text), match.end() + 50)
                    context = text[context_start:context_end]
                    values.append({
                        "value": value,
                        "unit": unit,
                        "context": context
                    })
        return values

    def _find_standard_value(self, param_name):
        """Extract the standard value and requirement from the standard"""
        if param_name not in self.standard:
            return None, None
        
        standard_text = self.standard.get(param_name, "")
        values = self._extract_numeric_values(standard_text)
        
        if values:
            standard_value = values[0]["value"]
            if values[0]["unit"]:
                standard_value += " " + values[0]["unit"]
            
            requirement_type = "unknown"
            lower_text = standard_text.lower()
            if "not more than" in lower_text or "maximum" in lower_text:
                requirement_type = "maximum"
            elif "not less than" in lower_text or "minimum" in lower_text:
                requirement_type = "minimum"
            elif "absence" in lower_text:
                requirement_type = "absence"
            
            return standard_value, requirement_type
        
        return None, None

    def _verify_parameter_with_values(self, param_name, extracted_data, standard_data):
        """Verify if a parameter meets the standard"""
        if param_name not in extracted_data:
            return False, "Parameter not found in document", [], None
        
        standard_text = standard_data.get(param_name, "")
        if not standard_text:
            return False, "Standard requirement not available", [], None
        
        # Get all data for this parameter
        contexts = extracted_data[param_name].get("contexts", [])
        raw_values = extracted_data[param_name].get("raw_values", [])
        
        combined_text = " ".join(contexts + [str(rv) for rv in raw_values])
        extracted_values = self._extract_numeric_values(combined_text)
        standard_value, requirement_type = self._find_standard_value(param_name)
        
        # Use NLI model or fallback
        if not self.nli_pipeline:
            compliant, message = self._fallback_verification(param_name, combined_text, standard_text)
        else:
            try:
                compliance_label = f"This food sample complies with the {param_name} safety standard"
                non_compliance_label = f"This food sample does not comply with the {param_name} safety standard"
                
                result = self.nli_pipeline(
                    combined_text,
                    [compliance_label, non_compliance_label],
                    multi_label=False
                )
                
                compliance_idx = result['labels'].index(compliance_label)
                compliance_score = result['scores'][compliance_idx]
                
                compliant = compliance_score > 0.5
                message = f"AI Confidence: {compliance_score:.2f}"
                
            except Exception as e:
                st.warning(f"AI inference error: {e}")
                compliant, message = self._fallback_verification(param_name, combined_text, standard_text)
        
        return compliant, message, extracted_values, standard_value

    def _fallback_verification(self, param_name, extracted_text, standard_text):
        """Simple fallback verification"""
        extracted_lower = extracted_text.lower()
        standard_lower = standard_text.lower()
        
        # Check for compliance keywords
        keywords = ["compliant", "meets", "standard", "acceptable", "within", "below limit", "absence"]
        if any(keyword in extracted_lower for keyword in keywords):
            return True, "Fallback: Found compliance indicator"
        
        # Check for non-compliance keywords
        fail_keywords = ["exceeds", "above limit", "non-compliant", "detected", "present"]
        if any(keyword in extracted_lower for keyword in fail_keywords):
            return False, "Fallback: Found non-compliance indicator"
        
        return True, "Fallback: Unable to verify definitively"

    def verify_parameters(self, extracted_params):
        """Verify if extracted parameters meet the standard requirements"""
        verification_results = {}
        compliant_count = 0
        parameters_checked = 0

        for param_name in extracted_params.keys():
            if param_name in self.standard:
                try:
                    compliant, message, extracted_values, standard_value = self._verify_parameter_with_values(
                        param_name, extracted_params, self.standard
                    )

                    verification_results[param_name] = {
                        "compliant": compliant,
                        "message": message,
                        "extracted_values": extracted_values,
                        "standard_value": standard_value
                    }

                    parameters_checked += 1
                    if compliant:
                        compliant_count += 1
                        
                except Exception as e:
                    st.error(f"Error verifying parameter {param_name}: {e}")
                    verification_results[param_name] = {
                        "compliant": False,
                        "message": f"Error: {str(e)}",
                        "extracted_values": [],
                        "standard_value": None
                    }

        if parameters_checked == 0:
            verification_results["no_parameters"] = {
                "compliant": False,
                "message": "No matching parameters found",
                "extracted_values": [],
                "standard_value": None
            }
            parameters_checked = 1

        # At least 2 parameters must be verified for food safety
        if parameters_checked < 2:
            overall_compliant = False
            compliance_reason = f"Only {parameters_checked} parameters verified (minimum 2 required)"
        else:
            compliance_threshold = 0.80  # 80% compliance required
            overall_compliant = (compliant_count / parameters_checked) >= compliance_threshold
            compliance_reason = f"{compliant_count} out of {parameters_checked} parameters compliant"

        return {
            "overall_compliant": overall_compliant,
            "compliance_reason": compliance_reason,
            "parameter_results": verification_results,
            "parameters_checked": parameters_checked
        }


class CertificateGenerator:
    def __init__(self, output_dir="certificates"):
        """Initialize the certificate generator"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_certificate(self, document_name, verification_results):
        """Generate a PDF certificate for a compliant document"""
        if not verification_results.get("overall_compliant", False):
            return None
        
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font("Arial", "B", 24)
        pdf.cell(0, 20, "FOOD SAFETY CERTIFICATE", 0, 1, "C")
        
        # Certificate details
        pdf.set_font("Arial", "", 12)
        pdf.ln(10)
        pdf.cell(0, 10, f"Certificate ID: CERT-{uuid.uuid4().hex[:8].upper()}", 0, 1)
        pdf.cell(0, 10, f"Date of Issue: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
        pdf.cell(0, 10, f"Document: {document_name}", 0, 1)
        pdf.cell(0, 10, "Standard: Food Safety Standards", 0, 1)
        
        # Add line
        pdf.line(20, pdf.get_y() + 5, 190, pdf.get_y() + 5)
        pdf.ln(15)
        
        # Verification results
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Verification Results:", 0, 1)
        pdf.set_font("Arial", "", 12)
        pdf.ln(5)
        
        for param, result in verification_results.get("parameter_results", {}).items():
            if param != "no_parameters":
                status = "✓ PASS" if result.get("compliant", False) else "✗ FAIL"
                pdf.cell(0, 8, f"{param.replace('_', ' ').title()}: {status}", 0, 1)
        
        # Overall status
        pdf.ln(10)
        pdf.set_font("Arial", "B", 16)
        pdf.set_text_color(0, 128, 0)  # Green color
        pdf.cell(0, 15, "CERTIFICATE: COMPLIANT", 0, 1, "C")
        
        # Footer
        pdf.set_y(-30)
        pdf.set_font("Arial", "I", 8)
        pdf.set_text_color(0, 0, 0)  # Black color
        pdf.cell(0, 10, "This certificate was automatically generated by the Food Safety Verification System.", 0, 1, "C")
        pdf.cell(0, 10, f"Verification completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, "C")
        
        # Save the PDF
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.output_dir}/Food_Safety_Certificate_{timestamp}.pdf"
        pdf.output(filename)
        
        return filename


def main():
    st.set_page_config(
        page_title="Food Safety Analysis System",
        page_icon="🍯",
        layout="wide"
    )
    
    st.title("🍯 Food Safety Analysis System")
    st.markdown("Upload a food analysis PDF to verify compliance with safety standards")
    
    # Sidebar for information
    with st.sidebar:

        st.header("ℹ️ How it works")
        st.markdown("""
        1. Upload your food analysis PDF
        2. AI extracts parameters from the document
        3. Values are verified against safety standards
        4. Get compliance results and certificate
        """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type="pdf",
        help="Upload a food analysis report in PDF format"
    )
    
    if uploaded_file is not None:
        # Process the PDF
        with st.spinner("Processing PDF..."):
            processor = PDFProcessor()
            pdf_bytes = uploaded_file.read()
            doc_data = processor.process_pdf(pdf_bytes)
        
        # Display extracted parameters
        if doc_data.get("parameters"):
            st.success("✅ Parameters extracted successfully!")
            
            with st.expander("📊 Extracted Parameters", expanded=True):
                for param, data in doc_data["parameters"].items():
                    st.write(f"**{param.replace('_', ' ').title()}**")
                    if data.get("values"):
                        st.write(f"Values found: {', '.join(data['values'])}")
                    else:
                        st.write("Parameter detected but values need manual review")
        else:
            st.warning("⚠️ No relevant parameters found in the PDF")
            return
        
        # Verify against standards
        with st.spinner("Verifying against food safety standards..."):
            verifier = DocumentVerifier()
            verification_results = verifier.verify_parameters(doc_data["parameters"])
        
        # Display results
        st.header("🔍 Verification Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if verification_results["overall_compliant"]:
                st.success("✅ **COMPLIANT**")
                st.balloons()
            else:
                st.error("❌ **NON-COMPLIANT**")
            
            st.write(f"**Reason**: {verification_results['compliance_reason']}")
        
        with col2:
            st.metric(
                "Parameters Checked", 
                verification_results["parameters_checked"]
            )
        
        # Detailed results
        with st.expander("📋 Detailed Results", expanded=True):
            for param, result in verification_results["parameter_results"].items():
                if param != "no_parameters":
                    col_param, col_status, col_details = st.columns([2, 1, 2])
                    
                    with col_param:
                        st.write(f"**{param.replace('_', ' ').title()}**")
                    
                    with col_status:
                        if result["compliant"]:
                            st.success("✅ PASS")
                        else:
                            st.error("❌ FAIL")
                    
                    with col_details:
                        st.write(f"*{result['message']}*")
                        if result.get("standard_value"):
                            st.write(f"Standard: {result['standard_value']}")
        
        # Generate certificate if compliant
        if verification_results["overall_compliant"]:
            st.header("📜 Certificate Generation")
            
            if st.button("Generate Compliance Certificate", type="primary"):
                with st.spinner("Generating certificate..."):
                    generator = CertificateGenerator()
                    cert_path = generator.generate_certificate(
                        uploaded_file.name,
                        verification_results
                    )
                
                if cert_path:
                    st.success("Certificate generated successfully!")
                    
                    # Provide download button
                    with open(cert_path, "rb") as cert_file:
                        st.download_button(
                            label="📥 Download Certificate",
                            data=cert_file.read(),
                            file_name=f"Food_Safety_Certificate_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
        else:
            st.info("📝 Certificate can only be generated for compliant documents")


if __name__ == "__main__":
    main()