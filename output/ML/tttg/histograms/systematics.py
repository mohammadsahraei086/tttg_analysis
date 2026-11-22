"""Scale / PDF / ISR / FSR systematic uncertainties from the varied signal histograms."""
import numpy as np
import hist

from typing import Dict


def compute_systematic_uncertainties(signal: str, hist_dict: Dict) -> Dict:
    """
    Compute MUR, MUF, and PDF uncertainties from histograms.
    
    Parameters:
    -----------
    hist_dict : dict
        Dictionary containing histograms with variations (from your coffea output)
    
    Returns:
    --------
    dict: Dictionary containing uncertainties for each sample
    """
    
    uncertainties = {}
    
    # Define the variations we need
    nominal_variation = 'Nominal'
    mur_up = 'MUF=1.0_MUR=2.0_PDF=247000'
    mur_down = 'MUF=1.0_MUR=0.5_PDF=247000'
    muf_up = 'MUF=2.0_MUR=1.0_PDF=247000'
    muf_down = 'MUF=0.5_MUR=1.0_PDF=247000'

    nominal_isr_fsr = "Nominal_isrMuR_fsrMuR"
    fsr_down = "fsrMuRDown"
    fsr_up = "fsrMuRUp"
    isr_down = "isrMuRDown"
    isr_up = "isrMuRUp"
    
    # PDF variations (100 eigenvariations from 247001 to 247100)
    pdf_variations = [f'MUF=1.0_MUR=1.0_PDF=247{i:03d}' for i in range(1, 101)]

    hist_obj = hist_dict[signal]
    
    if not isinstance(hist_obj, hist.Hist):
        raise TypeError(f"compute_syst needs hist object. {type(hist_obj)} is given")
            
    print(f"Processing {signal}...")
    nominal = hist_obj[{'variation': nominal_variation}].values()
    
    # 1. MUR uncertainty
    mur_up_vals = hist_obj[{'variation': mur_up}].values()
    mur_down_vals = hist_obj[{'variation': mur_down}].values()
            
    mur_unc_up = abs(mur_up_vals - nominal)
    mur_unc_down = abs(nominal - mur_down_vals)
    mur_unc = (mur_unc_up + mur_unc_down)/2
    
    # 2. MUF uncertainty
    muf_up_vals = hist_obj[{'variation': muf_up}].values()
    muf_down_vals = hist_obj[{'variation': muf_down}].values()
    
    muf_unc_up = abs(muf_up_vals - nominal)
    muf_unc_down = abs(nominal - muf_down_vals)
    muf_unc = (muf_unc_up + muf_unc_down)/2

    nominal_isr_fsr = hist_obj[{'variation': nominal_isr_fsr}].values()
    
    # 1. isr uncertainty
    isr_up_vals = hist_obj[{'variation': isr_up}].values()
    isr_down_vals = hist_obj[{'variation': isr_down}].values()
            
    isr_unc_up = abs(isr_up_vals - nominal_isr_fsr )
    isr_unc_down = abs(nominal_isr_fsr  - isr_down_vals)
    isr_unc = (isr_unc_up + isr_unc_down)/2
    
    # 2. fsr uncertainty
    fsr_up_vals = hist_obj[{'variation': fsr_up}].values()
    fsr_down_vals = hist_obj[{'variation': fsr_down}].values()
    
    fsr_unc_up = abs(fsr_up_vals - nominal_isr_fsr)
    fsr_unc_down = abs(nominal_isr_fsr - fsr_down_vals)
    fsr_unc = (fsr_unc_up + fsr_unc_down)/2
    
    # 3. PDF uncertainty (RMS of 100 variations)
    pdf_values = []
    for var in pdf_variations:
        try:
            values = hist_obj[{'variation': var}].values()
            pdf_values.append(values)
        except KeyError:
            continue
    
    if len(pdf_values) > 0:
        pdf_values = np.array(pdf_values)
        # RMS for each bin across all variations
        pdf_rms = np.sqrt(np.mean((pdf_values - nominal)**2, axis=0))
        
        # Symmetric PDF uncertainty (using RMS)
        pdf_unc = pdf_rms  # same for up and down
    else:
        pdf_unc = np.zeros_like(nominal)
    
    total = np.sqrt(mur_unc**2 + muf_unc**2 + isr_unc**2 + fsr_unc**2 + pdf_unc**2)
    
    def _rel(x, ref):
        return np.divide(x, ref, out=np.zeros_like(x, dtype=float), where=ref > 0)
    
    uncertainties = {
        'nominal': nominal,
        "nominal_isr_fsr": nominal_isr_fsr,
        "mur_unc": _rel(mur_unc, nominal),
        "muf_unc": _rel(muf_unc, nominal),
        "isr_unc": _rel(isr_unc, nominal_isr_fsr),
        "fsr_unc": _rel(fsr_unc, nominal_isr_fsr),
        "pdf_unc": _rel(pdf_unc, nominal),
        "total":   _rel(total, nominal),
        'bin_edges': hist_obj.axes[0].edges.tolist(),
    }
    return uncertainties
